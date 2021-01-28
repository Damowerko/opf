from functools import reduce
from typing import List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn

import alegnn.utils.graphML as gml
from alegnn.modules.architectures import SelectionGNN
from pyopf.complex import ComplexPolar, ComplexRect
from pyopf.power import NetworkManager


class LocalGNN(SelectionGNN):
    """
    LocalGNN: All operations are local, and the output is extracted at a single
    node. Note that a last layer MLP, applied to the features of each node,
    being the same for all nodes, is equivalent to an LSIGF.

    THINGS TO DO:
        - Is the adding an extra feature the best way of doing this?
        - Should I separate Local MLP from LSIGF? At least in the inputs for
          the initialization?
        - Is this class necessary at all?
        - How would I do pooling? If I do pooling, this might affect the
          labeling/ordering of the nodes. And I would need to ensure that the
          nodes where I want to take the output from where selected during
          pooling. So, no pooling for now.
        - I also don't like the idea of having a splitForward() as well.

    There is no coarsening, nor MLP because these two operations kill the
    locality. So, only local operations are included.
    """

    def __init__(self,
                 # Graph Filtering,
                 dimNodeSignals=None, nFilterTaps=None, bias=None,
                 # Nonlinearity,
                 nonlinearity=torch.nn.ReLU,
                 # Structure
                 GSO=None,
                 index: List[int] = None):

        # We need to compute the values of nSelectedNodes, and poolingSize
        # so that there is no pooling.

        # First, check the inputs

        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]])  # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2]  # E x N x N

        # Get the number of layers
        self.L = len(nFilterTaps)
        # Get the number of selected nodes so that there is no pooling
        nSelectedNodes = [GSO.shape[1]] * self.L
        # Define the no pooling function
        poolingFunction = gml.NoPool
        # And the pooling size, which is one (it doesn't matter)
        poolingSize = [1] * self.L

        self.index = index

        super().__init__(dimNodeSignals, nFilterTaps, bias,
                         nonlinearity,
                         nSelectedNodes, poolingFunction, poolingSize,
                         [],
                         GSO)

    def forward(self, x):
        x = super().forward(x)
        index = torch.tensor(self.index, dtype=torch.int64, device=x.device)
        x = torch.index_select(x, 1, index)
        return x


# noinspection PyAttributeOutsideInit,PyPep8Naming
class OPFLogBarrier(pl.LightningModule):
    def __init__(self, manager: NetworkManager, gnn, t=10, s=1000, type="log",
                 equality_threshold=0.001, cost_weight=1.0):
        super().__init__()
        self.manager = manager
        self.pm = self.manager.powermodels_data()
        self.gnn = gnn
        self.t = t
        self.s = s
        self.cost_weight = cost_weight
        self.type = type
        self.equality_threshold = equality_threshold

        self.n_bus = len(self.pm["bus"])
        self.n_branch = len(self.pm["branch"])
        self.init_gen()
        self.init_cost_coefficients()
        self.init_bus()
        self.init_branch()
        self.init_shunt()

    def loss(self, cost, constraints):
        constraint_losses = [val["loss"] for val in constraints.values() if
                             val["loss"] is not None and not val["loss"].isnan()]
        return cost * self.cost_weight + torch.stack(constraint_losses).mean()

    def metrics(self, cost, constraints, prefix):
        metrics = {f"{prefix}_cost": cost, f"{prefix}_equality_loss": 0, f"{prefix}_inequality_loss": 0}
        for constraint_name, constraint_values in constraints.items():
            constraint_type = constraint_name.split("_")[0]
            for value_name, value in constraint_values.items():
                metrics[f"{prefix}_{constraint_name}_{value_name}"] = value
                aggregate_name = f"{prefix}_{constraint_type}_{value_name}"
                if not torch.isnan(value):
                    if aggregate_name in metrics:
                        metrics[aggregate_name] += value
                    else:
                        metrics[aggregate_name] = value.clone().detach()
        # cast all values to tensor
        return {k: v if torch.is_tensor(v) else torch.as_tensor(v) for k, v in metrics.items()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        self.gnn.to(x.dtype).to(x.device)
        x = self.gnn(x)
        x = torch.reshape(x, (-1, 4, self.n_bus))
        p_load = load[:, :, 0]
        q_load = load[:, :, 1]
        return x, p_load, q_load

    def training_step(self, x, *args, **kwargs):
        cost, constraints = self.optimal_power_flow(*self(x[0]))
        loss = self.loss(cost, constraints)
        self.log_dict(self.metrics(cost, constraints, "train"))
        return loss

    def validation_step(self, x, *args, **kwargs):
        cost, constraints = self.optimal_power_flow(*self(x[0]))
        self.log_dict(self.metrics(cost, constraints, "val"))
        return cost

    def test_step(self, x, *args, **kwargs):
        cost, constraints = self.optimal_power_flow(*self(x[0]))
        self.log_dict(self.metrics(cost, constraints, "test"))
        return cost

    def optimal_power_flow(self, x, p_load: torch.Tensor, q_load: torch.Tensor):
        """
        Find the cost function using the log-barrier relaxation of the ACOPF problem.

        Reference:
            https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
            https://matpower.org/docs/MATPOWER-manual.pdf
        """

        assert x.shape[0] == p_load.shape[0] == q_load.shape[0]
        assert x.shape[1] == 4
        assert x.shape[2] == self.n_bus
        assert p_load.shape[1] == self.n_bus
        assert q_load.shape[1] == self.n_bus

        vm = x[:, 0, :]
        va = x[:, 1, :]
        p = x[:, 2, :]
        q = x[:, 3, :]

        V = ComplexPolar(vm, va).unsqueeze(-1)  # network will learn rectangular voltages
        S = ComplexRect(p, q).unsqueeze(-1)

        # Convert voltage and power to per unit
        V /= self.base_kv
        S /= self.base_mva
        p_load /= self.base_mva
        q_load /= self.base_mva

        Sd = ComplexRect(p_load, q_load).unsqueeze(-1)
        Sg = S + Sd
        If = self.Yf.matmul(V)  # Current from
        It = self.Yt.matmul(V)  # Current to
        Sf = self.Cf.matmul(V).squeeze(-1).diag() \
            .matmul(If.conj())  # [Cf V] If* = Power from branch
        St = self.Ct.matmul(V).squeeze(-1).diag() \
            .matmul(It.conj())  # [Cf V] It* = Power to branch
        Sbus_sh = V.squeeze(-1).diag().matmul(
            self.Ybus_sh.matmul(V.conj()))  # [V][Ybus_shunt] V* = shunt bus power
        Sbus = ComplexPolar(self.Cf.T).matmul(Sf) + ComplexPolar(self.Cf.T).matmul(Sf) + Sbus_sh

        cost = self.cost(S.real, S.imag)
        constraints = self.constraints(S, Sbus, V.abs(), V.angle(), Sg, Sf, St)
        return cost, constraints

    def init_cost_coefficients(self):
        """A tuple of two 3xN matrices, representing the polynomial coefficients of active and reactive power cost."""
        element_types = ["gen", "sgen"]
        pcs, qcs = zip(*map(lambda et: self.manager.cost_coefficients(et), element_types))
        p_coeff = reduce(lambda x, y: x + y, pcs)
        q_coeff = reduce(lambda x, y: x + y, qcs)
        self.cost_coefficients = (torch.from_numpy(p_coeff).to(torch.float32).to(self.device),
                                  torch.from_numpy(q_coeff).to(torch.float32).to(self.device))

    def cost(self, p, q):
        """Compute the cost to produce the active and reactive power."""
        p_coeff, q_coeff = self.cost_coefficients  # get 3xN cost coefficient matrices
        p_coeff, q_coeff = p_coeff.T, q_coeff.T
        return (p_coeff[:, 0] + p * p_coeff[:, 1] + (p ** 2) * p_coeff[:, 2] +
                q_coeff[:, 0] + q * q_coeff[:, 1] + (q ** 2) * q_coeff[:, 2]).mean()

    def constraints(self, S, Sbus, vm, va, Sg, Sf, St) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Map from constraint name => (value name => tensor value)
        """
        Sg = Sg.to_rect()  # cast to rectangular coordinates
        vad = va.unsqueeze(2) - va.unsqueeze(1)  # voltage angle difference matrix
        constraints = {
            "equality_powerflow": self.equality(S - Sbus),
            "equality_reference": self.equality(va[self.reference_buses]),
            "equality_real_power": self.equality((self.p_min - Sg.real)[:, self.p_mask]),
            "inequality_real_power_min": self.inequality((self.p_min - Sg.real)[:, ~self.p_mask]),
            "inequality_real_power_max": self.inequality((Sg.real - self.p_max)[:, ~self.p_mask]),
            "equality_reactive_power": self.equality((self.q_min - Sg.imag)[:, self.q_mask]),
            "inequality_reactive_power_min": self.inequality((self.q_min - Sg.imag)[:, ~self.q_mask]),
            "inequality_reactive_power_max": self.inequality((Sg.imag - self.q_max)[:, ~self.q_mask]),
            "equality_voltage_magnitude": self.equality((self.vm_min - vm)[:, self.vm_mask]),
            "inequality_voltage_magnitude_min": self.inequality((self.vm_min - vm)[:, ~self.vm_mask]),
            "inequality_voltage_magnitude_max": self.inequality((vm - self.vm_max)[:, ~self.vm_mask]),
            "inequality_forward_rate_max": self.inequality(Sf.abs() - self.rate_a),
            "inequality_backward_rate_max": self.inequality(St.abs() - self.rate_a),
            "equality_voltage_angle_difference": self.equality((self.vad_min - vad)[:, self.vad_mask], angle=True),
            "inequality_voltage_angle_difference_min": self.inequality((self.vad_min - vad)[:, ~self.vad_mask],
                                                                       angle=True),
            "inequality_voltage_angle_difference_max": self.inequality((vad - self.vad_max)[:, ~self.vad_mask],
                                                                       angle=True)
        }
        return constraints

    def equality(self, u, angle=False):
        if angle:
            u = torch.fmod(u, 2 * np.pi)
            u[u > 2 * np.pi] = 2 * np.pi - u[u > 2 * np.pi]
        loss = u.abs().square().mean()
        return {"loss": loss}

    def inequality(self, u, angle=False):
        assert not ((u > 0) * u.isinf()).any()

        if angle:
            u = torch.fmod(u, 2 * np.pi)
            u[u > 2 * np.pi] = 2 * np.pi - u[u > 2 * np.pi]

        unconstrained = (u < 0) * u.isinf()
        violated = u >= 0
        violated_rate = torch.Tensor([float(violated.sum()) / float(u.numel())])
        violated_rms = u[violated].square().mean().sqrt() if violated_rate > 0 else 0

        loss = None
        if self.type == "log":
            u = u[~unconstrained * ~violated]
            if u.numel() != 0:
                log = -torch.log(-u) / self.t
                loss = log.mean()
        elif self.type == "relaxed_log":
            u = u[~unconstrained]
            if u.numel() != 0:
                threshold = - 1 / (self.s * self.t)
                below = u <= threshold
                log = (-torch.log(-u[below]) / self.t).mean()
                linear = ((-np.log(-threshold) / self.t) + (u[~below] - threshold) * self.s).mean()
                loss = (log if not log.isnan() else 0) + (linear if not linear.isnan() else 0)
        return dict(loss=torch.as_tensor(loss, dtype=self.dtype, device=self.device),
                    violated_rate=torch.as_tensor(violated_rate, dtype=self.dtype, device=self.device),
                    violated_rms=torch.as_tensor(violated_rms, dtype=self.dtype, device=self.device))

    def init_gen(self):
        self.p_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        self.p_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        self.q_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        self.q_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        for gen in self.pm["gen"].values():
            i = gen["gen_bus"] - 1
            self.p_min[i] = gen["pmin"]
            self.q_min[i] = gen["qmin"]
            self.p_max[i] = gen["pmax"]
            self.q_max[i] = gen["qmax"]
        self.p_mask = self.equality_threshold > (self.p_max - self.p_min).abs()
        self.q_mask = self.equality_threshold > (self.q_max - self.q_min).abs()

    def init_branch(self):
        self.rate_a = torch.full((self.n_branch, 1), float("inf"), device=self.device, dtype=torch.float32)
        self.vad_max = torch.full((self.n_bus, self.n_bus), float("inf"), device=self.device, dtype=torch.float32)
        self.vad_min = torch.full((self.n_bus, self.n_bus), -float("inf"), device=self.device, dtype=torch.float32)
        Yff = np.zeros((self.n_branch,), dtype=np.csingle)
        Yft = np.zeros((self.n_branch,), dtype=np.csingle)
        Ytf = np.zeros((self.n_branch,), dtype=np.csingle)
        Ytt = np.zeros((self.n_branch,), dtype=np.csingle)
        Cf = np.zeros((self.n_branch, self.n_bus), dtype=np.csingle)
        Ct = np.zeros((self.n_branch, self.n_bus), dtype=np.csingle)
        for branch in self.pm['branch'].values():
            index = branch["index"] - 1
            fr_bus = branch["f_bus"] - 1
            to_bus = branch["t_bus"] - 1
            y = 1 / (branch["br_r"] + 1j * branch["br_x"])
            yc_fr = branch["g_fr"] + 1j * branch["b_fr"]
            yc_to = branch["g_to"] + 1j * branch["b_to"]
            ratio = branch["tap"] * np.exp(1j * branch["shift"])
            Yff[index] = (y + yc_fr) / np.abs(ratio) ** 2
            Yft[index] = -y / np.conj(ratio)
            Ytt[index] = (y + yc_to)
            Ytf[index] = -y / ratio
            Cf[index, fr_bus] = 1
            Ct[index, to_bus] = 1
            self.rate_a[index] = branch["rate_a"]
            self.vad_min[fr_bus, to_bus] = branch["angmin"]
            self.vad_max[fr_bus, to_bus] = branch["angmax"]
        Yt = np.diag(Yff).dot(Cf) + np.diag(Yft).dot(Ct)
        Yf = np.diag(Ytf).dot(Cf) + np.diag(Ytt).dot(Ct)
        self.Ybus_branch = ComplexRect.from_numpy(Cf.T.dot(Yf) + Ct.T.dot(Yt)).to_polar().to(self.device)
        self.Yt = ComplexRect.from_numpy(Yt).to_polar().to(self.device)
        self.Yf = ComplexRect.from_numpy(Yf).to_polar().to(self.device)
        self.Ct = ComplexRect.from_numpy(Ct).to_polar().to(self.device)
        self.Cf = ComplexRect.from_numpy(Cf).to_polar().to(self.device)
        self.vad_mask = self.equality_threshold > (self.vad_max - self.vad_min).abs()

    def init_bus(self):
        self.vm_min = torch.zeros(self.n_bus, device=self.device, dtype=torch.float32)
        self.vm_max = torch.zeros(self.n_bus, device=self.device, dtype=torch.float32)
        self.base_kv = torch.zeros(self.n_bus, device=self.device, dtype=torch.float32)
        self.base_mva = self.pm["baseMVA"]
        self.reference_buses = []
        for bus in self.pm["bus"].values():
            i = bus["bus_i"] - 1
            self.vm_min[i] = bus["vmin"]
            self.vm_max[i] = bus["vmax"]
            self.base_kv[i] = bus["base_kv"]
            if bus["bus_type"] == 3:
                self.reference_buses.append(i)
        self.vm_mask = self.equality_threshold > (self.vm_max - self.vm_min).abs()

    def init_shunt(self):
        Ybus_sh = np.zeros((self.n_bus, self.n_bus), dtype=np.csingle)
        for shunt in self.pm["shunt"].values():
            i = shunt["shunt_bus"] - 1
            Ybus_sh[i, i] += shunt["gs"] + 1j * shunt["bs"]
        self.Ybus_sh = ComplexRect.from_numpy(Ybus_sh).to_polar()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.p_min = self.p_min.to(*args, **kwargs)
        self.p_max = self.p_max.to(*args, **kwargs)
        self.q_min = self.q_min.to(*args, **kwargs)
        self.q_max = self.q_max.to(*args, **kwargs)
        self.rate_a = self.rate_a.to(*args, **kwargs)
        self.vad_max = self.vad_max.to(*args, **kwargs)
        self.vad_min = self.vad_min.to(*args, **kwargs)
        self.Yt = self.Yt.to(*args, **kwargs)
        self.Yf = self.Yf.to(*args, **kwargs)
        self.Ct = self.Ct.to(*args, **kwargs)
        self.Cf = self.Cf.to(*args, **kwargs)
        self.vm_min = self.vm_min.to(*args, **kwargs)
        self.vm_max = self.vm_max.to(*args, **kwargs)
        self.Ybus_sh = self.Ybus_sh.to(*args, **kwargs)
        return self
