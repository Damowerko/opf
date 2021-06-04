from functools import reduce, cached_property
from typing import List, Dict

import alegnn.utils.graphML as gml
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn
from alegnn.modules.architectures import SelectionGNN
import logging

from opf.complex import ComplexRect
from opf.power import NetWrapper


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

    def __init__(
        self,
        # Graph Filtering,
        dimNodeSignals=None,
        nFilterTaps=None,
        bias=None,
        # Nonlinearity,
        nonlinearity=torch.nn.ReLU,
        # Structure
        GSO=None,
        index: List[int] = None,
    ):

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

        super().__init__(
            dimNodeSignals,
            nFilterTaps,
            bias,
            nonlinearity,
            nSelectedNodes,
            poolingFunction,
            poolingSize,
            [],
            GSO,
        )

    def forward(self, x):
        x = super().forward(x)
        index = torch.tensor(self.index, dtype=torch.int64, device=x.device)
        x = torch.index_select(x, 1, index)
        return x


class GNN(pl.LightningModule):
    def __init__(self, gso, features, taps, mlp):
        super().__init__()
        self.save_hyperparameters(ignore=["gso"])

        n_layers = len(taps)
        self.gnn = SelectionGNN(
            features,
            taps,
            True,
            torch.nn.ReLU,
            [gso.shape[-1]] * n_layers,
            gml.NoPool,
            [1] * n_layers,
            mlp,
            gso,
        )

    def forward(self, x):
        return self.gnn(x)


class OPFLogBarrier(pl.LightningModule):
    def __init__(
        self,
        net_wrapper: NetWrapper,
        model,
        t=10,
        s=1000,
        type="log",
        equality_threshold=0.001,
        cost_weight=1.0,
        lr=1e-4,
        constraint_features=False,
    ):
        super().__init__()
        self.net_wrapper = net_wrapper
        self.pm = self.net_wrapper.to_powermodels()
        self.model = model
        self.t = t
        self.s = s
        self.cost_weight = cost_weight
        self.type = type
        self.equality_threshold = equality_threshold

        self.save_hyperparameters(ignore=["net_wrapper", "model"])

        self.n_bus = len(self.pm["bus"])
        self.n_branch = len(self.pm["branch"])
        self.init_gen()
        self.init_load()
        self.init_cost_coefficients()
        self.init_bus()
        self.init_branch()
        self.init_shunt()

    def init_gen(self):
        p_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        p_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        q_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        q_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        gen_matrix = torch.zeros(
            (self.n_bus, len(self.pm["gen"])), device=self.device, dtype=torch.float32
        )
        for gen in self.pm["gen"].values():
            i = gen["gen_bus"] - 1
            p_min[i] = gen["pmin"]
            q_min[i] = gen["qmin"]
            p_max[i] = gen["pmax"]
            q_max[i] = gen["qmax"]
            gen_matrix[i, gen["index"] - 1] = 1
        self.p_mask = self.equality_threshold > (p_max - p_min).abs()
        self.q_mask = self.equality_threshold > (q_max - q_min).abs()

        self.register_buffer("p_min", p_min, False)
        self.register_buffer("p_max", p_max, False)
        self.register_buffer("q_min", q_min, False)
        self.register_buffer("q_max", q_max, False)
        self.register_buffer("gen_matrix", gen_matrix, False)

    def init_load(self):
        load_matrix = torch.zeros(
            (self.n_bus, len(self.pm["load"])), device=self.device, dtype=torch.float32
        )
        for load in self.pm["load"].values():
            i = load["load_bus"] - 1
            load_matrix[i, load["index"] - 1] = 1
        self.register_buffer("load_matrix", load_matrix, False)

    def init_branch(self):
        rate_a = torch.full(
            (self.n_branch, 1), float("inf"), device=self.device, dtype=torch.float32
        )
        vad_max = torch.full(
            (self.n_bus, self.n_bus),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        vad_min = torch.full(
            (self.n_bus, self.n_bus),
            -float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        Yff = np.zeros((self.n_branch,), dtype=np.csingle)
        Yft = np.zeros((self.n_branch,), dtype=np.csingle)
        Ytf = np.zeros((self.n_branch,), dtype=np.csingle)
        Ytt = np.zeros((self.n_branch,), dtype=np.csingle)
        Cf = np.zeros((self.n_branch, self.n_bus), dtype=np.csingle)
        Ct = np.zeros((self.n_branch, self.n_bus), dtype=np.csingle)
        for branch in self.pm["branch"].values():
            index = branch["index"] - 1
            fr_bus = branch["f_bus"] - 1
            to_bus = branch["t_bus"] - 1
            y = 1 / (branch["br_r"] + 1j * branch["br_x"])
            yc_fr = branch["g_fr"] + 1j * branch["b_fr"]
            yc_to = branch["g_to"] + 1j * branch["b_to"]
            ratio = branch["tap"] * np.exp(1j * branch["shift"])
            Yff[index] = (y + yc_fr) / np.abs(ratio) ** 2
            Yft[index] = -y / np.conj(ratio)
            Ytt[index] = y + yc_to
            Ytf[index] = -y / ratio
            Cf[index, fr_bus] = 1
            Ct[index, to_bus] = 1
            rate_a[index] = branch["rate_a"]
            vad_min[fr_bus, to_bus] = branch["angmin"]
            vad_max[fr_bus, to_bus] = branch["angmax"]
        Yt = np.diag(Yff).dot(Cf) + np.diag(Yft).dot(Ct)
        Yf = np.diag(Ytf).dot(Cf) + np.diag(Ytt).dot(Ct)
        self.vad_mask = self.equality_threshold > (vad_max - vad_min).abs()

        self.register_buffer(
            "Ybus_branch", torch.from_numpy(Cf.T.dot(Yf) + Ct.T.dot(Yt)), False
        )
        self.register_buffer("Yt", torch.from_numpy(Yt), False)
        self.register_buffer("Yf", torch.from_numpy(Yf), False)
        self.register_buffer("Ct", torch.from_numpy(Ct), False)
        self.register_buffer("Cf", torch.from_numpy(Cf), False)
        self.register_buffer("rate_a", rate_a, False)
        self.register_buffer("vad_max", vad_max, False)
        self.register_buffer("vad_min", vad_min, False)

    def init_bus(self):
        vm_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        vm_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        base_kv = torch.zeros((self.n_bus, 1), device=self.device, dtype=torch.float32)
        self.base_mva = self.pm["baseMVA"]
        self.reference_buses = []
        for bus in self.pm["bus"].values():
            i = bus["bus_i"] - 1
            vm_min[i] = bus["vmin"]
            vm_max[i] = bus["vmax"]
            base_kv[i] = bus["base_kv"]
            if bus["bus_type"] == 3:
                self.reference_buses.append(i)
        self.vm_mask = self.equality_threshold > (vm_max - vm_min).abs()

        self.register_buffer("vm_min", vm_min, False)
        self.register_buffer("vm_max", vm_max, False)
        self.register_buffer("base_kv", base_kv, False)

    def init_shunt(self):
        Ybus_sh = np.zeros((self.n_bus, 1), dtype=np.csingle)
        for shunt in self.pm["shunt"].values():
            i = shunt["shunt_bus"] - 1
            Ybus_sh[i] += shunt["gs"] + 1j * shunt["bs"]
        self.register_buffer("Ybus_sh", torch.from_numpy(Ybus_sh), False)

    def init_cost_coefficients(self):
        """A tuple of two 3xN matrices, representing the polynomial coefficients of active and reactive power cost."""
        element_types = ["gen", "sgen"]
        pcs, qcs = zip(
            *map(lambda et: self.net_wrapper.cost_coefficients(et), element_types)
        )
        p_coeff = reduce(lambda x, y: x + y, pcs)
        q_coeff = reduce(lambda x, y: x + y, qcs)
        p_coeff = torch.from_numpy(p_coeff).to(torch.float32).to(self.device).T
        q_coeff = torch.from_numpy(q_coeff).to(torch.float32).to(self.device).T
        self.register_buffer("p_coeff", p_coeff, False)
        self.register_buffer("q_coeff", q_coeff, False)

    def forward(self, load):
        load = load.transpose(1, 2) @ self.load_matrix.T
        if self.hparams.constraint_features:
            x = torch.cat(
                (
                    load,
                    self.bus_constraints_matrix.T.unsqueeze(0).repeat(
                        load.shape[0], 1, 1
                    ),
                ),
                dim=1,
            )
        else:
            x = load
        bus = self.model(x)
        bus = torch.reshape(bus, (-1, 4, self.n_bus))
        return bus, load

    def optimal_power_flow(self, bus, load):
        V, S, Sg, Sd = self.bus(bus, load)
        If, It, Sf, St, Sbus = self.power_flow(V)
        cost = self.cost(S.real, S.imag)
        constraints = self.constraints(V, S, Sbus, Sg, Sf, St)
        return cost, constraints

    def cost(self, p, q):
        """Compute the cost to produce the active and reactive power."""
        p_coeff, q_coeff = self.p_coeff, self.q_coeff
        return (
            p_coeff[:, 0]
            + p * p_coeff[:, 1]
            + (p ** 2) * p_coeff[:, 2]
            + q_coeff[:, 0]
            + q * q_coeff[:, 1]
            + (q ** 2) * q_coeff[:, 2]
        ).mean()

    def power_flow(self, V):
        """
        Find the branch variables given the bus voltages. The inputs and outputs should both be
        in the per unit system.

        Reference:
            https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
            https://matpower.org/docs/MATPOWER-manual.pdf
        """
        If = self.Yf @ V  # Current from
        It = self.Yt @ V  # Current to
        Sf = (self.Cf @ V) * If.conj()  # [Cf V] If* = Power from branch
        St = (self.Ct @ V) * It.conj()  # [Cf V] It* = Power to branch
        Sbus_sh = (
            V * self.Ybus_sh.unsqueeze(0)
        ) * V.conj()  # [V][Ybus_shunt] V* = shunt bus power
        Sbus = self.Cf.T @ Sf + self.Cf.T @ Sf + Sbus_sh
        return If, It, Sf, St, Sbus

    def bus(self, bus, load):
        """
        Compute the bus variables: voltage, net power, power generated and power demanded. All of them are complex.

        :param bus: [|V| angle(V) Re(S) Im(S)]
        :param load: [Re(load) Im(load)]
        :return: V, S, Sg, Sd.
        """
        assert bus.shape[0] == load.shape[0]
        assert bus.shape[1] == 4
        assert bus.shape[2] == self.n_bus
        assert load.shape[1] == 2
        assert load.shape[2] == self.n_bus

        # Convert voltage and power to per unit
        vr = bus[:, 0, :]
        vi = bus[:, 1, :]
        p = bus[:, 2, :]
        q = bus[:, 3, :]

        V = torch.complex(vr, vi).unsqueeze(-1)
        S = torch.complex(p, q).unsqueeze(-1)
        Sd = torch.complex(load[:, 0, :], load[:, 1, :]).unsqueeze(-1)
        Sg = S + Sd
        return V, S, Sg, Sd

    def constraints(self, V, S, Sbus, Sg, Sf, St) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Map from constraint name => (value name => tensor value)
        """
        # voltage angle difference
        vad = (V @ V.conj().transpose(-1, -2)).angle()

        constraints = {
            "equality_powerflow": self.equality(S - Sbus),
            # "equality_reference": self.equality(V.angle()[self.reference_buses]),
            "equality_real_power": self.equality(
                (self.p_min - Sg.real)[:, self.p_mask]
            ),
            "inequality_real_power_min": self.inequality(
                (self.p_min - Sg.real)[:, ~self.p_mask]
            ),
            "inequality_real_power_max": self.inequality(
                (Sg.real - self.p_max)[:, ~self.p_mask]
            ),
            "equality_reactive_power": self.equality(
                (self.q_min - Sg.imag)[:, self.q_mask]
            ),
            "inequality_reactive_power_min": self.inequality(
                (self.q_min - Sg.imag)[:, ~self.q_mask]
            ),
            "inequality_reactive_power_max": self.inequality(
                (Sg.imag - self.q_max)[:, ~self.q_mask]
            ),
            "equality_voltage_magnitude": self.equality(
                (self.vm_min - V.abs())[:, self.vm_mask]
            ),
            "inequality_voltage_magnitude_min": self.inequality(
                (self.vm_min - V.abs())[:, ~self.vm_mask]
            ),
            "inequality_voltage_magnitude_max": self.inequality(
                (V.abs() - self.vm_max)[:, ~self.vm_mask]
            ),
            "inequality_forward_rate_max": self.inequality(Sf.abs() - self.rate_a),
            "inequality_backward_rate_max": self.inequality(St.abs() - self.rate_a),
            "equality_voltage_angle_difference": self.equality(
                (self.vad_min - vad)[:, self.vad_mask], angle=True
            ),
            "inequality_voltage_angle_difference_min": self.inequality(
                (self.vad_min - vad)[:, ~self.vad_mask], angle=True
            ),
            "inequality_voltage_angle_difference_max": self.inequality(
                (vad - self.vad_max)[:, ~self.vad_mask], angle=True
            ),
        }
        return constraints

    @property
    def bus_constraints_matrix(self):
        """Returns a matrix representing the bus constraints as a graph signal."""
        return torch.cat(
            (
                self.p_min,
                self.p_max,
                self.q_min,
                self.q_max,
                self.vm_min,
                self.vm_max,
            ),
            dim=1,
        ).to(self.device)

    @property
    def branch_constraints_matrix(self):
        return torch.stack((self.rate_a, self.vad_min, self.vad_max), dim=0)

    def loss(self, cost, constraints):
        constraint_losses = [
            val["loss"]
            for val in constraints.values()
            if val["loss"] is not None and not torch.isnan(val["loss"])
        ]
        return cost * self.cost_weight + torch.stack(constraint_losses).mean()

    def metrics(self, cost, constraints, prefix):
        metrics = {
            f"{prefix}_cost": cost,
            f"{prefix}_equality_loss": 0,
            f"{prefix}_inequality_loss": 0,
        }
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
        return {
            k: v if torch.is_tensor(v) else torch.as_tensor(v)
            for k, v in metrics.items()
        }

    def training_step(self, x, *args, **kwargs):
        cost, constraints = self.optimal_power_flow(*self(x[0]))
        loss = self.loss(cost, constraints)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(self.metrics(cost, constraints, "train"))
        return loss

    def validation_step(self, x, *args, **kwargs):
        cost, constraints = self.optimal_power_flow(*self(x[0]))
        loss = self.loss(cost, constraints)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(self.metrics(cost, constraints, "val"))
        return cost

    def test_step(self, x, *args, **kwargs):
        cost, constraints = self.optimal_power_flow(*self(x[0]))
        self.log_dict(self.metrics(cost, constraints, "test"))
        return cost

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.hparams.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, []),
            "monitor": "val_loss",
            "name": "scheduler",
        }
        return [optimizer], [lr_scheduler]

    def equality(self, u, angle=False):
        if u.nelement() == 0:
            return {"loss": torch.tensor(0, device=self.device)}
        if angle:
            u = torch.fmod(u, 2 * np.pi)
            u[u > 2 * np.pi] = 2 * np.pi - u[u > 2 * np.pi]
        loss = u.abs().square().mean()
        return {"loss": loss}

    def inequality(self, u, angle=False):
        assert not ((u > 0) * torch.isinf(u)).any()

        if angle:
            u = torch.fmod(u, 2 * np.pi)
            u[u > 2 * np.pi] = 2 * np.pi - u[u > 2 * np.pi]

        unconstrained = (u < 0) * torch.isinf(u)
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
                threshold = -1 / (self.s * self.t)
                below = u <= threshold
                log = (-torch.log(-u[below]) / self.t).mean()
                linear = (
                    (-np.log(-threshold) / self.t) + (u[~below] - threshold) * self.s
                ).mean()
                loss = (log if not torch.isnan(log) else 0) + (
                    linear if not torch.isnan(linear) else 0
                )
        return dict(
            loss=torch.as_tensor(loss, dtype=self.dtype, device=self.device),
            violated_rate=torch.as_tensor(
                violated_rate, dtype=self.dtype, device=self.device
            ),
            violated_rms=torch.as_tensor(
                violated_rms, dtype=self.dtype, device=self.device
            ),
        )
