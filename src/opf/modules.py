from functools import reduce
from typing import List, Dict

import alegnn.utils.graphML as gml
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn
from alegnn.modules.architectures import SelectionGNN

from opf.power import NetWrapper
from opf.constraints import equality, inequality


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
        cost_weight=1.0,
        lr=1e-4,
        constraint_features=False,
        eps=1e-3,
    ):
        super().__init__()
        self.net_wrapper = net_wrapper
        self.pm = self.net_wrapper.to_powermodels()
        self.model = model
        self.t = t
        self.s = s
        self.cost_weight = cost_weight
        self.eps = eps
        self.detailed_metrics = False

        self.save_hyperparameters(ignore=["net_wrapper", "model"])

        self.n_bus = len(self.pm["bus"])
        self.n_branch = len(self.pm["branch"])
        self.init_gen()
        self.init_load()
        self.init_bus()
        self.init_branch()
        self.init_shunt()

    def init_gen(self):
        p_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        p_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        q_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        q_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        p_coeff = torch.zeros((self.n_bus, 3), device=self.device, dtype=self.dtype)
        q_coeff = torch.zeros((self.n_bus, 3), device=self.device, dtype=self.dtype)
        gen_matrix = torch.zeros(
            (self.n_bus, len(self.pm["gen"])), device=self.device, dtype=self.dtype
        )
        for gen in self.pm["gen"].values():
            i = gen["gen_bus"] - 1
            p_min[i] = gen["pmin"]
            q_min[i] = gen["qmin"]
            p_max[i] = gen["pmax"]
            q_max[i] = gen["qmax"]
            assert gen["model"] == 2  # cost is polynomial
            assert len(gen["cost"]) == 3  # only real cost
            p_coeff[i, :] = torch.as_tensor(
                gen["cost"][::-1]
            )  # Cost is polynomial c0 x^2 + c1x
            gen_matrix[i, gen["index"] - 1] = 1
        self.register_buffer("p_mask", self.eps > (p_max - p_min).abs())
        self.register_buffer("q_mask", self.eps > (q_max - q_min).abs())
        self.register_buffer("p_min", p_min)
        self.register_buffer("p_max", p_max)
        self.register_buffer("q_min", q_min)
        self.register_buffer("q_max", q_max)
        self.register_buffer("gen_matrix", gen_matrix)
        self.register_buffer("p_coeff", p_coeff)
        self.register_buffer("q_coeff", q_coeff)

    def init_load(self):
        load_matrix = torch.zeros(
            (self.n_bus, len(self.pm["load"])), device=self.device, dtype=self.dtype
        )
        for load in self.pm["load"].values():
            i = load["load_bus"] - 1
            load_matrix[i, load["index"] - 1] = 1
        self.register_buffer("load_matrix", load_matrix)

    def init_branch(self):
        rate_a = torch.full(
            (self.n_branch, 1), float("inf"), device=self.device, dtype=self.dtype
        )
        vad_max = torch.full(
            (self.n_branch, 1),
            float("inf"),
            device=self.device,
            dtype=self.dtype,
        )
        vad_min = torch.full(
            (self.n_branch, 1),
            -float("inf"),
            device=self.device,
            dtype=self.dtype,
        )

        Yff = np.zeros((self.n_branch,), dtype=np.cdouble)
        Yft = np.zeros((self.n_branch,), dtype=np.cdouble)
        Ytf = np.zeros((self.n_branch,), dtype=np.cdouble)
        Ytt = np.zeros((self.n_branch,), dtype=np.cdouble)
        Cf = np.zeros((self.n_branch, self.n_bus), dtype=np.cdouble)
        Ct = np.zeros((self.n_branch, self.n_bus), dtype=np.cdouble)
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
            vad_min[index] = branch["angmin"]
            vad_max[index] = branch["angmax"]
        Yf = np.diag(Yff).dot(Cf) + np.diag(Yft).dot(Ct)
        Yt = np.diag(Ytf).dot(Cf) + np.diag(Ytt).dot(Ct)

        self.register_buffer("vad_mask", self.eps > (vad_max - vad_min).abs())
        self.Yf = torch.from_numpy(Yf)
        self.Yt = torch.from_numpy(Yt)
        self.Cf = torch.from_numpy(Cf)
        self.Ct = torch.from_numpy(Ct)
        # self.register_buffer("Yf", torch.from_numpy(Yf))
        # self.register_buffer("Yt", torch.from_numpy(Yt))
        # self.register_buffer("Cf", torch.from_numpy(Cf))
        # self.register_buffer("Ct", torch.from_numpy(Ct))
        self.register_buffer("rate_a", rate_a)
        self.register_buffer("vad_max", vad_max)
        self.register_buffer("vad_min", vad_min)

    def init_bus(self):
        vm_min = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        vm_max = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        base_kv = torch.zeros((self.n_bus, 1), device=self.device, dtype=self.dtype)
        self.base_mva = self.pm["baseMVA"]
        self.reference_buses = []
        for bus in self.pm["bus"].values():
            i = bus["bus_i"] - 1
            vm_min[i] = bus["vmin"]
            vm_max[i] = bus["vmax"]
            base_kv[i] = bus["base_kv"]
            if bus["bus_type"] == 3:
                self.reference_buses.append(i)

        self.register_buffer("vm_mask", self.eps > (vm_max - vm_min).abs())
        self.register_buffer("vm_min", vm_min)
        self.register_buffer("vm_max", vm_max)
        self.register_buffer("base_kv", base_kv)

    def init_shunt(self):
        Ybus_sh = np.zeros((self.n_bus, 1), dtype=np.cdouble)
        for shunt in self.pm["shunt"].values():
            i = shunt["shunt_bus"] - 1
            Ybus_sh[i] += shunt["gs"] + 1j * shunt["bs"]
        self.Ybus_sh = torch.from_numpy(Ybus_sh)
        # self.register_buffer("Ybus_sh", torch.from_numpy(Ybus_sh))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        if len(args) == 0:
            device = kwargs["device"] if "device" in kwargs else self.device
            dtype = kwargs["dtype"] if "dtype" in kwargs else self.dtype
        elif isinstance(args[0], torch.dtype):
            device = self.device
            dtype = args[0]
        elif isinstance(args[0], torch.device):
            device = args[0]
            dtype = self.dtype
        elif isinstance(args[0], torch.Tensor):
            device = args[0].device
            dtype = args[0].dtype
        else:
            raise ValueError(f"Unexpected arguments")

        if dtype == torch.float32:
            dtype = torch.complex64
        elif dtype == torch.float64:
            dtype = torch.complex128
        elif dtype is not None:
            raise ValueError(f"Unexpected dtype: {dtype}.")

        self.Yf = self.Yf.to(device=device, dtype=dtype)
        self.Yt = self.Yt.to(device=device, dtype=dtype)
        self.Cf = self.Cf.to(device=device, dtype=dtype)
        self.Ct = self.Ct.to(device=device, dtype=dtype)
        self.Ybus_sh = self.Ybus_sh.to(device=device, dtype=dtype)
        return self

    def forward(self, load):
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
        return bus

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
            V * self.Ybus_sh.unsqueeze(0).conj() * V.conj()
        )  # [V][Ybus_shunt*] V* = shunt bus power
        Sbus_branch = V * (self.Cf.T @ If.conj() + self.Ct.T @ It.conj())
        Sbus = Sbus_branch + Sbus_sh
        return If, It, Sf, St, Sbus

    def bus(self, bus, load):
        """
        Compute the bus variables: voltage, net power, power generated and power demanded. All of them are complex.

        :param bus: [|V| angle(V) Re(S) Im(S)]. Respectively: bus voltage, bus voltage angle, and power injected at
        the bus. Note that PandaPower uses the power demanded convention.
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
        Vf = self.Cf @ V
        Vt = self.Ct @ V
        vad = (Vf * Vt.conj()).angle()

        constraints = {
            "equality/bus_power": equality(S, Sbus, eps=self.eps),
            # "equality/reference": equality(V.angle()[self.reference_buses]),
            "inequality/active_power": inequality(
                Sg.real,
                self.p_min,
                self.p_max,
                self.s,
                self.t,
                eps=self.eps,
            ),
            "inequality/reactive_power": inequality(
                Sg.imag,
                self.q_min,
                self.q_max,
                self.s,
                self.t,
                eps=self.eps,
            ),
            "inequality/voltage_magnitude": inequality(
                V.abs(),
                self.vm_min,
                self.vm_max,
                self.s,
                self.t,
                eps=self.eps,
            ),
            "inequality/forward_rate": inequality(
                Sf.abs(),
                torch.zeros_like(self.rate_a),
                self.rate_a,
                self.s,
                self.t,
                eps=self.eps,
            ),
            "inequality/backward_rate": inequality(
                St.abs(),
                torch.zeros_like(self.rate_a),
                self.rate_a,
                self.s,
                self.t,
                eps=self.eps,
            ),
            "inequality/voltage_angle_difference": inequality(
                vad,
                self.vad_min,
                self.vad_max,
                self.s,
                self.t,
                angle=True,
                eps=self.eps,
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
        return cost * self.cost_weight + torch.stack(constraint_losses).sum()

    def metrics(self, cost, constraints, prefix, detailed=False):
        aggregate_metrics = {
            f"{prefix}/cost": [cost],
            f"{prefix}/equality/loss": [],
            f"{prefix}/equality/rate": [],
            f"{prefix}/equality/error_mean": [],
            f"{prefix}/equality/error_max": [],
            f"{prefix}/inequality/loss": [],
            f"{prefix}/inequality/rate": [],
            f"{prefix}/inequality/error_mean": [],
            f"{prefix}/inequality/error_max": [],
        }
        detailed_metrics = {}
        reduce_fn = {
            "default": torch.sum,
            "error_mean": torch.mean,
            "error_max": torch.max,
            "rate": torch.mean,
        }

        for constraint_name, constraint_values in constraints.items():
            constraint_type = constraint_name.split("/")[0]
            for value_name, value in constraint_values.items():
                if detailed:
                    detailed_metrics[f"{prefix}/{constraint_name}/{value_name}"] = value
                aggregate_name = f"{prefix}/{constraint_type}/{value_name}"
                aggregate_metrics[aggregate_name].append(value.reshape(1))
        for aggregate_name in aggregate_metrics:
            value_name = aggregate_name.rsplit("/", 1)[1]
            fn = (
                reduce_fn[value_name]
                if value_name in reduce_fn
                else reduce_fn["default"]
            )
            aggregate_metrics[aggregate_name] = fn(
                torch.stack(aggregate_metrics[aggregate_name])
            )
        return {**aggregate_metrics, **detailed_metrics}

    def training_step(self, batch, *args):
        load = batch[0] @ self.load_matrix.T
        bus = self(load)
        cost, constraints = self.optimal_power_flow(bus, load)
        loss = self.loss(cost, constraints)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, *args):
        load = batch[0] @ self.load_matrix.T
        bus = self(load)
        cost, constraints = self.optimal_power_flow(bus, load)
        loss = self.loss(cost, constraints)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(
            self.metrics(cost, constraints, "val", self.detailed_metrics),
            sync_dist=True,
        )

    def test_step(self, batch, *args):
        with torch.no_grad():
            load, acopf_bus = batch
            load = load @ self.load_matrix.T
            bus = self(load)
            # Run actual powerflow calculations
            bus = self.project_pandapower(bus, load)
            cost, constraints = self.optimal_power_flow(bus, load)
            test_metrics = self.metrics(cost, constraints, "test", self.detailed_metrics)
            self.log_dict(test_metrics)

            # Test the ACOPF solution for reference.
            cost, constraints = self.optimal_power_flow(self.bus_from_polar(acopf_bus), load)
            acopf_metrics = self.metrics(cost, constraints, "acopf", self.detailed_metrics)
            self.log_dict(acopf_metrics)

            return dict(**test_metrics, **acopf_metrics)

    def project_pandapower(self, bus, load):
        with torch.no_grad():
            _, _, Sg, Sd = self.bus(bus, load)
            self.net_wrapper.set_gen_sparse(Sg.real.squeeze().cpu().numpy(), Sg.imag.squeeze().cpu().numpy())
            self.net_wrapper.set_load_sparse(
                Sd.real.squeeze().cpu().numpy(), Sd.imag.squeeze().cpu().numpy()
            )
            res_powerflow = self.net_wrapper.powerflow()
            if res_powerflow is not None:
                bus, _, _ = res_powerflow
                bus = torch.as_tensor(bus, device=self.device, dtype=self.dtype)
                bus = self.bus_from_polar(bus.unsqueeze(0))
        return bus

    def bus_from_polar(self, bus):
        """
        Convert bus voltage from polar to rectangular.
        """
        bus = bus.clone()
        V = torch.polar(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.real
        bus[:, 1, :] = V.imag
        return bus

    def bus_to_polar(self, bus):
        """
        Convert bus voltage from rectangular to polar.
        """
        bus = bus.clone()
        V = torch.complex(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.abs()
        bus[:, 1, :] = V.angle()
        return bus

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.hparams.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, []),
            "monitor": "val/loss",
            "name": "scheduler",
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def relaxed_log(u, s, t):
        threshold = -1 / (s * t)
        return torch.where(
            u <= threshold,
            -torch.log(-u) / t,
            (-np.log(-threshold) / t) + (u - threshold) * s,
        )
