import argparse
import subprocess
import typing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, HeteroData

import opf.powerflow as pf
from opf.constraints import equality, inequality
from opf.dataset import PowerflowBatch, PowerflowData

class OPFDual(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lamb: torch.Tensor,
        mu: torch.Tensor,
        t=500.0,
        t_step=0.0,
        s=100.0,
        s_step=0.0,
        equality_weight=1.0,
        equality_step=1.0,
        lr=1e-4,
        weight_decay=0.0,
        eps=1e-3,
        enforce_constraints=False,
        detailed_metrics=False,
        **kwargs,            
    ):
        super().__init__()
        self.model = model
        self.automatic_optimization=False
        
        self.lamb = lamb
        self.mu = mu
        # alternatively:
        # self.equality_multiplier = torch.zeros([shape])
        # self.inequality_multiplier = torch.zeros([shape])

        # variables controlling the evolution of the log barrier
        self.t_start = t
        self.t_step = t_step
        self.s_start = s
        self.s_step = s_step
        self.equality_start = equality_weight
        self.equality_step = equality_step
        # other parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self._enforce_constraints = enforce_constraints
        self.detailed_metrics = detailed_metrics
        self.save_hyperparameters(ignore=["model", "kwargs"])

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("OPFLogBarrier")
        group.add_argument("--s", type=float, default=100)
        group.add_argument("--s_step", type=float, default=0.0)
        group.add_argument("--t", type=float, default=500)
        group.add_argument("--t_step", type=float, default=0.0)
        group.add_argument("--equality_weight", type=float, default=1.0)
        group.add_argument("--equality_step", type=float, default=1.0)
        group.add_argument("--lr", type=float, default=3e-4)
        group.add_argument("--weight_decay", type=float, default=0.0)
        group.add_argument("--eps", type=float, default=1e-3)
        group.add_argument("--enforce_constraints", action="store_true", default=False)
        group.add_argument("--detailed_metrics", action="store_true", default=False)

    @property
    def s(self):
        return self.s_start + self.s_step * self.current_epoch

    @property
    def t(self):
        return self.t_start + self.t_step * self.current_epoch

    @property
    def equality_weight(self):
        return self.equality_start + self.equality_step * self.current_epoch

    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
    ):
        
        """
        TODO:
        Make another forwad function to update multipliers
        """
        data, powerflow_parameters = input
        if isinstance(data, HeteroData):
            n_batch = data["bus"].x.shape[0] // powerflow_parameters.n_bus
            output = self.model(data.x_dict, data.adj_t_dict)
            bus = output["bus"][:, :2]
            gen = output["gen"][:, :2]

            # make an assumption here that there is at most one generator per bus
            # assert gen == bus[powerflow_parameters.gen_bus_ids,2;]

            load = data["bus"].x[:, :2]
        elif isinstance(data, Data):
            n_batch = data.x.shape[0] // powerflow_parameters.n_bus
            bus = self.model(data.x, data.edge_index, data.edge_attr)
            load = data.x[:, :2]
        else:
            raise ValueError(
                f"Unsupported data type {type(data)}, expected Data or HeteroData."
            )

        # Reshape the output to (batch_size, n_features, n_bus or n_gen)
        bus = bus.view(n_batch, powerflow_parameters.n_bus, 2).mT
        gen = gen.view(n_batch, powerflow_parameters.n_gen, 2).mT
        load = load.view(n_batch, powerflow_parameters.n_bus, 2).mT
        V = self.parse_bus(bus)
        Sg = self.parse_gen(gen)
        Sd = self.parse_load(load)
        if self._enforce_constraints:
            V, Sg = self.enforce_constraints(V, Sg, powerflow_parameters)
        return V, Sg, Sd

    def sigmoid_bound(self, x, lb, ub):
        scale = ub - lb
        return scale * torch.sigmoid(x) + lb

    def enforce_constraints(self, V, Sg, params: pf.PowerflowParameters):
        vm = self.sigmoid_bound(V.abs(), params.vm_min, params.vm_max)
        V = torch.polar(vm, V.angle())  # V * vm / V.abs()
        Sg.real = self.sigmoid_bound(Sg.real, params.Sg_min.real, params.Sg_max.real)
        Sg.imag = self.sigmoid_bound(Sg.imag, params.Sg_min.imag, params.Sg_max.imag)
        return V, Sg

    def _step_helper(
        self,
        V: torch.Tensor,
        Sg: torch.Tensor,
        Sd: torch.Tensor,
        parameters: pf.PowerflowParameters,
        substitute_equality=False,
        project_powermodels=False,
    ):
        if substitute_equality and project_powermodels:
            raise ValueError(
                "substitute_equality and project_powermodels are mutually exclusive"
            )
        if project_powermodels:
            # Project the solution to the powermodels solution
            V, Sg, Sd = self.project_powermodels(V, Sg, Sd, parameters)
        variables = pf.powerflow(V, Sg, Sd, parameters)

        # print(f'Sg shape: {Sg.shape}')                   # Sg shape: torch.Size([256, 6])
        # print(f'Sbus shape: {variables.Sbus.shape}')     # Sbus shape: torch.Size([256, 30])
        # print(f'Sd shape: {Sd.shape}')                   # Sd shape: torch.Size([256, 30])

        if substitute_equality:
            # Make a substitution to enforce equality constraints
            Sg = variables.Sbus + Sd
            variables = pf.powerflow(V, Sg, Sd, parameters)
        constraints = self.constraints(variables, parameters)
        cost = self.cost(variables, parameters)
        loss = self.loss(cost, constraints)
        return variables, constraints, cost, loss

    def training_step(self, batch: PowerflowBatch):

        # Not exactly sure what to do here... think about it...
        # Change forward
        _, constraints, cost, loss_uno = self._step_helper(
            *self.forward(batch), batch.powerflow_parameters
        )

        loss_dos = self.loss(cost, constraints)
        self.log(
            "train/loss",
            loss_dos,
            prog_bar=True,
            batch_size=batch.data.num_graphs,
        )
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics),
            batch_size=batch.data.num_graphs,
        )
        return

    def validation_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            batch_size = batch.data.num_graphs
            _, constraints, cost, loss = self._step_helper(
                *self.forward(batch), batch.powerflow_parameters
            )
            self.log(
                "val/loss",
                loss,
                batch_size=batch_size,
            )
            metrics = self.metrics(cost, constraints, "val", self.detailed_metrics)
            self.log_dict(metrics, batch_size=batch_size)

            # Metric that does not depend on the loss function shape
            self.log(
                "val/invariant",
                cost
                + metrics["val/equality/error_mean"]
                + metrics["val/inequality/error_mean"],
                batch_size=batch_size,
                prog_bar=True,
            )

    def test_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            _, constraints, cost, _ = self._step_helper(
                *self.forward(batch),
                batch.powerflow_parameters,
                project_powermodels=True,
            )
            test_metrics = self.metrics(
                cost, constraints, "test", self.detailed_metrics
            )
            self.log_dict(
                test_metrics,
                batch_size=batch.data.num_graphs,
            )
            # TODO: rethink how to do comparison against ACOPF
            # Test the ACOPF solution for reference.
            # acopf_bus = self.bus_from_polar(acopf_bus)
            # _, constraints, cost, _ = self._step_helper(
            #     *self.parse_bus(acopf_bus),
            #     self.parse_load(load),
            #     project_pandapower=False,
            # )
            # acopf_metrics = self.metrics(
            #     cost, constraints, "acopf", self.detailed_metrics
            # )
            # self.log_dict(acopf_metrics)
            # return dict(**test_metrics, **acopf_metrics)
            return test_metrics

    def project_powermodels(
        self,
        V: torch.Tensor,
        Sg: torch.Tensor,
        Sd: torch.Tensor,
        parameters: pf.PowerflowParameters,
    ):
        bus_shape = V.shape
        gen_shape = Sg.shape
        dtype = V.dtype
        device = V.device
        V = torch.view_as_real(V.cpu()).view(-1, parameters.n_bus, 2).numpy()
        Sg = torch.view_as_real(Sg.cpu()).view(-1, parameters.n_gen, 2).numpy()
        Sd = torch.view_as_real(Sd.cpu()).view(-1, parameters.n_bus, 2).numpy()

        # TODO: make this more robust, maybe use PyJulia
        # currently what we do is save the data to a temporary directory
        # then run the julia script and load the data back
        with TemporaryDirectory() as tempdir:
            script_path = Path(__file__).parent / "project.jl"
            busfile = Path(tempdir) / "busfile.npz"
            np.savez(busfile, V=V, Sg=Sg, Sd=Sd)
            subprocess.run(
                [
                    "julia",
                    "--project=@.",
                    script_path.as_posix(),
                    "--casefile",
                    parameters.casefile,
                    "--busfile",
                    busfile.as_posix(),
                ]
            )
            bus = np.load(busfile)
        V, Sg, Sd = bus["V"], bus["Sg"], bus["Sd"]
        # convert back to torch tensors with the original device, dtype, and shape
        V = torch.from_numpy(V)
        Sg = torch.from_numpy(Sg)
        Sd = torch.from_numpy(Sd)
        V = torch.complex(V[..., 0], V[..., 1]).to(device, dtype).view(bus_shape)
        Sg = torch.complex(Sg[..., 0], Sg[..., 1]).to(device, dtype).view(gen_shape)
        Sd = torch.complex(Sd[..., 0], Sd[..., 1]).to(device, dtype).view(bus_shape)
        return V, Sg, Sd

    def parse_bus(self, bus: torch.Tensor):
        assert bus.shape[1] == 2

        # Convert voltage and power to per unit
        vr = bus[:, 0, :]
        vi = bus[:, 1, :]
        # IF HOMO
        # pg = bus[:, 2, :]
        # qg = bus[:, 3, :]

        V = torch.complex(vr, vi)
        # Sg = torch.complex(pg, qg)
        return V

    def parse_load(self, load: torch.Tensor):
        """
        Converts the load data to the format required by the powerflow module (complex tensor).

        Args:
            load: A tensor of shape (batch_size, n_features, n_bus). The first two features should contain the active and reactive load.

        """
        Sd = torch.complex(load[:, 0, :], load[:, 1, :])
        return Sd

    def loss(self, cost, constraints):
        """
        TODO:
        Change this to equation in overleaf
        """
        constraint_losses = [
            val["loss"]
            for val in constraints.values()
            if val["loss"] is not None and not torch.isnan(val["loss"])
        ]
        if len(constraint_losses) == 0:
            constraint_losses = [torch.zeros(1, device=self.device, dtype=self.dtype)]  # type: ignore
        return cost + torch.stack(constraint_losses).sum()

    def parse_gen(self, gen: torch.Tensor):
        assert gen.shape[1] == 2
        Sg = torch.complex(gen[:, 0, :], gen[:, 1, :])
        return Sg

    def cost(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
    ) -> torch.Tensor:
        """Compute the cost to produce the active and reactive power."""
        p = variables.Sg.real
        p_coeff = powerflow_parameters.cost_coeff
        cost = torch.zeros_like(p)
        for i in range(p_coeff.shape[1]):
            cost += p_coeff[:, i] * p.squeeze() ** i
        # cost cannot be negative
        cost = torch.clamp(cost, min=0)
        # # normalize the cost by the number of generators
        return cost.mean(0).sum() / powerflow_parameters.reference_cost
        # return cost.mean()

    def constraints(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Nested map from constraint name => (value name => tensor value)
        """

        """
        TODO:
        Update this for the new loss
        """
        constraints = pf.build_constraints(variables, powerflow_parameters)
        values = {}
        # l_i = 0
        # m_i = 0
        for name, constraint in constraints.items():
            if isinstance(constraint, pf.EqualityConstraint):
                values[name] = equality(
                    constraint.value,
                    constraint.target,
                    constraint.mask,
                    self.eps,
                    constraint.isAngle,
                )

                # apply weight
                values[name]["loss"] *= self.equality_weight
                # values[name]["loss"] *= self.lamb[l_i]
                # l_i+=1                
            elif isinstance(constraint, pf.InequalityConstraint):
                values[name] = inequality(
                    constraint.variable,
                    constraint.min,
                    constraint.max,
                    self.s,
                    self.t,
                    self.eps,
                    constraint.isAngle,
                )
                # values[name]["loss"] *= self.mu[m_i]
                # m_i+=1
        return values

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

    def configure_optimizers(self):
        primal_opt = torch.optim.AdamW(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )
        dual_opt = torch.optim.AdamW(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )
        return primal_opt, dual_opt

    @staticmethod
    def bus_from_polar(bus):
        """
        Convert bus voltage from polar to rectangular.
        """
        bus = bus.clone()
        V = torch.polar(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.real
        bus[:, 1, :] = V.imag
        return bus

    @staticmethod
    def bus_to_polar(bus):
        """
        Convert bus voltage from rectangular to polar.
        """
        bus = bus.clone()
        V = torch.complex(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.abs()
        bus[:, 1, :] = V.angle()
        return bus


class OPFLogBarrier(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        t=500.0,
        t_step=0.0,
        s=100.0,
        s_step=0.0,
        equality_weight=1.0,
        equality_step=1.0,
        lr=1e-4,
        weight_decay=0.0,
        eps=1e-3,
        enforce_constraints=False,
        detailed_metrics=False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        # variables controlling the evolution of the log barrier
        self.t_start = t
        self.t_step = t_step
        self.s_start = s
        self.s_step = s_step
        self.equality_start = equality_weight
        self.equality_step = equality_step
        # other parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self._enforce_constraints = enforce_constraints
        self.detailed_metrics = detailed_metrics
        self.save_hyperparameters(ignore=["model", "kwargs"])

        # zeros of full? what shape also?
        # should this be passed as an argument?
        # self.equality_multiplier = torch.zeros()
        # self.inequality_multiplier = torch.zeros()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("OPFLogBarrier")
        group.add_argument("--s", type=float, default=100)
        group.add_argument("--s_step", type=float, default=0.0)
        group.add_argument("--t", type=float, default=500)
        group.add_argument("--t_step", type=float, default=0.0)
        group.add_argument("--equality_weight", type=float, default=1.0)
        group.add_argument("--equality_step", type=float, default=1.0)
        group.add_argument("--lr", type=float, default=3e-4)
        group.add_argument("--weight_decay", type=float, default=0.0)
        group.add_argument("--eps", type=float, default=1e-3)
        group.add_argument("--enforce_constraints", action="store_true", default=False)
        group.add_argument("--detailed_metrics", action="store_true", default=False)

    @property
    def s(self):
        return self.s_start + self.s_step * self.current_epoch

    @property
    def t(self):
        return self.t_start + self.t_step * self.current_epoch

    @property
    def equality_weight(self):
        return self.equality_start + self.equality_step * self.current_epoch

    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
    ):
        data, powerflow_parameters = input
        if isinstance(data, HeteroData):
            n_batch = data["bus"].x.shape[0] // powerflow_parameters.n_bus
            output = self.model(data.x_dict, data.adj_t_dict)
            bus = output["bus"][:, :2]
            gen = output["gen"][:, :2]

            # make an assumption here that there is at most one generator per bus
            # assert gen == bus[powerflow_parameters.gen_bus_ids,2;]

            load = data["bus"].x[:, :2]
        elif isinstance(data, Data):
            n_batch = data.x.shape[0] // powerflow_parameters.n_bus
            bus = self.model(data.x, data.edge_index, data.edge_attr)
            load = data.x[:, :2]
        else:
            raise ValueError(
                f"Unsupported data type {type(data)}, expected Data or HeteroData."
            )

        # Reshape the output to (batch_size, n_features, n_bus or n_gen)
        bus = bus.view(n_batch, powerflow_parameters.n_bus, 2).mT
        gen = gen.view(n_batch, powerflow_parameters.n_gen, 2).mT
        load = load.view(n_batch, powerflow_parameters.n_bus, 2).mT
        V = self.parse_bus(bus)
        Sg = self.parse_gen(gen)
        Sd = self.parse_load(load)
        if self._enforce_constraints:
            V, Sg = self.enforce_constraints(V, Sg, powerflow_parameters)
        return V, Sg, Sd

    def sigmoid_bound(self, x, lb, ub):
        scale = ub - lb
        return scale * torch.sigmoid(x) + lb

    def enforce_constraints(self, V, Sg, params: pf.PowerflowParameters):
        vm = self.sigmoid_bound(V.abs(), params.vm_min, params.vm_max)
        V = torch.polar(vm, V.angle())  # V * vm / V.abs()
        Sg.real = self.sigmoid_bound(Sg.real, params.Sg_min.real, params.Sg_max.real)
        Sg.imag = self.sigmoid_bound(Sg.imag, params.Sg_min.imag, params.Sg_max.imag)
        return V, Sg

    def _step_helper(
        self,
        V: torch.Tensor,
        Sg: torch.Tensor,
        Sd: torch.Tensor,
        parameters: pf.PowerflowParameters,
        substitute_equality=False,
        project_powermodels=False,
    ):
        if substitute_equality and project_powermodels:
            raise ValueError(
                "substitute_equality and project_powermodels are mutually exclusive"
            )
        if project_powermodels:
            # Project the solution to the powermodels solution
            V, Sg, Sd = self.project_powermodels(V, Sg, Sd, parameters)
        variables = pf.powerflow(V, Sg, Sd, parameters)

        # print(f'Sg shape: {Sg.shape}')                   # Sg shape: torch.Size([256, 6])
        # print(f'Sbus shape: {variables.Sbus.shape}')     # Sbus shape: torch.Size([256, 30])
        # print(f'Sd shape: {Sd.shape}')                   # Sd shape: torch.Size([256, 30])

        if substitute_equality:
            # Make a substitution to enforce equality constraints
            Sg = variables.Sbus + Sd
            variables = pf.powerflow(V, Sg, Sd, parameters)
        constraints = self.constraints(variables, parameters)
        cost = self.cost(variables, parameters)
        loss = self.loss(cost, constraints)
        return variables, constraints, cost, loss

    def training_step(self, batch: PowerflowBatch):
        _, constraints, cost, loss = self._step_helper(
            *self.forward(batch), batch.powerflow_parameters
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=batch.data.num_graphs,
        )
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics),
            batch_size=batch.data.num_graphs,
        )
        return loss

    def validation_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            batch_size = batch.data.num_graphs
            _, constraints, cost, loss = self._step_helper(
                *self.forward(batch), batch.powerflow_parameters
            )
            self.log(
                "val/loss",
                loss,
                batch_size=batch_size,
            )
            metrics = self.metrics(cost, constraints, "val", self.detailed_metrics)
            self.log_dict(metrics, batch_size=batch_size)

            # Metric that does not depend on the loss function shape
            self.log(
                "val/invariant",
                cost
                + metrics["val/equality/error_mean"]
                + metrics["val/inequality/error_mean"],
                batch_size=batch_size,
                prog_bar=True,
            )

    def test_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            _, constraints, cost, _ = self._step_helper(
                *self.forward(batch),
                batch.powerflow_parameters,
                project_powermodels=True,
            )
            test_metrics = self.metrics(
                cost, constraints, "test", self.detailed_metrics
            )
            self.log_dict(
                test_metrics,
                batch_size=batch.data.num_graphs,
            )
            # TODO: rethink how to do comparison against ACOPF
            # Test the ACOPF solution for reference.
            # acopf_bus = self.bus_from_polar(acopf_bus)
            # _, constraints, cost, _ = self._step_helper(
            #     *self.parse_bus(acopf_bus),
            #     self.parse_load(load),
            #     project_pandapower=False,
            # )
            # acopf_metrics = self.metrics(
            #     cost, constraints, "acopf", self.detailed_metrics
            # )
            # self.log_dict(acopf_metrics)
            # return dict(**test_metrics, **acopf_metrics)
            return test_metrics

    def project_powermodels(
        self,
        V: torch.Tensor,
        Sg: torch.Tensor,
        Sd: torch.Tensor,
        parameters: pf.PowerflowParameters,
    ):
        bus_shape = V.shape
        gen_shape = Sg.shape
        dtype = V.dtype
        device = V.device
        V = torch.view_as_real(V.cpu()).view(-1, parameters.n_bus, 2).numpy()
        Sg = torch.view_as_real(Sg.cpu()).view(-1, parameters.n_gen, 2).numpy()
        Sd = torch.view_as_real(Sd.cpu()).view(-1, parameters.n_bus, 2).numpy()

        # TODO: make this more robust, maybe use PyJulia
        # currently what we do is save the data to a temporary directory
        # then run the julia script and load the data back
        with TemporaryDirectory() as tempdir:
            script_path = Path(__file__).parent / "project.jl"
            busfile = Path(tempdir) / "busfile.npz"
            np.savez(busfile, V=V, Sg=Sg, Sd=Sd)
            subprocess.run(
                [
                    "julia",
                    "--project=@.",
                    script_path.as_posix(),
                    "--casefile",
                    parameters.casefile,
                    "--busfile",
                    busfile.as_posix(),
                ]
            )
            bus = np.load(busfile)
        V, Sg, Sd = bus["V"], bus["Sg"], bus["Sd"]
        # convert back to torch tensors with the original device, dtype, and shape
        V = torch.from_numpy(V)
        Sg = torch.from_numpy(Sg)
        Sd = torch.from_numpy(Sd)
        V = torch.complex(V[..., 0], V[..., 1]).to(device, dtype).view(bus_shape)
        Sg = torch.complex(Sg[..., 0], Sg[..., 1]).to(device, dtype).view(gen_shape)
        Sd = torch.complex(Sd[..., 0], Sd[..., 1]).to(device, dtype).view(bus_shape)
        return V, Sg, Sd

    def parse_bus(self, bus: torch.Tensor):
        assert bus.shape[1] == 2

        # Convert voltage and power to per unit
        vr = bus[:, 0, :]
        vi = bus[:, 1, :]
        # IF HOMO
        # pg = bus[:, 2, :]
        # qg = bus[:, 3, :]

        V = torch.complex(vr, vi)
        # Sg = torch.complex(pg, qg)
        return V

    def parse_load(self, load: torch.Tensor):
        """
        Converts the load data to the format required by the powerflow module (complex tensor).

        Args:
            load: A tensor of shape (batch_size, n_features, n_bus). The first two features should contain the active and reactive load.

        """
        Sd = torch.complex(load[:, 0, :], load[:, 1, :])
        return Sd

    def loss(self, cost, constraints):
        constraint_losses = [
            val["loss"]
            for val in constraints.values()
            if val["loss"] is not None and not torch.isnan(val["loss"])
        ]
        if len(constraint_losses) == 0:
            constraint_losses = [torch.zeros(1, device=self.device, dtype=self.dtype)]  # type: ignore
        return cost + torch.stack(constraint_losses).sum()
    
    def dual_loss(self, cost, constraints):
        eq_loss

    def parse_gen(self, gen: torch.Tensor):
        assert gen.shape[1] == 2
        Sg = torch.complex(gen[:, 0, :], gen[:, 1, :])
        return Sg

    def cost(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
    ) -> torch.Tensor:
        """Compute the cost to produce the active and reactive power."""
        p = variables.Sg.real
        p_coeff = powerflow_parameters.cost_coeff
        cost = torch.zeros_like(p)
        for i in range(p_coeff.shape[1]):
            cost += p_coeff[:, i] * p.squeeze() ** i
        # cost cannot be negative
        cost = torch.clamp(cost, min=0)
        # # normalize the cost by the number of generators
        return cost.mean(0).sum() / powerflow_parameters.reference_cost
        # return cost.mean()

    def constraints(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Nested map from constraint name => (value name => tensor value)
        """
        constraints = pf.build_constraints(variables, powerflow_parameters)
        values = {}
        for name, constraint in constraints.items():
            if isinstance(constraint, pf.EqualityConstraint):
                values[name] = equality(
                    constraint.value,
                    constraint.target,
                    constraint.mask,
                    self.eps,
                    constraint.isAngle,
                )

                # apply weight
                values[name]["loss"] *= self.equality_weight
            elif isinstance(constraint, pf.InequalityConstraint):
                values[name] = inequality(
                    constraint.variable,
                    constraint.min,
                    constraint.max,
                    self.s,
                    self.t,
                    self.eps,
                    constraint.isAngle,
                )
        return values

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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )

    @staticmethod
    def bus_from_polar(bus):
        """
        Convert bus voltage from polar to rectangular.
        """
        bus = bus.clone()
        V = torch.polar(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.real
        bus[:, 1, :] = V.imag
        return bus

    @staticmethod
    def bus_to_polar(bus):
        """
        Convert bus voltage from rectangular to polar.
        """
        bus = bus.clone()
        V = torch.complex(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.abs()
        bus[:, 1, :] = V.angle()
        return bus