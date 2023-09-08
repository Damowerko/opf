import argparse
from typing import Dict

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

import opf.powerflow as pf
from opf.constraints import equality, inequality
from opf.dataset import PowerflowBatch, PowerflowData


class OPFLogBarrier(pl.LightningModule):
    def __init__(
        self,
        model,
        t=500,
        s=100,
        cost_weight=1.0,
        equality_weight=1.0,
        inequality_weight=1.0,
        lr=1e-4,
        weight_decay=0.0,
        eps=1e-3,
        enforce_constraints=False,
        detailed_metrics=False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.detailed_metrics = detailed_metrics
        self.t = t
        self.s = s
        self.cost_weight = cost_weight
        self.equality_weight = equality_weight
        self.inequality_weight = inequality_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self._enforce_constraints = enforce_constraints
        self.save_hyperparameters(ignore=["model", "kwargs"])

        # Normalization factor to be applied to the cost function
        self.cost_normalization = 1.0

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("OPFLogBarrier")
        group.add_argument("--s", type=int, default=100)
        group.add_argument("--t", type=int, default=500)
        group.add_argument("--cost_weight", type=float, default=0.01)
        group.add_argument("--equality_weight", type=float, default=1.0)
        group.add_argument("--inequality_weight", type=float, default=0.1)
        group.add_argument("--lr", type=float, default=3e-4)
        group.add_argument("--weight_decay", type=float, default=0.0)
        group.add_argument("--eps", type=float, default=1e-3)
        group.add_argument("--enforce_constraints", action="store_true", default=False)
        group.add_argument("--detailed_metrics", action="store_true", default=False)

    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
    ):
        data, powerflow_parameters = input
        assert isinstance(data, Data)
        n_batch = data.x.shape[0] // powerflow_parameters.n_bus
        bus = self.model(data.x, data.edge_index, data.edge_attr)

        # Reshape the output to (batch_size, n_features, n_bus)
        # Where n_features is 4 and 2 for bus and load respectively.
        bus = bus.view(n_batch, powerflow_parameters.n_bus, 4).mT
        # assume that the first two features are the active and reactive load
        load = data.x[:, :2].view(n_batch, powerflow_parameters.n_bus, 2).mT
        V, Sg = self.parse_bus(bus)
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
    ):
        variables = pf.powerflow(V, Sg, Sd, parameters)
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
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.metrics(cost, constraints, "train", self.detailed_metrics))
        return loss

    def validation_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            _, constraints, cost, loss = self._step_helper(
                *self.forward(batch), batch.powerflow_parameters
            )
            self.log("val/loss", loss, prog_bar=True)
            self.log_dict(self.metrics(cost, constraints, "val", self.detailed_metrics))

    def test_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            _, constraints, cost, loss = self._step_helper(
                *self.forward(batch),
                batch.powerflow_parameters,
                substitute_equality=True,
            )
            test_metrics = self.metrics(
                cost, constraints, "test", self.detailed_metrics
            )
            self.log_dict(test_metrics)
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

    def parse_bus(self, bus: torch.Tensor):
        assert bus.shape[1] == 4

        # Convert voltage and power to per unit
        vr = bus[:, 0, :]
        vi = bus[:, 1, :]
        pg = bus[:, 2, :]
        qg = bus[:, 3, :]

        V = torch.complex(vr, vi)
        Sg = torch.complex(pg, qg)
        return V, Sg

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
        return (
            cost * self.cost_weight * self.cost_normalization
            + torch.stack(constraint_losses).sum()
        )

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
        # normalize the cost by the number of generators
        return cost.mean(0).sum()

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
                # apply weight
                values[name]["loss"] *= self.inequality_weight
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
