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

import opf.katya.k_powerflow as pf
from opf.katya.k_constraints import equality, inequality
from opf.dataset import PowerflowBatch, PowerflowData

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
        # self.save_hyperparameters(ignore=["model", "kwargs"])
    

    # def __init__(
    #       self,
    #       s,
    #       s_step,
    #       t,
    #       t_step,
    #       equality_weight,
    #       equality_step,
    #       lr,
    #       eps,
    #       model = torch.nn.Module,
    # ):
    #     """
    #     Inputs:
    #         s: 
    #     """
    #     super().__init__()
    #     self.s_start = s
    #     self.s_step = s_step
    #     self.t_start = s
    #     self.t_step = s_step
    #     self.eq_start = equality_weight
    #     self.eq_step = equality_step
    #     self.lr = lr
    #     self.eps = eps
    #     self.model = model


    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
    ):
        """
        Inputs:
            input (PowerflowBatch | PowerflowData): batch of data
        Outputs:
            V: bus voltage
            Sg: power generated
            Sd: load demanded
        """
        data, powerflow_parameters = input
        n_batch = data.x.shape[0] // powerflow_parameters.n_bus
        bus = self.model(data.x, data.edge_index, data.edge_attr)
        load = data.x[:, :2]

        # Reshape the output to (batch_size, n_features, n_bus)
        # Where n_features is 4 and 2 for bus and load respectively.
        bus = bus.view(n_batch, powerflow_parameters.n_bus, 4).mT
        # Similar shape for load
        load = load.view(n_batch, powerflow_parameters.n_bus, 2).mT
        V, Sg = self.parse_bus(bus)
        Sd = self.parse_load(load)
        if self._enforce_constraints:
            V, Sg = self.enforce_constraints(V, Sg, powerflow_parameters)
        return V, Sg, Sd
    

    def sigmoid_bound(self, x, lb, ub):
        """
        Inputs:
            x: input value
            lb: lower bound
            ub: upper bound
        Outputs:
            bounded x
        """
        scale = ub - lb
        return scale * torch.sigmoid(x) + lb
    

    def constraints(self, variables: pf.PowerFlowVariables, parameters: pf.PowerFlowParameters):
        """
        Input:
            variables (PowerFlowVariables):
            parameters (PowerFlowParameters):
        Output:
            values (Dict[str, Dict[str, torch.Tensor]]):
        """
        constraints = pf.build_constraints(variables, parameters)
        values = {}
        for name, constraint in constraints.items():
            if isinstance(constraint, pf.EqualityConstraint):
                values[name] = equality(
                    value = constraint.value,
                    target = constraint.target,
                    eps = self.eps
                )
        for name, constraint in constraints.items():
            if isinstance(constraint, pf.InequalityConstraint):
                values[name] = inequality(
                    value = constraint.value,
                    low_bound = constraint.lower_bound,
                    high_bound = constraint.upper_bound,
                    eps = self.eps
                )
        return values
    

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


    def enforce_constraints(self, V, Sg, params: pf.PowerFlowParameters):
        """
        Inputs:
            V: bus voltage
            Sg: power generated
            params (PowerFlowParameters): parameters
        Outputs:
            V: bus voltage with constraints applied
            Sg: power generated with constraints applied
        """
        vm = self.sigmoid_bound(V.abs(), params.vm_min, params.vm_max)
        V = torch.polar(vm, V.angle())  # V * vm / V.abs()
        Sg.real = self.sigmoid_bound(Sg.real, params.Sg_min.real, params.Sg_max.real)
        Sg.imag = self.sigmoid_bound(Sg.imag, params.Sg_min.imag, params.Sg_max.imag)
        return V, Sg
    

    def cost(self, variables: pf.PowerFlowVariables, parameters: pf.PowerFlowParameters):
        """
        Overview: 
            this function outputs a cost based on the power generated
            and the cost coefficient
        Inputs:
            variables (PowerFlowVariables): where you get Sg
            parameters (PowerFlowParameters): where you get cost_coeff
        Outputs:
            cost (torch.Tensor): a tensor of costs    
        """
        Sg = variables.Sg
        cost_coeff = parameters.cost_coeff
        cost = torch.zeros_like(Sg)
        for i in range(cost_coeff.shape[1]):
            # DAMIAN'S WAY:
            # cost += cost_coeff[:, i] * Sg.squeeze() ** i
            cost += cost_coeff[:, i] * Sg.squeeze()
            # ensure that cost is positive...
        # DAMIAN'S WAY:
        # return cost.mean(0).sum() / parameters.reference_cost
        return cost.mean(0).sum()


    def loss(self, cost, constraints):
        """
        Inputs:
            cost: cost fo generation
            constraints: the nested dictionary of violated constraints
        Outputs:
            loss: the sum of cost and constraint losses
        """
        constraint_losses = [
            val["loss"]
            for val in constraints.values()
            if val["loss"] is not None and not torch.isnan(val["loss"])
        ]
        if len(constraint_losses) == 0:
            constraint_losses = [torch.zeros(1, device=self.device, dtype=self.dtype)]  # type: ignore
        return cost + torch.stack(constraint_losses).sum()
        
        
    def _step_helper(self, V, Sg, Sd, params: pf.PowerFlowParameters):
        """
        Input:
            V: voltages
            Sg: supplies
            Sd: loads
            params: PowerFlowParameters
        Output:
            variables
            cost
            loss
            constraints
        """
        variables = pf.powerflow(V, Sg, Sd, params)
        constraints = self.constraints(variables, params)
        cost = self.cost(variables, params)
        loss = self.loss(cost, params)
        return variables, constraints, cost, loss
    

    @property
    def s(self):
        return self.s_start + self.s_step * self.current_epoch
    

    @property
    def t(self):
        return self.t_start + self.t_step * self.current_epoch


    @property
    def equality_weight(self):
        return self.eq_start + self.eq_step * self.current_epoch


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
            return test_metrics


    def configure_optimizers(self):
        """
        Overview:
            set optimizer to AdamW with parameters, lr,
            and weight_decay from self
        """
        return torch.optim.AdamW(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )


    def metrics(self, cost, constraints, prefix, detailed=False):
        """
        GOOD LORD >:0!!!!!!!!!!!
        """
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


    def parse_bus(self, bus: torch.Tensor):
        """
        Inputs:
            bus (torch.Tensor): bus data
        Outputs:
            V: bus voltage
            Sg: power generated
        """
        assert bus.shape[1] == 4

        V = torch.complex(bus[:, 0, :], bus[:, 1, :])
        Sg = torch.complex(bus[:, 2, :], bus[:, 3, :])
        return V, Sg

    def parse_load(self, load: torch.Tensor):
        """
        Inputs:
            load (torch.Tensor): load data
        Outputs:
            Sd: power demanded
        """
        assert load.shape[1] == 2

        Sd = torch.complex(load[:, 0, :], load[:, 1, :])
        return Sd