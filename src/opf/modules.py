from functools import reduce
from math import exp
from typing import List, Dict

import alegnn.utils.graphML as gml
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn
from alegnn.modules.architectures import SelectionGNN

import opf.powerflow as pf
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

        # Parse parameters such as admittance matrix to be used in powerflow calculations.
        self.powerflow_parameters = pf.parameters_from_pm(self.pm)

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
        bus = torch.reshape(bus, (-1, 4, self.powerflow_parameters.n_bus))
        return bus

    def _step_helper(self, bus, load, project_pandapower=False):
        V, S = self.parse_bus(bus)
        Sd = self.parse_load(load)
        if project_pandapower:
            V, S = self.project_pandapower(V, S, Sd)
        variables = pf.powerflow(V, S, Sd, self.powerflow_parameters)
        constraints = self.constraints(variables)
        cost = self.cost(variables)
        loss = self.loss(cost, constraints)
        return variables, constraints, cost, loss

    def training_step(self, batch, *args):
        load = batch[0] @ self.powerflow_parameters.load_matrix
        _, constraints, cost, loss = self._step_helper(
            self(load), load, project_pandapower=False
        )
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, *args):
        with torch.no_grad():
            load = batch[0] @ self.powerflow_parameters.load_matrix
            _, constraints, cost, loss = self._step_helper(
                self(load), load, project_pandapower=False
            )
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            self.log_dict(
                self.metrics(cost, constraints, "val", self.detailed_metrics),
                sync_dist=True,
            )

    def test_step(self, batch, *args):
        with torch.no_grad():
            load, acopf_bus = batch
            load @= self.powerflow_parameters.load_matrix
            _, constraints, cost, _ = self._step_helper(
                self(load), load, project_pandapower=True
            )
            test_metrics = self.metrics(
                cost, constraints, "test", self.detailed_metrics
            )
            self.log_dict(test_metrics)

            # Test the ACOPF solution for reference.
            _, constraints, cost, _ = self._step_helper(
                self.bus_from_polar(acopf_bus), load, project_pandapower=False
            )
            acopf_metrics = self.metrics(
                cost, constraints, "acopf", self.detailed_metrics
            )
            self.log_dict(acopf_metrics)
            return dict(**test_metrics, **acopf_metrics)


    def parse_bus(self, bus: torch.Tensor):
        assert bus.shape[1] == 4
        assert bus.shape[2] == self.powerflow_parameters.n_bus

        # Convert voltage and power to per unit
        vr = bus[:, 0, :]
        vi = bus[:, 1, :]
        p = bus[:, 2, :]
        q = bus[:, 3, :]

        V = torch.complex(vr, vi).unsqueeze(-1)
        S = torch.complex(p, q).unsqueeze(-1)
        return V, S

    def parse_load(self, load: torch.Tensor):
        assert load.shape[1] == 2
        assert load.shape[2] == self.powerflow_parameters.n_bus
        Sd = torch.complex(load[:, 0, :], load[:, 1, :]).unsqueeze(-1)
        return Sd

    def loss(self, cost, constraints):
        constraint_losses = [
            val["loss"]
            for val in constraints.values()
            if val["loss"] is not None and not torch.isnan(val["loss"])
        ]
        return cost * self.cost_weight + torch.stack(constraint_losses).sum()

    def cost(self, variables: pf.PowerflowVariables) -> torch.Tensor:
        """Compute the cost to produce the active and reactive power."""
        p = variables.S.real
        p_coeff = self.powerflow_parameters.cost_coeff
        return (p_coeff[:, 0] + p * p_coeff[:, 1] + (p ** 2) * p_coeff[:, 2]).mean()

    def constraints(self, variables) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Nested map from constraint name => (value name => tensor value)
        """
        values = {}
        for name, constraint in self.powerflow_parameters.constraints.items():
            if isinstance(constraint, pf.EqualityConstraint):
                values[name] = equality(
                    constraint.value(self.powerflow_parameters, variables),
                    constraint.target(self.powerflow_parameters, variables),
                    self.eps,
                    constraint.isAngle,
                )
            elif isinstance(constraint, pf.InequalityConstraint):
                values[name] = inequality(
                    constraint.variable(self.powerflow_parameters, variables),
                    constraint.min,
                    constraint.max,
                    self.s,
                    self.t,
                    self.eps,
                    constraint.isAngle,
                )
        return values

    @property
    def bus_constraints_matrix(self):
        """Returns a matrix representing the bus constraints as a graph signal."""
        bus_constraints = []
        for constraint in self.powerflow_parameters.constraints.values():
            if constraint.isBus:
                bus_constraints += [constraint.min, constraint.max]
        return torch.cat(bus_constraints, dim=1).to(self.device)

    @property
    def branch_constraints_matrix(self):
        """Returns a matrix representing the branch constraint features.
        The matrix size is # branches x # branch constraints"""
        branch_constraints = []
        for constraint in self.powerflow_parameters.constraints.values():
            if constraint.isBranch:
                branch_constraints += [constraint.min, constraint.max]
        return torch.stack(branch_constraints, dim=0)

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

    def project_pandapower(self, V: torch.Tensor, S: torch.Tensor, Sd: torch.Tensor):
        with torch.no_grad():
            Sg = S + Sd
            self.net_wrapper.set_gen_sparse(
                Sg.real.squeeze().cpu().numpy(), Sg.imag.squeeze().cpu().numpy()
            )
            self.net_wrapper.set_load_sparse(
                Sd.real.squeeze().cpu().numpy(), Sd.imag.squeeze().cpu().numpy()
            )
            res_powerflow = self.net_wrapper.powerflow()
            if res_powerflow is None:
                return V, S
            else:
                bus, _, _ = res_powerflow
                bus = torch.as_tensor(bus, device=self.device, dtype=self.dtype)
                bus = self.bus_from_polar(bus.unsqueeze(0))
                return self.parse_bus(bus)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.hparams.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, []),
            "monitor": "val/loss",
            "name": "scheduler",
        }
        return [optimizer], [lr_scheduler]

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
