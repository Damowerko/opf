import argparse
import subprocess
import typing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data, HeteroData

import opf.powerflow as pf
from opf.constraints import equality, inequality
from opf.dataset import PowerflowBatch, PowerflowData


class OPFDual(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        n_nodes: tuple[int, int, int],
        lr=1e-4,
        weight_decay=0.0,
        lr_dual=1e-3,
        weight_decay_dual=0.0,
        dual_interval=1,
        eps=1e-3,
        enforce_constraints=False,
        detailed_metrics=False,
        **kwargs,
    ):
        """
        Args:
            model (torch.nn.Module): The model to be trained.
            n_nodes (tuple[int, int, int]): A tuple containing the number of buses, branches, and generators.
            lr (float, optional): The learning rate. Defaults to 1e-4.
            weight_decay (float, optional): The weight decay. Defaults to 0.0.
            eps (float, optional): Numerical threshold, below which values will be assumed to be approximately zero. Defaults to 1e-3.
            enforce_constraints (bool, optional): Whether to enforce the constraints. Defaults to False.
            detailed_metrics (bool, optional): Whether to log detailed metrics. Defaults to False.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "kwargs"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_dual = lr_dual
        self.weight_decay_dual = weight_decay_dual
        self.dual_interval = dual_interval
        self.eps = eps
        self._enforce_constraints = enforce_constraints
        self.detailed_metrics = detailed_metrics
        self.automatic_optimization = False

        # setup the multipliers
        n_bus, n_branch, n_gen = n_nodes
        self.multipliers = torch.nn.ParameterDict(
            {
                "equality/bus_active_power": Parameter(
                    torch.zeros([n_bus], device=self.device)
                ),
                "equality/bus_reactive_power": Parameter(
                    torch.zeros([n_bus], device=self.device)
                ),
                "equality/bus_reference": Parameter(
                    torch.zeros([n_bus], device=self.device)
                ),
                "inequality/voltage_magnitude": Parameter(
                    torch.zeros([n_bus, 2], device=self.device)
                ),
                "inequality/active_power": Parameter(
                    torch.zeros([n_gen, 2], device=self.device)
                ),
                "inequality/reactive_power": Parameter(
                    torch.zeros([n_gen, 2], device=self.device)
                ),
                "inequality/forward_rate": Parameter(
                    torch.zeros([n_branch, 2], device=self.device)
                ),
                "inequality/backward_rate": Parameter(
                    torch.zeros([n_branch, 2], device=self.device)
                ),
                "inequality/voltage_angle_difference": Parameter(
                    torch.zeros([n_branch, 2], device=self.device)
                ),
            }
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("OPFDual")
        group.add_argument("--lr", type=float, default=3e-4)
        group.add_argument("--weight_decay", type=float, default=0.0)
        group.add_argument("--lr_dual", type=float, default=0.1)
        group.add_argument("--weight_decay_dual", type=float, default=0.0)
        group.add_argument("--dual_interval", type=int, default=1)
        group.add_argument("--eps", type=float, default=1e-3)
        group.add_argument("--enforce_constraints", action="store_true", default=False)
        group.add_argument("--detailed_metrics", action="store_true", default=False)

    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
    ) -> pf.PowerflowVariables:
        data, powerflow_parameters = input
        if isinstance(data, HeteroData):
            n_batch = data["bus"].x.shape[0] // powerflow_parameters.n_bus
            output = self.model(data.x_dict, data.adj_t_dict)
            bus = output["bus"][:, :2]
            gen = output["gen"][:, :2]
            load = data["bus"].x[:, :2]
        elif isinstance(data, Data):
            raise NotImplementedError("Removed support for homogenous data for now.")
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
        return pf.powerflow(V, Sd, Sg, powerflow_parameters)

    def enforce_constraints(
        self, V, Sg, params: pf.PowerflowParameters, strategy="sigmoid"
    ):
        """
        Ensure that voltage and power generation are within the specified bounds.

        Args:
            V: The bus voltage. Magnitude must be between params.vm_min and params.vm_max.
            Sg: The generator power. Real and reactive power must be between params.Sg_min and params.Sg_max.
            params: The powerflow parameters.
            strategy: The strategy to use for enforcing the constraints. Defaults to "sigmoid".
                "sigmoid" uses a sigmoid function to enforce the constraints.
                "clamp" uses torch.clamp to enforce the constraints.
        """
        if strategy == "sigmoid":
            fn = lambda x, lb, ub: (ub - lb) * torch.sigmoid(x) + lb
        elif strategy == "clamp":
            fn = lambda x, lb, ub: torch.clamp(x, lb, ub)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        vm = fn(V.abs(), params.vm_min, params.vm_max)
        V = torch.polar(vm, V.angle())
        Sg = torch.complex(
            fn(Sg.real, params.Sg_min.real, params.Sg_max.real),
            fn(Sg.imag, params.Sg_min.imag, params.Sg_max.imag),
        )
        return V, Sg

    def _step_helper(
        self,
        variables: pf.PowerflowVariables,
        parameters: pf.PowerflowParameters,
        project_powermodels=False,
    ):
        if project_powermodels:
            variables = self.project_powermodels(variables, parameters)
        constraints = self.constraints(variables, parameters)
        cost = self.cost(variables, parameters)
        constraint_loss = self.constraint_loss(constraints)
        return variables, constraints, cost, constraint_loss

    def training_step(self, batch: PowerflowBatch):
        primal_optimizer, dual_optimizer = self.optimizers()  # type: ignore
        _, constraints, cost, constraint_loss = self._step_helper(
            self.forward(batch),
            batch.powerflow_parameters,
        )

        primal_optimizer.zero_grad()
        dual_optimizer.zero_grad()
        (cost + 1e3 * constraint_loss).backward()
        primal_optimizer.step()
        if (self.global_step + 1) % self.dual_interval == 0:
            dual_optimizer.step()

        # enforce inequality multipliers to be non-negative
        for name in self.multipliers:
            if not name.startswith("inequality/"):
                continue
            self.multipliers[name].data.relu_()

        self.log(
            "train/loss",
            cost + constraint_loss,
            prog_bar=True,
            batch_size=batch.data.num_graphs,
        )
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics),
            batch_size=batch.data.num_graphs,
        )

    def validation_step(self, batch: PowerflowBatch, *args):
        with torch.no_grad():
            batch_size = batch.data.num_graphs
            _, constraints, cost, constraint_loss = self._step_helper(
                self.forward(batch),
                batch.powerflow_parameters,
            )
            self.log(
                "val/loss",
                cost + constraint_loss,
                batch_size=batch_size,
            )
            metrics = self.metrics(cost, constraints, "val", self.detailed_metrics)
            self.log_dict(metrics, batch_size=batch_size)

            # Metric that does not depend on the loss function shape
            self.log(
                "val/invariant",
                cost
                + 1e3 * metrics["val/equality/error_mean"]
                + 1e3 * metrics["val/inequality/error_mean"],
                batch_size=batch_size,
                prog_bar=True,
            )

    def test_step(self, batch: PowerflowBatch, *args):
        # TODO
        # change to make faster
        # project_powermodels taking too long
        # go over batch w/ project pm, then individual steps without
        with torch.no_grad():
            _, constraints, cost, _ = self._step_helper(
                self.forward(batch),
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
        variables: pf.PowerflowVariables,
        parameters: pf.PowerflowParameters,
    ) -> pf.PowerflowVariables:
        V, Sg, Sd = variables.V, variables.Sg, variables.Sd
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
        return pf.powerflow(V, Sd, Sg, parameters)

    def parse_bus(self, bus: torch.Tensor):
        assert bus.shape[1] == 2
        V = torch.complex(bus[:, 0, :], bus[:, 1, :])
        return V

    def parse_load(self, load: torch.Tensor):
        """
        Converts the load data to the format required by the powerflow module (complex tensor).

        Args:
            load: A tensor of shape (batch_size, n_features, n_bus). The first two features should contain the active and reactive load.

        """
        assert load.shape[1] == 2
        Sd = torch.complex(load[:, 0, :], load[:, 1, :])
        return Sd

    def parse_gen(self, gen: torch.Tensor):
        assert gen.shape[1] == 2
        Sg = torch.complex(gen[:, 0, :], gen[:, 1, :])
        return Sg

    def constraint_loss(self, constraints) -> torch.Tensor:
        constraint_losses = [
            val["loss"]
            for val in constraints.values()
            if val["loss"] is not None and not torch.isnan(val["loss"])
        ]
        if len(constraint_losses) == 0:
            return torch.zeros(1, device=self.device)
        return torch.stack(constraint_losses).sum()

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
                    self.multipliers[name],
                    constraint.mask,
                    self.eps,
                    constraint.isAngle,
                )
            elif isinstance(constraint, pf.InequalityConstraint):
                values[name] = inequality(
                    constraint.variable,
                    constraint.min,
                    constraint.max,
                    self.multipliers[name][..., 0],
                    self.multipliers[name][..., 1],
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
            f"{prefix}/equality/multiplier_mean": [],
            f"{prefix}/equality/multiplier_max": [],
            f"{prefix}/inequality/loss": [],
            f"{prefix}/inequality/rate": [],
            f"{prefix}/inequality/error_mean": [],
            f"{prefix}/inequality/error_max": [],
            f"{prefix}/inequality/multiplier_mean": [],
            f"{prefix}/inequality/multiplier_max": [],
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
        primal_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        dual_optimizer = torch.optim.Adam(
            self.multipliers.parameters(),
            lr=self.lr_dual,
            weight_decay=self.weight_decay_dual,
            maximize=True,
        )
        return primal_optimizer, dual_optimizer

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
