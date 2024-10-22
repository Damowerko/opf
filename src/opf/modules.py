import argparse
import subprocess
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

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
        multiplier_table_length=0,
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
            multiplier_table_length (int, optional): The number of entries in the multiplier table. Defaults to 0.
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
        self.multiplier_table_length = multiplier_table_length

        # setup the multipliers
        n_bus, n_branch, n_gen = n_nodes
        self.init_multipliers(n_bus, n_gen, n_branch, multiplier_table_length)

    def init_multipliers(self, n_bus, n_gen, n_branch, multiplier_table_length: int):
        with torch.no_grad():
            self.multiplier_metadata = OrderedDict(
                [
                    ("equality/bus_active_power", torch.Size([n_bus])),
                    ("equality/bus_reactive_power", torch.Size([n_bus])),
                    ("equality/bus_reference", torch.Size([n_bus])),
                    ("inequality/voltage_magnitude", torch.Size([n_bus, 2])),
                    ("inequality/active_power", torch.Size([n_gen, 2])),
                    ("inequality/reactive_power", torch.Size([n_gen, 2])),
                    ("inequality/forward_rate", torch.Size([n_branch, 2])),
                    ("inequality/backward_rate", torch.Size([n_branch, 2])),
                    ("inequality/voltage_angle_difference", torch.Size([n_branch, 2])),
                ]
            )
            self.multiplier_numel = torch.tensor(
                [x.numel() for _, x in self.multiplier_metadata.items()]
            )
            self.multiplier_offsets = torch.cumsum(self.multiplier_numel, 0)

            multiplier_inequality_mask_list = []
            for name, numel in zip(
                self.multiplier_metadata.keys(), self.multiplier_numel
            ):
                if "inequality" in name:
                    multiplier_inequality_mask_list.append(
                        torch.ones(int(numel.item()), dtype=torch.bool)
                    )
                else:
                    multiplier_inequality_mask_list.append(
                        torch.zeros(int(numel.item()), dtype=torch.bool)
                    )
            self.register_buffer(
                "multiplier_inequality_mask",
                torch.cat(multiplier_inequality_mask_list),
                persistent=False,
            )

        self.multipliers = torch.nn.Embedding(
            num_embeddings=multiplier_table_length + 1,
            embedding_dim=int(self.multiplier_offsets[-1].item()),
        )
        self.multipliers.weight.data.zero_()

    def get_multipliers(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get the multiplier for the tensor corresponding to the given index.
        """
        # The fist entry is a shared multiplier common to all entries
        common = self.multipliers(torch.tensor([0], device=idx.device))
        if self.multiplier_table_length > 0:
            # The remaining entries are personalized multipliers one for each sample in the training dataset
            personalized = self.multipliers(idx + 1)
            multipliers = common + personalized
        else:
            # If there are no personalized multipliers, just use the common multiplier
            multipliers = common.expand(idx.shape[0], -1)

        multiplier_dict = {}
        for data, (name, shape) in zip(
            torch.tensor_split(multipliers, self.multiplier_offsets, dim=1),
            self.multiplier_metadata.items(),
        ):
            multiplier_dict[name] = data.view((idx.shape[0],) + shape)
        return multiplier_dict

    def project_multipliers(self):
        """
        Project the inequality multipliers to be non-negative.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        self.multipliers.weight.data[:, self.multiplier_inequality_mask] = (
            self.multipliers.weight.data[:, self.multiplier_inequality_mask].relu_()
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
        data, powerflow_parameters, index = input
        if isinstance(data, HeteroData):
            n_batch = data["bus"].x.shape[0] // powerflow_parameters.n_bus
            load = data["bus"].x[:, :2]
            bus, gen = self.model(data)
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
        multipliers: Dict[str, torch.Tensor] | None = None,
        project_powermodels=False,
    ):
        if project_powermodels:
            variables = self.project_powermodels(variables, parameters)
        constraints = self.constraints(variables, parameters, multipliers)
        cost = self.cost(variables, parameters)
        return variables, constraints, cost

    def training_step(self, batch: PowerflowBatch):
        primal_optimizer, dual_optimizer = self.optimizers()  # type: ignore
        multipliers = self.get_multipliers(batch.index)
        _, constraints, cost = self._step_helper(
            self.forward(batch), batch.powerflow_parameters, multipliers
        )
        constraint_loss = self.constraint_loss(constraints)

        primal_optimizer.zero_grad()
        dual_optimizer.zero_grad()
        (cost + 100 * constraint_loss).backward()
        primal_optimizer.step()
        if (self.global_step + 1) % self.dual_interval == 0:
            dual_optimizer.step()

        # enforce inequality multipliers to be non-negative
        self.project_multipliers()

        self.log(
            "train/loss",
            cost + constraint_loss,
            prog_bar=True,
            batch_size=batch.data.num_graphs,
            sync_dist=True,
        )
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics, train=True),
            batch_size=batch.data.num_graphs,
            sync_dist=True,
        )

    def validation_step(self, batch: PowerflowBatch, *args):
        batch_size = batch.data.num_graphs
        _, constraints, cost = self._step_helper(
            self.forward(batch), batch.powerflow_parameters
        )
        metrics = self.metrics(cost, constraints, "val", self.detailed_metrics)
        self.log_dict(metrics, batch_size=batch_size, sync_dist=True)

        # Metric that does not depend on the loss function shape
        self.log(
            "val/invariant",
            cost
            + 1e3 * metrics["val/equality/error_mean"]
            + 1e3 * metrics["val/inequality/error_mean"],
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch: PowerflowBatch, *args):
        # TODO
        # change to make faster
        # project_powermodels taking too long
        # go over batch w/ project pm, then individual steps without
        _, constraints, cost = self._step_helper(
            self.forward(batch),
            batch.powerflow_parameters,
            project_powermodels=True,
        )
        test_metrics = self.metrics(cost, constraints, "test", self.detailed_metrics)
        self.log_dict(
            test_metrics,
            batch_size=batch.data.num_graphs,
            sync_dist=True,
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
        clamp=True,
    ) -> pf.PowerflowVariables:
        V, Sg, Sd = variables.V, variables.Sg, variables.Sd
        if clamp:
            V, Sg = self.enforce_constraints(V, Sg, parameters, strategy="clamp")
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
        # cost = torch.clamp(cost, min=0)
        # # normalize the cost by the number of generators
        return cost.mean(0).sum() / powerflow_parameters.reference_cost

    def constraints(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
        multipliers: Dict[str, torch.Tensor] | None,
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
                    multipliers[name] if multipliers is not None else None,
                    constraint.mask,
                    self.eps,
                    constraint.isAngle,
                )
            elif isinstance(constraint, pf.InequalityConstraint):
                values[name] = inequality(
                    constraint.variable,
                    constraint.min,
                    constraint.max,
                    multipliers[name][..., 0] if multipliers is not None else None,
                    multipliers[name][..., 1] if multipliers is not None else None,
                    self.eps,
                    constraint.isAngle,
                )
        return values

    def metrics(self, cost, constraints, prefix, detailed=False, train=False):
        """
        Args:
            cost: The cost of the powerflow.
            constraints: The constraints of the powerflow.
            prefix: The prefix to use for the metric names.
            detailed: Whether to log detailed
            train: Whether the metrics are for training or validation/test.
        """

        aggregate_metrics = {
            f"{prefix}/cost": [cost],
            f"{prefix}/equality/rate": [],
            f"{prefix}/equality/error_mean": [],
            f"{prefix}/equality/error_max": [],
            f"{prefix}/inequality/rate": [],
            f"{prefix}/inequality/error_mean": [],
            f"{prefix}/inequality/error_max": [],
        }
        if train:
            aggregate_metrics.update(
                **{
                    f"{prefix}/equality/loss": [],
                    f"{prefix}/equality/multiplier/mean": [],
                    f"{prefix}/equality/multiplier/max": [],
                    f"{prefix}/equality/multiplier/min": [],
                    f"{prefix}/inequality/loss": [],
                    f"{prefix}/inequality/multiplier/mean": [],
                    f"{prefix}/inequality/multiplier/max": [],
                    f"{prefix}/inequality/multiplier/min": [],
                }
            )

        detailed_metrics = {}
        reduce_fn = {
            "default": torch.sum,
            "error_mean": torch.mean,
            "error_max": torch.max,
            "rate": torch.mean,
            "multiplier/mean": torch.mean,
            "multiplier/max": torch.max,
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
        primal_optimizer = torch.optim.Adam(  # type: ignore
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        dual_optimizer = torch.optim.SGD(  # type: ignore
            self.multipliers.parameters(),
            lr=self.lr_dual,
            weight_decay=self.weight_decay_dual,
            maximize=True,
            fused=True,
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
