import argparse
import subprocess
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data, HeteroData

import opf.powerflow as pf
from opf.constraints import equality, inequality
from opf.dataset import PowerflowBatch, PowerflowData
from opf.hetero import HeteroGCN


class OPFDual(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        n_nodes: tuple[int, int, int],
        lr=1e-4,
        weight_decay=0.0,
        lr_dual=1e-3,
        lr_common=1e-4,
        weight_decay_dual=0.0,
        eps=1e-3,
        enforce_constraints=False,
        detailed_metrics=False,
        multiplier_table_length=0,
        cost_weight=1.0,
        augmented_weight: float = 0.0,
        supervised_weight: float = 0.0,
        powerflow_weight: float = 0.0,
        warmup: int = 0,
        supervised_warmup: int = 0,
        common: bool = True,
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
            augmented_weight (float, optional): The weight of the augmented loss. Defaults to 0.0.
            supervised_weight (float, optional): The weight of the supervised loss. Defaults to 0.0.
            powerflow_weight (float, optional): The weight of the powerflow loss. Defaults to 0.0.
            warmup (int, optional): Number of epochs before starting to update the multipliers. Defaults to 0.
            supervised_warmup (int, optional): Number of epochs during which supervised loss is used. Defaults to 0.
            common (bool, optional): Whether to use a common multiplier for all constraints. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "kwargs"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_dual = lr_dual
        self.lr_common = lr_common
        self.weight_decay_dual = weight_decay_dual
        self.eps = eps
        self._enforce_constraints = enforce_constraints
        self.detailed_metrics = detailed_metrics
        self.automatic_optimization = False
        self.multiplier_table_length = multiplier_table_length
        self.cost_weight = cost_weight
        self.augmented_weight = augmented_weight
        self.supervised_weight = supervised_weight
        self.powerflow_weight = powerflow_weight
        self.warmup = warmup
        self.supervised_warmup = supervised_warmup
        self.common = common
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
            self.n_multipliers = int(self.multiplier_offsets[-1].item())

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

        self.multipliers_table = torch.nn.Embedding(
            num_embeddings=multiplier_table_length,
            embedding_dim=self.n_multipliers,
        )
        self.multipliers_table.weight.data.zero_()
        self.multipliers_common = Parameter(
            torch.zeros(self.n_multipliers, device=self.device)
        )

    def get_multipliers(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get the multiplier for the tensor corresponding to the given index.
        """
        # The fist entry is a shared multiplier common to all entries
        multipliers = torch.zeros(idx.shape[0], self.n_multipliers, device=self.device)
        if self.common:
            multipliers = multipliers + self.multipliers_common.expand(idx.shape[0], -1)
        if self.multiplier_table_length > 0:
            # The remaining entries are personalized multipliers one for each sample in the training dataset
            multipliers = multipliers + self.multipliers_table(idx)

        multiplier_dict = {}
        for data, (name, shape) in zip(
            torch.tensor_split(multipliers, self.multiplier_offsets, dim=1),
            self.multiplier_metadata.items(),
        ):
            multiplier_dict[name] = data.view((idx.shape[0],) + shape)
        return multiplier_dict

    def project_multipliers_table(self):
        """
        Project the inequality multipliers to be non-negative. Only the (sample-wise) table multipliers are projected.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        self.multipliers_table.weight.data[:, self.multiplier_inequality_mask] = (
            self.multipliers_table.weight.data[
                :, self.multiplier_inequality_mask
            ].relu_()
        )

    def project_multipliers_common(self):
        """
        Project the inequality multipliers to be non-negative. Only the common multipliers are projected.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        self.multipliers_common.data[self.multiplier_inequality_mask] = (
            self.multipliers_common.data[self.multiplier_inequality_mask].relu_()
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("OPFDual")
        group.add_argument("--lr", type=float, default=3e-4)
        group.add_argument("--weight_decay", type=float, default=0.0)
        group.add_argument("--lr_dual", type=float, default=0.1)
        group.add_argument("--lr_common", type=float, default=0.01)
        group.add_argument("--weight_decay_dual", type=float, default=0.0)
        group.add_argument("--eps", type=float, default=1e-3)
        group.add_argument("--enforce_constraints", action="store_true", default=False)
        group.add_argument("--detailed_metrics", action="store_true", default=False)
        group.add_argument("--cost_weight", type=float, default=1.0)
        group.add_argument("--augmented_weight", type=float, default=0.0)
        group.add_argument("--supervised_weight", type=float, default=0.0)
        group.add_argument("--powerflow_weight", type=float, default=0.0)
        group.add_argument("--warmup", type=int, default=0)
        group.add_argument("--supervised_warmup", type=int, default=0)
        group.add_argument("--no_common", dest="common", action="store_false")

    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
    ) -> tuple[pf.PowerflowVariables, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[pf.PowerflowVariables, torch.Tensor, torch.Tensor]:
              - The powerflow variables,
              - the predicted forward power,
              - and the predicted backward power.
        """
        data, powerflow_parameters, index = input
        if isinstance(data, HeteroData):
            n_batch = data["bus"].x.shape[0] // powerflow_parameters.n_bus
            if isinstance(self.model, HeteroGCN):
                # hetero GCN expect hetero graph in homogeneous form
                homo = data.to_homogeneous()
                homo.y = self.model(
                    homo.x, homo.edge_index, homo.node_type, homo.edge_type
                )
                y_dict = homo.to_heterogeneous().y_dict
            else:
                y_dict = self.model(data.x_dict, data.edge_index_dict)
            # reshape data to size (batch_size, n_nodes_of_type, n_features)
            load = data["bus"].x[:, :2].view(n_batch, powerflow_parameters.n_bus, 2)
            bus = y_dict["bus"].view(n_batch, powerflow_parameters.n_bus, 4)[..., :2]
            gen = y_dict["gen"].view(n_batch, powerflow_parameters.n_gen, 4)[..., :2]
            branch = y_dict["branch"].view(n_batch, powerflow_parameters.n_branch, 4)
        elif isinstance(data, Data):
            raise NotImplementedError("Removed support for homogenous data for now.")
        else:
            raise ValueError(
                f"Unsupported data type {type(data)}, expected Data or HeteroData."
            )
        V = self.parse_bus(bus)
        Sg = self.parse_gen(gen)
        Sd = self.parse_load(load)
        Sf_pred, St_pred = self.parse_branch(branch)
        if self._enforce_constraints:
            V, Sg = self.enforce_constraints(V, Sg, powerflow_parameters)
        return pf.powerflow(V, Sd, Sg, powerflow_parameters), Sf_pred, St_pred

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

    def supervised_loss(
        self,
        batch: PowerflowBatch,
        variables: pf.PowerflowVariables,
        Sf_pred: torch.Tensor,
        St_pred: torch.Tensor,
    ):
        """
        Calculate the MSE between the predicted and target bus voltage and generator power.
        """
        data, powerflow_parameters, _ = batch
        # parse auxiliary data from the batch
        V_target = data["bus"]["V"].view(-1, powerflow_parameters.n_bus, 2)
        Sg_target = data["gen"]["Sg"].view(-1, powerflow_parameters.n_gen, 2)
        Sf_target = data["branch"]["Sf"].view(-1, powerflow_parameters.n_branch, 2)
        St_target = data["branch"]["St"].view(-1, powerflow_parameters.n_branch, 2)
        # convert to complex numbers
        V_target = torch.complex(V_target[..., 0], V_target[..., 1])
        Sg_target = torch.complex(Sg_target[..., 0], Sg_target[..., 1])
        Sf_target = torch.complex(Sf_target[..., 0], Sf_target[..., 1])
        St_target = torch.complex(St_target[..., 0], St_target[..., 1])

        loss_voltage = (variables.V - V_target).abs().pow(2).mean()
        loss_gen = (variables.Sg - Sg_target).abs().pow(2).mean()
        loss_sf = (Sf_pred - Sf_target).abs().pow(2).mean()
        loss_st = (St_pred - St_target).abs().pow(2).mean()
        loss_power = loss_sf + loss_st
        loss_supervised = loss_voltage + loss_gen + loss_power
        self.log_dict(
            {
                "train/supervised_voltage": loss_voltage,
                "train/supervised_gen": loss_gen,
                "train/supervised_power": loss_power,
            },
            batch_size=batch.data.num_graphs,
        )
        self.log(
            "train/supervised_loss",
            loss_supervised,
            batch_size=batch.data.num_graphs,
        )
        return loss_supervised

    def powerflow_loss(
        self, batch: PowerflowBatch, variables: pf.PowerflowVariables, Sf_pred, St_pred
    ):
        """
        Loss between the predicted Sf and St by the model and the Sf and St implied by the powerflow equations.
        Assumes variables.Sf and variables.St are computed from powerflow(V).
        """
        Sf_loss = (variables.Sf - Sf_pred).abs().pow(2).mean()
        St_loss = (variables.St - St_pred).abs().pow(2).mean()
        powerflow_loss = Sf_loss + St_loss
        self.log(
            "train/powerflow_loss",
            powerflow_loss,
            batch_size=batch.data.num_graphs,
        )
        return powerflow_loss

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

    def on_train_epoch_start(self):
        _, dual_optimizer, _ = self.optimizers()  # type: ignore
        dual_optimizer.zero_grad()

    def on_train_epoch_end(self):
        _, dual_optimizer, _ = self.optimizers()  # type: ignore
        if self.current_epoch >= self.warmup:
            dual_optimizer.step()
            self.project_multipliers_table()

    def training_step(self, batch: PowerflowBatch):
        primal_optimizer, _, common_optimizer = self.optimizers()  # type: ignore
        variables, Sf_pred, St_pred = self(batch)
        _, constraints, cost = self._step_helper(
            variables, batch.powerflow_parameters, self.get_multipliers(batch.index)
        )
        constraint_loss = self.constraint_loss(constraints)

        supervised_loss = self.supervised_loss(batch, variables, Sf_pred, St_pred)
        # linearly decay the supervised loss until 0 at self.current_epoch > self.supervised_warmup
        supervised_weight = (
            max(1.0 - self.current_epoch / self.supervised_warmup, 0.0)
            if self.supervised_warmup > 0
            else 1.0
        )
        powerflow_loss = self.powerflow_loss(batch, variables, Sf_pred, St_pred)

        loss = (
            self.cost_weight * cost / batch.powerflow_parameters.n_gen
            + constraint_loss
            + self.powerflow_weight * powerflow_loss
            + supervised_weight * supervised_loss
        )

        primal_optimizer.zero_grad()
        common_optimizer.zero_grad()
        loss.backward()
        primal_optimizer.step()
        if self.current_epoch >= self.warmup:
            common_optimizer.step()
            self.project_multipliers_common()

        self.log(
            "train/loss",
            loss,
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
            self(batch)[0], batch.powerflow_parameters
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
            self(batch),
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
        assert bus.shape[-1] == 2
        V = torch.complex(bus[..., 0], bus[..., 1])
        return V

    def parse_load(self, load: torch.Tensor):
        """
        Converts the load data to the format required by the powerflow module (complex tensor).

        Args:
            load: A tensor of shape (batch_size, n_features, n_bus). The first two features should contain the active and reactive load.

        """
        assert load.shape[-1] == 2
        Sd = torch.complex(load[..., 0], load[..., 1])
        return Sd

    def parse_gen(self, gen: torch.Tensor):
        assert gen.shape[-1] == 2
        Sg = torch.complex(gen[..., 0], gen[..., 1])
        return Sg

    def parse_branch(self, branch: torch.Tensor):
        assert branch.shape[-1] == 4
        Sf = torch.complex(branch[..., 0], branch[..., 1])
        St = torch.complex(branch[..., 2], branch[..., 3])
        return Sf, St

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
                    self.augmented_weight if constraint.augmented else 0.0,
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
                    self.augmented_weight if constraint.augmented else 0.0,
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
        primal_optimizer = torch.optim.AdamW(  # type: ignore
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        dual_optimizer = torch.optim.AdamW(  # type: ignore
            self.multipliers_table.parameters(),
            lr=self.lr_dual,
            weight_decay=self.weight_decay_dual,
            maximize=True,
            fused=True,
        )
        common_optimizer = torch.optim.AdamW(  # type: ignore
            [self.multipliers_common],
            lr=self.lr_common,
            weight_decay=self.weight_decay_dual,
            fused=True,
            maximize=True,
        )

        optimizers = [primal_optimizer, dual_optimizer, common_optimizer]
        return optimizers

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
