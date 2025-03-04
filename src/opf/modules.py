import abc
import logging
import subprocess
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.optimizer import Optimizer
from torch_geometric.data import HeteroData
from torchcps.utils import add_model_specific_args

import opf.powerflow as pf
from opf.constraints import equality, inequality
from opf.dataset import PowerflowData
from opf.models.base import OPFModel

logger = logging.getLogger(__name__)


class NullOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super().__init__([nn.Parameter(torch.zeros(0))], {})

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DualModel(nn.Module, abc.ABC):
    def forward(self, data: PowerflowData):
        return self.get_multipliers(data)

    @abc.abstractmethod
    def get_multipliers(self, data: PowerflowData) -> dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def parameters_shared(self) -> list[nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def parameters_pointwise(self) -> list[nn.Parameter]:
        raise NotImplementedError()


class DualTable(DualModel):
    def __init__(
        self,
        n_bus: int,
        n_gen: int,
        n_branch: int,
        n_train: int,
        enable_shared: bool,
        enable_pointwise: bool,
    ):
        super().__init__()
        self._init_metadata(n_bus, n_gen, n_branch)
        self.enable_shared = enable_shared
        self.enable_pointwise = enable_pointwise
        if not self.enable_shared and not self.enable_pointwise:
            raise ValueError(
                "At least one of enable_shared or enable_pointwise must be True."
            )
        if self.enable_shared:
            logger.info("Enabling shared multipliers.")
        self.multipliers_shared = nn.Parameter(torch.zeros(self.n_multipliers))
        if self.enable_pointwise:
            logger.info("Enabling pointwise multipliers.")
        self.multipliers_pointwise = nn.Parameter(
            torch.zeros(
                n_train if self.enable_pointwise else 0,
                self.n_multipliers,
            )
        )

    def _init_metadata(self, n_bus, n_gen, n_branch):
        with torch.no_grad():
            # the size of each multiplier category
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

            # mask to separate the inequality multipliers from the equality multipliers
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

    def get_multipliers(self, data: PowerflowData) -> dict[str, torch.Tensor]:
        """
        Get the multiplier for the tensor corresponding to the given index.
        """
        idx = data.index
        multipliers = torch.zeros(
            idx.shape[0], self.n_multipliers, device=self.multipliers_shared.device
        )
        if self.enable_shared:
            multipliers = multipliers + self.multipliers_shared.expand(idx.shape[0], -1)
        if self.enable_pointwise:
            multipliers = multipliers + self.multipliers_pointwise[idx]

        multiplier_dict = {}
        for multipliers_split, (name, shape) in zip(
            torch.tensor_split(multipliers, self.multiplier_offsets, dim=1),
            self.multiplier_metadata.items(),
        ):
            multiplier_dict[name] = multipliers_split.reshape(
                (idx.shape[0] * shape[0],) + shape[1:]
            )
        return multiplier_dict

    def parameters_shared(self) -> list[nn.Parameter]:
        return [self.multipliers_shared]

    def parameters_pointwise(self) -> list[nn.Parameter]:
        return [self.multipliers_pointwise]

    def project_shared(self):
        """
        Project the inequality multipliers to be non-negative. Only the common multipliers are projected.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        self.multipliers_shared.data[self.multiplier_inequality_mask] = (
            self.multipliers_shared.data[self.multiplier_inequality_mask].relu_()
        )

    def project_pointwise(self, index: torch.Tensor | None = None):
        """
        Project the inequality multipliers to be non-negative. Only the pointwise multipliers are projected.

        Args:
            index: The indices of the multipliers to project. If None, all multipliers are projected.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        if index is None:
            self.multipliers_pointwise.data[:, self.multiplier_inequality_mask] = (
                self.multipliers_pointwise.data[
                    :, self.multiplier_inequality_mask
                ].relu_()
            )
        else:
            self.multipliers_pointwise.data[
                index[:, None], self.multiplier_inequality_mask
            ] = self.multipliers_pointwise.data[
                index[:, None], self.multiplier_inequality_mask
            ].relu_()

    def zero_grad_pointwise(self):
        self.multipliers_pointwise.grad = None

    def grad_clip_norm_shared(self, value: float, p: float):
        with torch.no_grad():
            if self.multipliers_shared.grad is None:
                logger.warning(
                    "No gradient found for shared multipliers. Skipping clip."
                )
                return
            norm = torch.norm(self.multipliers_shared.grad, p=p)
            scale = norm.clamp(max=value) / (norm + 1e-12)
            self.multipliers_shared.grad *= scale

    def grad_clip_norm_pointwise(
        self, value: float, p: float, index: torch.Tensor | None = None
    ):
        """
        Clip the gradients of the pointwise multipliers. Will clip each row of the gradient table independently.

        Args:
            value: The maximum norm of the gradients.
            p: The p-norm to use.
            index: The indices of the multipliers to clip. If None, all multipliers are clipped.
        """

        with torch.no_grad():
            if self.multipliers_pointwise.grad is None:
                logger.warning(
                    "No gradient found for pointwise multipliers. Skipping clip."
                )
                return
            if index is None:
                norm = torch.norm(
                    self.multipliers_pointwise.grad, p=p, dim=-1, keepdim=True
                )
                scale = norm.clamp(max=value) / (norm + 1e-12)
                self.multipliers_pointwise.grad *= scale
            else:
                norm = torch.norm(
                    self.multipliers_pointwise.grad[index], p=p, dim=-1, keepdim=True
                )
                scale = norm.clamp(max=value) / (norm + 1e-12)
                self.multipliers_pointwise.grad[index] *= scale

    def step_pointwise(
        self,
        idx: torch.Tensor,
        lr: float,
        weight_decay: float,
        maximize: bool = False,
    ):
        """
        Manual SGD step for the pointwise multipliers corresponding to the given indices.
        1. Apply weight decay.
        2. Update the multipliers.
        3. Project the multipliers.

        Args:
            idx: The indices of the multipliers to update.
            lr: The learning rate.
            weight_decay: The weight decay.
            maximize: Whether to maximize the multipliers.
        """
        with torch.no_grad():
            if self.multipliers_pointwise.grad is None:
                logger.warning(
                    "No gradient found for pointwise multipliers. Skipping SGD step"
                )
                return
            grad = self.multipliers_pointwise.grad[idx]
            if maximize:
                grad = -grad
            # apply weight decay
            if weight_decay > 0:
                grad += weight_decay * self.multipliers_pointwise.data[idx]
            # get a view of the tensor data since we are modifying tensors in-place that require grad
            self.multipliers_pointwise.data[idx] -= lr * grad


class OPFDual(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: OPFModel,
        n_nodes: tuple[int, int, int],
        n_train: int,
        dual_graph: bool,
        lr: float = 1e-4,
        lr_dual_pointwise: float = 1e-3,
        lr_dual_shared: float = 1e-4,
        wd: float = 0.0,
        wd_dual_pointwise: float = 0.0,
        wd_dual_shared: float = 0.0,
        grad_clip_norm: float = 0.0,
        grad_clip_p: float = 2.0,
        grad_clip_norm_dual: float = 10.0,
        grad_clip_p_dual: float = 1.0,
        eps: float = 1e-3,
        enforce_constraints: bool = False,
        detailed_metrics: bool = False,
        cost_weight: float = 1.0,
        augmented_weight: float = 0.0,
        supervised_weight: float = 0.0,
        powerflow_weight: float = 0.0,
        warmup: int = 0,
        supervised_warmup: int = 0,
        multiplier_type: str = "hybrid",
        **kwargs,
    ):
        """
        Args:
            model (torch.nn.Module): The model to be trained.
            n_nodes (tuple[int, int, int]): A tuple containing the number of buses, branches, and generators.
            n_train (int): The number of training samples.
            dual_graph (bool): Whether the input graph will be a regular or dual graph.
            lr (float, optional): The learning rate. Defaults to 1e-4.
            lr_dual_pointwise (float, optional): The learning rate for the pointwise multipliers. Defaults to 1e-3.
            lr_dual_shared (float, optional): The learning rate for the shared multipliers. Defaults to 1e-4.
            wd (float, optional): The weight decay. Defaults to 0.0.
            wd_dual_pointwise (float, optional): The weight decay for the pointwise multipliers. Defaults to 0.0.
            wd_dual_shared (float, optional): The weight decay for the shared multipliers. Defaults to 0.0.
            grad_clip_norm (float, optional): The gradient clipping value for the primal variables. Defaults to 0.1. Set to 0 to disable gradient clipping.
            grad_clip_p (float, optional): The gradient clipping p-norm for the primal variables. Defaults to 2.0.
            grad_clip_norm_dual (float, optional): The gradient clipping value for the dual variables. Defaults to 1.0. Set to 0 to disable gradient clipping.
            grad_clip_p_dual (float, optional): The gradient clipping p-norm for the dual variables. Defaults to 1.0.
            eps (float, optional): Numerical threshold, below which values will be assumed to be approximately zeros. Defaults to 1e-3.
            enforce_constraints (bool, optional): Whether to enforce the constraints. Defaults to False.
            detailed_metrics (bool, optional): Whether to log detailed metrics. Defaults to False.
            cost_weight (float, optional): The weight of the cost. Defaults to 1.0.
            augmented_weight (float, optional): The weight of the augmented loss. Defaults to 0.0.
            supervised_weight (float, optional): The weight of the supervised loss. Defaults to 0.0.
            powerflow_weight (float, optional): The weight of the powerflow loss. Defaults to 0.0.
            warmup (int, optional): Number of epochs before starting to update the multipliers. Defaults to 0.
            supervised_warmup (int, optional): Number of epochs during which supervised loss is used. Defaults to 0.
            multiplier_type (str, optional): Default "shared". Can be "shared", "pointwise", or "hybrid".
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "n_nodes", "n_train", "kwargs"])
        self.model = model
        self.dual_graph = dual_graph
        self.lr = lr
        self.weight_decay = wd
        self.lr_dual_pointwise = lr_dual_pointwise
        self.lr_dual_shared = lr_dual_shared
        self.wd_dual_pointwise = wd_dual_pointwise
        self.wd_dual_shared = wd_dual_shared
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_p = grad_clip_p
        self.grad_clip_norm_dual = grad_clip_norm_dual
        self.grad_clip_p_dual = grad_clip_p_dual
        self.eps = eps
        self._enforce_constraints = enforce_constraints
        self.detailed_metrics = detailed_metrics
        self.automatic_optimization = False
        self.cost_weight = cost_weight
        self.augmented_weight = augmented_weight
        self.supervised_weight = supervised_weight
        self.powerflow_weight = powerflow_weight
        self.warmup = warmup
        self.supervised_warmup = supervised_warmup
        if multiplier_type in ["shared", "pointwise", "hybrid"]:
            n_bus, n_branch, n_gen = n_nodes
            self.model_dual = DualTable(
                n_bus,
                n_gen,
                n_branch,
                n_train,
                enable_shared=multiplier_type in ["shared", "hybrid"],
                enable_pointwise=multiplier_type in ["pointwise", "hybrid"],
            )

    def forward(
        self,
        input: PowerflowData | PowerflowData,
    ) -> tuple[pf.PowerflowVariables, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[pf.PowerflowVariables, torch.Tensor, torch.Tensor]:
              - The powerflow variables,
              - the predicted forward power,
              - and the predicted backward power.
        """
        graph, _ = input
        if not isinstance(graph, HeteroData):
            raise ValueError(
                f"Unsupported data type {type(graph)}, expected HeteroData."
            )
        y_dict = self.model(graph)
        load = graph["bus"].load
        bus = y_dict["bus"][..., :2]
        gen = y_dict["gen"][..., :2]
        branch = y_dict["branch"]

        bus_params = pf.BusParameters.from_tensor(graph["bus"].params)
        gen_params = pf.GenParameters.from_tensor(graph["gen"].params)

        V = self.parse_bus(bus)
        Sg = self.parse_gen(gen)
        Sd = self.parse_load(load)
        Sf_pred, St_pred = self.parse_branch(branch)
        if self._enforce_constraints:
            V, Sg = self.enforce_constraints(V, Sg, bus_params, gen_params)
        return (
            pf.powerflow_from_graph(V, Sd, Sg, graph, self.dual_graph),
            Sf_pred,
            St_pred,
        )

    def enforce_constraints(
        self,
        V,
        Sg,
        bus_params: pf.BusParameters,
        gen_params: pf.GenParameters,
        strategy="sigmoid",
    ):
        """
        Ensure that voltage and power generation are within the specified bounds.

        Args:
            V: The bus voltage. Magnitude must be between params.vm_min and params.vm_max.
            Sg: The generator power. Real and reactive power must be between params.Sg_min and params.Sg_max.
            bus_params: The bus parameters.
            gen_params: The generator parameters.
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
        vm = fn(V.abs(), bus_params.vm_min, bus_params.vm_max)
        V = torch.polar(vm, V.angle())
        Sg = torch.complex(
            fn(Sg.real, gen_params.Sg_min.real, gen_params.Sg_max.real),
            fn(Sg.imag, gen_params.Sg_min.imag, gen_params.Sg_max.imag),
        )
        return V, Sg

    def supervised_loss(
        self,
        batch: PowerflowData,
        variables: pf.PowerflowVariables,
        Sf_pred: torch.Tensor,
        St_pred: torch.Tensor,
    ):
        """
        Calculate the MSE between the predicted and target bus voltage and generator power.
        """
        data, _ = batch
        # parse auxiliary data from the batch
        V_target = torch.view_as_complex(data["bus"]["V"])
        Sg_target = torch.view_as_complex(data["gen"]["Sg"])
        Sf_target = torch.view_as_complex(data["branch"]["Sf"])
        St_target = torch.view_as_complex(data["branch"]["St"])

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
            batch_size=batch.graph.num_graphs,
        )
        self.log(
            "train/supervised_loss",
            loss_supervised,
            batch_size=batch.graph.num_graphs,
        )
        return loss_supervised

    def powerflow_loss(
        self, batch: PowerflowData, variables: pf.PowerflowVariables, Sf_pred, St_pred
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
            batch_size=batch.graph.batch_size,
        )
        return powerflow_loss

    def _step_helper(
        self,
        variables: pf.PowerflowVariables,
        graph: HeteroData,
        multipliers: Dict[str, torch.Tensor] | None = None,
        project_powermodels=False,
    ):
        if project_powermodels:
            variables = self.project_powermodels(variables, graph)
        constraints = self.constraints(variables, graph, multipliers)
        cost = self.cost(variables, graph)
        return variables, constraints, cost

    def training_step(self, data: PowerflowData, batch_idx: int):
        primal_optimizer, dual_shared_optimizer = self.optimizers()  # type: ignore
        variables, Sf_pred, St_pred = self(data)
        _, constraints, cost = self._step_helper(
            variables, data.graph, self.model_dual.get_multipliers(data)
        )
        constraint_loss = self.constraint_loss(constraints)

        supervised_loss = self.supervised_loss(data, variables, Sf_pred, St_pred)
        # linearly decay the supervised loss until 0 at self.current_epoch > self.supervised_warmup
        supervised_weight = self.supervised_weight * (
            max(1.0 - self.current_epoch / self.supervised_warmup, 0.0)
            if self.supervised_warmup > 0
            else 1.0
        )
        powerflow_loss = self.powerflow_loss(data, variables, Sf_pred, St_pred)

        loss = (
            self.cost_weight * cost
            + constraint_loss
            + self.powerflow_weight * powerflow_loss
            + supervised_weight * supervised_loss
        )

        primal_optimizer.zero_grad()
        self.model_dual.zero_grad_pointwise()
        dual_shared_optimizer.zero_grad()
        self.manual_backward(loss)

        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm,
                norm_type=self.grad_clip_p,
            )
        primal_optimizer.step()

        is_warmed_up = self.current_epoch >= self.warmup
        if is_warmed_up and self.model_dual.enable_pointwise:
            if self.grad_clip_norm_dual > 0:
                self.model_dual.grad_clip_norm_pointwise(
                    value=self.grad_clip_norm_dual,
                    p=self.grad_clip_p_dual,
                    index=data.index,
                )
            # updating the pointwise multipliers is done manually
            self.model_dual.step_pointwise(
                idx=data.index,
                lr=self.lr_dual_pointwise,
                weight_decay=self.wd_dual_pointwise,
                maximize=True,
            )
            self.model_dual.project_pointwise(data.index)
        if is_warmed_up and self.model_dual.enable_shared:
            # update the shared multipliers
            if self.grad_clip_norm_dual > 0:
                self.model_dual.grad_clip_norm_shared(
                    value=self.grad_clip_norm_dual,
                    p=self.grad_clip_p_dual,
                )
            dual_shared_optimizer.step()
            self.model_dual.project_shared()

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=data.graph.num_graphs,
            sync_dist=True,
        )
        self.log_dict(
            self.metrics(cost, constraints, "train", self.detailed_metrics, train=True),
            batch_size=data.graph.num_graphs,
            sync_dist=True,
        )

    def validation_step(self, batch: PowerflowData, *args):
        graph = batch.graph
        assert isinstance(graph, HeteroData)

        batch_size = batch.graph.num_graphs
        _, constraints, cost = self._step_helper(
            self(batch)[0], graph, project_powermodels=False
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

    def test_step(self, batch: PowerflowData, *args):
        graph = batch.graph
        assert isinstance(graph, HeteroData)
        # TODO
        # change to make faster
        # project_powermodels taking too long
        # go over batch w/ project pm, then individual steps without
        _, constraints, cost = self._step_helper(
            self(batch)[0], graph, project_powermodels=True
        )
        test_metrics = self.metrics(cost, constraints, "test", self.detailed_metrics)
        self.log_dict(
            test_metrics,
            batch_size=batch.graph.num_graphs,
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
        graph: HeteroData,
        clamp=True,
    ) -> pf.PowerflowVariables:
        V, Sg, Sd = variables.V, variables.Sg, variables.Sd
        bus_params = pf.BusParameters.from_tensor(graph["bus"].params)
        gen_params = pf.GenParameters.from_tensor(graph["gen"].params)

        # TODO: Add support for multiple casefiles, will need to batch them together or loop over them
        casefiles = set(graph.casefile)
        if len(casefiles) != 1:
            raise NotImplementedError(
                f"Currently all graphs in the batch must have the same casefile. Found {casefiles}."
            )

        if clamp:
            V, Sg = self.enforce_constraints(
                V, Sg, bus_params, gen_params, strategy="clamp"
            )

        bus_shape = V.shape
        gen_shape = Sg.shape
        dtype = V.dtype
        device = V.device
        n_bus = graph["bus"].num_nodes // graph.batch_size
        n_gen = graph["gen"].num_nodes // graph.batch_size

        V = torch.view_as_real(V.cpu()).view(-1, n_bus, 2).numpy()
        Sg = torch.view_as_real(Sg.cpu()).view(-1, n_gen, 2).numpy()
        Sd = torch.view_as_real(Sd.cpu()).view(-1, n_bus, 2).numpy()
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
                    graph.casefile[0],
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
        V = torch.complex(V[..., 0], V[..., 1]).to(device, dtype).reshape(bus_shape)
        Sg = torch.complex(Sg[..., 0], Sg[..., 1]).to(device, dtype).reshape(gen_shape)
        Sd = torch.complex(Sd[..., 0], Sd[..., 1]).to(device, dtype).reshape(bus_shape)
        return pf.powerflow_from_graph(V, Sd, Sg, graph, self.dual_graph)

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
        graph: HeteroData,
    ) -> torch.Tensor:
        """Compute the cost to produce the active and reactive power."""
        gen_parameters = pf.GenParameters.from_tensor(graph["gen"].params)

        p = variables.Sg.real
        p_coeff = gen_parameters.cost_coeff
        cost = torch.zeros_like(p)
        for i in range(p_coeff.shape[-2]):
            cost += p_coeff.select(-2, i) * p.squeeze() ** i
        # generator cost cannot be negative
        cost = cost.relu()
        # compute the total cost for each sample in the batch
        cost_per_batch = torch.zeros_like(graph.reference_cost).index_add(
            0, graph["gen"].batch, cost
        )
        # normalize the cost by the reference cost (IPOPT cost)
        cost_per_batch = cost_per_batch / graph.reference_cost
        return cost_per_batch.mean()

    def constraints(
        self,
        variables: pf.PowerflowVariables,
        graph: HeteroData,
        multipliers: Dict[str, torch.Tensor] | None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Nested map from constraint name => (value name => tensor value)
        """
        constraints = pf.build_constraints(variables, graph, self.dual_graph)
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
            "error_max": torch.mean,
            "rate": torch.mean,
            "multiplier/mean": torch.mean,
            "multiplier/max": torch.mean,
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
        primal_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        shared_params = self.model_dual.parameters_shared()
        if len(shared_params) > 0:
            logger.info("Creating optimizer for dual shared parameters.")
            dual_shared_optimizer = torch.optim.Adamax(
                shared_params,
                lr=self.lr_dual_shared,
                weight_decay=self.wd_dual_shared,
                maximize=True,
            )
            # Add linear scheduler for the Adamax optimizer
            dual_shared_scheduler = torch.optim.lr_scheduler.LinearLR(
                dual_shared_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=100,
            )
        else:
            logger.info("Using NullOptimizer for dual shared parameters.")
            dual_shared_optimizer = NullOptimizer()
            dual_shared_scheduler = None

        return [primal_optimizer, dual_shared_optimizer], [dual_shared_scheduler]

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
