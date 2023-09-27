import copy
import json
import os
import typing
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data, HeteroData, download_url, extract_tar

import opf.powerflow as pf

PGLIB_VERSION = "21.07"


class PowerflowData(typing.NamedTuple):
    data: Data | HeteroData
    powerflow_parameters: pf.PowerflowParameters


class PowerflowBatch(typing.NamedTuple):
    data: Batch
    powerflow_parameters: pf.PowerflowParameters


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


def build_graph(
    params: pf.PowerflowParameters,
) -> Data:
    """Convert a pf.PowerflowParameters object to a torch_geometric.data.Data object.

    Args:
        powerflow_parameters: Powerflow parameters.

    Returns:
        edge_index: Edge index of the graph.
        edge_attr: Edge attributes of the graph:
            - Real part of admittance
            - Imaginary part of admittance
            - Power limit
            - Any other branch constraints in `params.constraints`.
    """
    # Use the branch admittance matrix to construct the graph
    # see https://matpower.org/docs/MATPOWER-manual.pdf
    edge_index_forward = torch.stack([params.fr_bus, params.to_bus], dim=0)
    edge_index_backward = torch.stack([params.to_bus, params.fr_bus], dim=0)
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)

    # add +-1 to the edge attributes to indicate weather the edge is forward or backward
    edge_attr_forward = torch.cat(
        [params.branch_parameters(), torch.ones(params.n_branch, 1)], dim=1
    )
    edge_attr_backward = torch.cat(
        [params.branch_parameters(), -torch.ones(params.n_branch, 1)], dim=1
    )
    edge_attr = torch.cat(
        [
            edge_attr_forward,
            edge_attr_backward,
        ],
        dim=0,
    )
    return Data(edge_index=edge_index, edge_attr=edge_attr.float())


def build_hetero_graph(
    params: pf.PowerflowParameters,
    self_loops: bool = True,
) -> HeteroData:
    n_bus = params.n_bus
    branch_nodes = torch.arange(params.n_branch) + n_bus

    graph = HeteroData()

    # Define Anti-Symmetric Graph

    # From side of branches
    graph["bus", "from", "branch"].edge_index = torch.stack(
        [params.fr_bus, branch_nodes], dim=0
    )
    graph["branch", "from", "bus"].edge_index = torch.stack(
        [branch_nodes, params.fr_bus], dim=0
    )
    graph["bus", "from", "branch"].edge_weight = torch.ones(params.n_branch)
    graph["branch", "from", "bus"].edge_weight = -torch.ones(params.n_branch)

    # To side of branches
    graph["bus", "to", "branch"].edge_index = torch.stack(
        [params.to_bus, branch_nodes], dim=0
    )
    graph["branch", "to", "bus"].edge_index = torch.stack(
        [branch_nodes, params.to_bus], dim=0
    )
    graph["bus", "to", "branch"].edge_weight = torch.ones(params.n_branch)
    graph["branch", "to", "bus"].edge_weight = -torch.ones(params.n_branch)

    # Self-loops
    if self_loops:
        graph["bus", "self", "bus"].edge_index = torch.stack(
            [torch.arange(n_bus), torch.arange(n_bus)], dim=0
        )
        graph["branch", "self", "branch"].edge_index = torch.stack(
            [branch_nodes, branch_nodes], dim=0
        )
        graph["bus", "self", "bus"].edge_weight = torch.ones(n_bus)
        graph["branch", "self", "branch"].edge_weight = torch.ones(params.n_branch)

    return graph


def _concat_features(*features: torch.Tensor):
    """
    Concatenate the features along the last dimension.
    Args:
        features: Node features with shape (n_samples, num_nodes, num_features) or (num_nodes, num_features).
    """
    # Get the number of samples
    n_samples = max(x.shape[0] for x in features if len(x.shape) == 3)
    # If node features do not have a `n_samples` dimension add it
    # Useing expand creates a view instead of a copy of the tensor
    expanded = [
        x[None, ...].expand(n_samples, -1, -1) if len(x.shape) == 2 else x
        for x in features
    ]
    concatenated = torch.cat(expanded, dim=2)
    return concatenated


@staticmethod
def graph_collate_fn(input: list[PowerflowData]) -> PowerflowBatch:
    data_list, powerflow_parameters_list = zip(*input)
    # cast to appropriate types
    data_list = typing.cast(tuple[Data | HeteroData], data_list)
    powerflow_parameters_list = typing.cast(
        tuple[pf.PowerflowParameters], powerflow_parameters_list
    )
    batch: Batch = Batch.from_data_list(data_list)  # type: ignore
    # let's assume that all samples have the same powerflow parameters
    powerflow_parameters = powerflow_parameters_list[0]
    assert all(powerflow_parameters == p for p in powerflow_parameters_list)
    return PowerflowBatch(batch, powerflow_parameters)


class StaticGraphDataset(Dataset[PowerflowData]):
    def __init__(
        self,
        x: torch.Tensor,
        graph: Data,
        powerflow_parameters: pf.PowerflowParameters,
    ):
        """
        A datapipe that wraps a torch_geometric.data.Data object with a single graphs but
        multiple samples on the graph.

        Args:
            features: Node features with shape (n_samples, num_nodes, num_features).
            edge_index: Edge index with shape (2, num_edges)
            edge_attr: Edge attributes with shape (num_edges, num_edge_features)
        """
        super().__init__()
        self.x = x
        self.edge_index = graph.edge_index
        self.edge_attr = graph.edge_attr
        self.powerflow_parameters = powerflow_parameters

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> PowerflowData:
        return PowerflowData(
            Data(
                x=self.x[index],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
            ),
            self.powerflow_parameters,
        )

    @staticmethod
    def collate_fn(input: list[PowerflowData]) -> PowerflowBatch:
        return graph_collate_fn(input)


class StaticHeteroDataset(Dataset[PowerflowData]):
    def __init__(
        self,
        bus_features: torch.Tensor,
        branch_features: torch.Tensor,
        graph: HeteroData,
        powerflow_parameters: pf.PowerflowParameters,
    ):
        """
        A datapipe that wraps a torch_geometric.data.Data object with a single graphs but
        multiple samples on the graph.

        Args:
            x_bus: Bus features with shape (n_samples, n_bus, n_features).
            x_branch: Branch features with shape (n_samples, n_branch, n_features).
            edge_index: Edge index with shape (2, num_edges)
            edge_attr: Edge attributes with shape (num_edges, num_edge_features)
        """
        super().__init__()
        if bus_features.shape[0] != branch_features.shape[0]:
            raise ValueError(
                f"Expected node_features and branch_features to have the same number of samples, but got {bus_features.shape[0]} and {branch_features.shape[0]}."
            )
        self.x_bus = bus_features
        self.x_branch = branch_features
        self.graph = T.ToSparseTensor()(graph)
        self.powerflow_parameters = powerflow_parameters

    def __len__(self) -> int:
        return len(self.x_bus)

    def __getitem__(self, index) -> PowerflowData:
        # shallow copy, to avoid copying tensors
        graph = copy.copy(self.graph)
        graph["bus"].x = self.x_bus[index]
        graph["branch"].x = self.x_branch[index]
        return PowerflowData(
            graph,
            self.powerflow_parameters,
        )

    @staticmethod
    def collate_fn(input: list[PowerflowData]) -> PowerflowBatch:
        return graph_collate_fn(input)


class CaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        case_name="case1354_pegase__api",
        data_dir="./data",
        batch_size=32,
        ratio_train=0.95,
        load_distribution_width=0.2,
        num_workers=min(cpu_count(), 8),
        pin_memory=False,
        bipartite=False,
        **kwargs,
    ):
        super().__init__()
        self.case_name = case_name
        self.data_dir = Path(data_dir)
        self.pglib_path = Path(self.data_dir, f"pglib-opf-{PGLIB_VERSION}")
        self.batch_size = batch_size
        self.ratio_trian = ratio_train
        self.load_distribution_width = load_distribution_width
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.hetero = bipartite

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not self.pglib_path.exists():
            with TemporaryDirectory() as tmp_dir:
                download_url(
                    f"https://github.com/power-grid-lib/pglib-opf/archive/refs/tags/v{PGLIB_VERSION}.tar.gz",
                    tmp_dir,
                )
                extract_tar(
                    os.path.join(tmp_dir, f"v{PGLIB_VERSION}.tar.gz"),
                    self.data_dir.as_posix(),
                )

    @property
    def case_path(self):
        return Path(self.data_dir / f"{self.case_name}.json")

    def setup(self, stage: Optional[str] = None):
        # load from json
        with open(self.case_path) as f:
            powermodels_dict = json.load(f)

        # parse the powermodels dict
        powerflow_parameters = pf.parameters_from_powermodels(
            powermodels_dict, self.case_path.as_posix()
        )
        bus_parameters = powerflow_parameters.bus_parameters()
        if self.hetero:
            graph = build_hetero_graph(powerflow_parameters)
        else:
            graph = build_graph(powerflow_parameters)

        def dataset_from_load(load: torch.Tensor):
            bus_load = (load.mT @ powerflow_parameters.load_matrix).mT
            bus_features = _concat_features(
                bus_load,
                bus_parameters,
            ).to(bus_load.dtype)
            if self.hetero:
                assert isinstance(graph, HeteroData)
                branch_features = _concat_features(
                    powerflow_parameters.branch_parameters(),
                ).to(bus_load.dtype)
                return StaticHeteroDataset(
                    bus_features,
                    branch_features,
                    graph,
                    powerflow_parameters,
                )
            else:
                assert isinstance(graph, Data)
                return StaticGraphDataset(
                    bus_features,
                    graph,
                    powerflow_parameters,
                )

        if stage in (None, "fit"):
            # The file contains a list of dicts, each dict contains the load, solution to a case
            # they are guaranteed to be feasible this way
            with open(self.data_dir / f"{self.case_name}.train.json") as f:
                train_dicts: list[dict] = json.load(f)
                n_samples = len(train_dicts)
                n_train = int(self.ratio_trian * n_samples)
                n_val = n_samples - n_train

                # average objective value for the validation set
                powerflow_parameters.reference_cost = sum(
                    [d["result"]["objective"] / n_val for d in train_dicts[n_train:]]
                )

                # convert training data to torch tensors
                train_load = torch.stack(
                    [
                        pf.powermodels_to_tensor(d["load"], ["pd", "qd"])
                        for d in train_dicts[:n_train]
                    ]
                )
                self.train_dataset = dataset_from_load(train_load)

                # convert validation data to torch tensors
                validation_load = torch.stack(
                    [
                        pf.powermodels_to_tensor(d["load"], ["pd", "qd"])
                        for d in train_dicts[n_train:]
                    ]
                )
                self.val_dataset = dataset_from_load(validation_load)

        if stage in (None, "test"):
            with open(self.data_dir / f"{self.case_name}.test.json") as f:
                test_dicts: list[dict] = json.load(f)
                n_test = len(test_dicts)

                # average objective value for the test set
                powerflow_parameters.reference_cost = sum(
                    [d["result"]["objective"] / n_test for d in test_dicts]
                )

                test_load = torch.stack(
                    [
                        pf.powermodels_to_tensor(d["load"], ["pd", "qd"])
                        for d in test_dicts
                    ]
                )
                self.test_dataset = dataset_from_load(test_load)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError(
                "Expected the traininig datapipe to be initialized. Ensure `setup()` was called."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError(
                "Expected the validation datapipe to be initialized. Ensure `setup()` was called."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError(
                "Expected the test datapipe to be initialized. Ensure `setup()` was called."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.test_dataset.collate_fn,
        )
