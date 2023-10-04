import copy
import json
import os
import typing
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data, HeteroData, download_url, extract_tar
from torch_geometric.typing import EdgeType, NodeType

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
    antisymmetric: bool = True,
) -> HeteroData:
    graph = HeteroData()
    graph["bus"].num_nodes = params.n_bus
    graph["branch"].num_nodes = params.n_branch
    branch_index = torch.arange(params.n_branch)
    reverse_coef = -1 if antisymmetric else 1

    # Define Anti-Symmetric Graph

    # From side of branches
    graph["bus", "from", "branch"].edge_index = torch.stack(
        [params.fr_bus, branch_index], dim=0
    )
    graph["bus", "from", "branch"].edge_weight = torch.ones(params.n_branch)
    graph["branch", "from", "bus"].edge_index = torch.stack(
        [branch_index, params.fr_bus], dim=0
    )
    graph["branch", "from", "bus"].edge_weight = reverse_coef * torch.ones(
        params.n_branch
    )

    # To side of branches
    graph["bus", "to", "branch"].edge_index = torch.stack(
        [params.to_bus, branch_index], dim=0
    )
    graph["bus", "to", "branch"].edge_weight = torch.ones(params.n_branch)
    graph["branch", "to", "bus"].edge_index = torch.stack(
        [branch_index, params.to_bus], dim=0
    )
    graph["branch", "to", "bus"].edge_weight = reverse_coef * torch.ones(
        params.n_branch
    )

    # Self-loops
    if self_loops:
        graph["bus", "self", "bus"].edge_index = torch.stack(
            [torch.arange(params.n_bus), torch.arange(params.n_bus)], dim=0
        )
        graph["bus", "self", "bus"].edge_weight = torch.ones(params.n_bus)
        graph["branch", "self", "branch"].edge_index = torch.stack(
            [branch_index, branch_index], dim=0
        )
        graph["branch", "self", "branch"].edge_weight = torch.ones(params.n_branch)

    return graph


def _concat_features(n_samples: int, *features: torch.Tensor):
    """
    Concatenate the features along the last dimension.
    Args:
        features: Node features with shape (n_samples, num_nodes, num_features) or (num_nodes, num_features).
    """
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
        graph = T.GCNNorm(False)(graph.to_homogeneous()).to_heterogeneous()
        self.graph = T.ToSparseTensor()(graph)
        self.powerflow_parameters = powerflow_parameters

    def __len__(self) -> int:
        return len(self.x_bus)

    def __getitem__(self, index) -> PowerflowData:
        # shallow copy, to avoid copying tensors
        data = copy.copy(self.graph)
        data["bus"].x = self.x_bus[index]
        data["branch"].x = self.x_branch[index]
        return PowerflowData(
            data,
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
        num_workers=min(cpu_count(), 8),
        pin_memory=False,
        homo=False,
        **kwargs,
    ):
        super().__init__()
        self.case_name = case_name
        self.data_dir = Path(data_dir)
        self.pglib_path = Path(self.data_dir, f"pglib-opf-{PGLIB_VERSION}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.homo = homo

        self.powerflow_parameters = None
        self.graph = None
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

    def metadata(self) -> tuple[list[NodeType], list[EdgeType]]:
        if self.homo:
            raise ValueError("Metadata is only available for hetero graphs.")
        if self.graph is None:
            raise ValueError("Graph is not initialized. Call `setup()` first.")
        dataset = self.train_dataset or self.val_dataset or self.test_dataset
        assert isinstance(dataset, StaticHeteroDataset)
        return dataset[0][0].metadata()

    def setup(self, stage: Optional[str] = None):
        if self.powerflow_parameters is None:
            with open(self.case_path) as f:
                powermodels_dict = json.load(f)

            # parse the powermodels dict
            self.powerflow_parameters = pf.parameters_from_powermodels(
                powermodels_dict, self.case_path.as_posix()
            )
        if self.graph is None:
            if self.homo:
                self.graph = build_graph(self.powerflow_parameters)
            else:
                self.graph = build_hetero_graph(self.powerflow_parameters, True, False)

        bus_parameters = self.powerflow_parameters.bus_parameters()
        branch_parameters = self.powerflow_parameters.branch_parameters()

        def parse_dataset(dicts: list[dict]):
            assert self.powerflow_parameters is not None

            # convert to torch tensors
            load = torch.stack(
                [pf.powermodels_to_tensor(d["load"], ["pd", "qd"]) for d in dicts]
            )
            n_samples = load.shape[0]

            bus_load = (load.mT @ self.powerflow_parameters.load_matrix).mT
            bus_features = _concat_features(
                n_samples,
                bus_load,
                bus_parameters,
            ).to(bus_load.dtype)

            if self.homo:
                assert isinstance(self.graph, Data)
                return StaticGraphDataset(
                    bus_features,
                    self.graph,
                    self.powerflow_parameters,
                )
            else:
                assert isinstance(self.graph, HeteroData)
                branch_features = _concat_features(
                    n_samples,
                    branch_parameters,
                ).to(bus_load.dtype)
                return StaticHeteroDataset(
                    bus_features,
                    branch_features,
                    self.graph,
                    self.powerflow_parameters,
                )

        # Labeled data is stored as a json file, which describes a list of dictionaries
        # Each dict contains {"load" => another dict describing all the loads, "solution" => a solution to the powermodels problem}
        if stage in (None, "fit"):
            with self.case_path.with_suffix(".train.json").open() as f:
                train_dicts: list[dict] = json.load(f)
                self.train_dataset = parse_dataset(train_dicts)

            with self.case_path.with_suffix(".valid.json").open() as f:
                valid_dicts = json.load(f)
                self.val_dataset = parse_dataset(valid_dicts)
                # average objective value for the validation set
                self.powerflow_parameters.reference_cost = sum(
                    [d["result"]["objective"] / len(valid_dicts) for d in valid_dicts]
                )

        if stage in (None, "test"):
            with self.case_path.with_suffix(".test.json").open() as f:
                test_dicts: list[dict] = json.load(f)
                self.test_dataset = parse_dataset(test_dicts)
                # average objective value for the test set
                self.powerflow_parameters.reference_cost = sum(
                    [d["result"]["objective"] / len(test_dicts) for d in test_dicts]
                )

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
