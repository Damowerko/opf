import copy
import json
import os
import typing
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import h5py
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data, HeteroData, download_url, extract_tar
from torch_geometric.typing import EdgeType, NodeType

import opf.powerflow as pf

PGLIB_VERSION = "21.07"


class PowerflowData(typing.NamedTuple):
    data: Data | HeteroData
    index: torch.Tensor


class PowerflowBatch(typing.NamedTuple):
    data: Batch
    index: torch.Tensor


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


def pad_node_features(graph: HeteroData):
    """
    Pad the node features to the same length.
    """
    max_dim = max(graph.num_node_features.values())
    for nt, dim in graph.num_node_features.items():
        if dim < max_dim:
            graph[nt].x = torch.cat(
                (graph[nt].x, torch.zeros(graph[nt].x.shape[0], max_dim - dim)),
                dim=-1,
            )
    return graph


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
    bus_params = pf.BusParameters.from_pf_parameters(params)
    gen_params = pf.GenParameters.from_pf_parameters(params)
    branch_params = pf.BranchParameters.from_pf_parameters(params)

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


def build_dual_graph(params: pf.PowerflowParameters) -> HeteroData:
    graph = HeteroData()

    graph["bus"].num_nodes = params.n_bus
    graph["branch"].num_nodes = params.n_branch
    graph["gen"].num_nodes = params.n_gen

    branch_index = torch.arange(params.n_branch)
    gen_index = torch.arange(params.n_gen)

    # Define Anti-Symmetric Graph

    # BFR
    graph["bus", "from", "branch"].edge_index = torch.stack(
        [params.fr_bus, branch_index], dim=0
    )
    graph["bus", "from", "branch"].edge_weight = torch.ones(params.n_branch)
    # RFB
    graph["branch", "from", "bus"].edge_index = torch.stack(
        [branch_index, params.fr_bus], dim=0
    )
    graph["branch", "from", "bus"].edge_weight = torch.ones(params.n_branch)

    # BTR
    graph["bus", "to", "branch"].edge_index = torch.stack(
        [params.to_bus, branch_index], dim=0
    )
    graph["bus", "to", "branch"].edge_weight = torch.ones(params.n_branch)
    # RTB
    graph["branch", "to", "bus"].edge_index = torch.stack(
        [branch_index, params.to_bus], dim=0
    )
    graph["branch", "to", "bus"].edge_weight = torch.ones(params.n_branch)

    # B&G
    graph["bus", "tie", "gen"].edge_index = torch.stack(
        [params.gen_bus_ids, gen_index], dim=0
    )
    graph["bus", "tie", "gen"].edge_weight = torch.ones(params.n_gen)
    # G&B
    graph["gen", "tie", "bus"].edge_index = torch.stack(
        [gen_index, params.gen_bus_ids], dim=0
    )
    graph["gen", "tie", "bus"].edge_weight = torch.ones(params.n_gen)

    # add all the powerflow parameters as node features
    graph["bus"].params = pf.BusParameters.from_pf_parameters(params).to_tensor()
    graph["branch"].params = pf.BranchParameters.from_pf_parameters(params).to_tensor()
    graph["gen"].params = pf.GenParameters.from_pf_parameters(params).to_tensor()

    return graph


def static_collate(input: list[PowerflowData]) -> PowerflowBatch:
    """
    Collate function for static graphs.
    """

    data_list, indices = zip(*input)
    # cast to appropriate types
    data_list = typing.cast(tuple[Data | HeteroData], data_list)
    batch: Batch = Batch.from_data_list(data_list)  # type: ignore
    return PowerflowBatch(
        batch,
        torch.cat(indices),
    )


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
        self.powerflow_parameters = copy.deepcopy(powerflow_parameters)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> PowerflowData:
        return PowerflowData(
            Data(
                x=self.x[index],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
            ),
            torch.tensor(index, dtype=torch.long),
        )

    @staticmethod
    def collate_fn(input: list[PowerflowData]) -> PowerflowBatch:
        return static_collate(input)


class StaticHeteroDataset(Dataset[PowerflowData]):
    def __init__(
        self,
        load: torch.Tensor,
        graph: HeteroData,
        case_name: str,
        additional_features: dict[tuple[NodeType, str], torch.Tensor] = {},
    ):
        """
        A datapipe that wraps a torch_geometric.data.Data object with a single graphs but
        multiple samples on the graph.

        Args:
            load: Load samples with shape (n_samples, n_bus, n_features_bus).
            graph: HeteroData object with the graph structure.
            additional_features: Additional features for each node type. The key is a tuple of the node type and the feature name.
                The tensors should have shape (n_samples, n_{node_type}, n_dim_{feature_name})
        """
        super().__init__()
        self.load = load
        self.additional_features = additional_features
        self.graph = graph
        self.graph.case_name = case_name

    def __len__(self) -> int:
        return len(self.load)

    def __getitem__(self, index) -> PowerflowData:
        # shallow copy, to avoid copying tensors
        data = copy.copy(self.graph)
        data["bus"].load = self.load[index]
        data["bus"].x = torch.cat([data["bus"].load, data["bus"].params], dim=-1)
        data["branch"].x = data["branch"].params
        data["gen"].x = data["gen"].params
        for (node_type, feature_name), feature in self.additional_features.items():
            data[node_type][feature_name] = feature[index]

        return PowerflowData(
            data,
            torch.tensor([index], dtype=torch.long),
        )

    @staticmethod
    def collate_fn(input: list[PowerflowData]) -> PowerflowBatch:
        return static_collate(input)


class CaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        case_name="case1354_pegase__api",
        data_dir="./data",
        batch_size=32,
        num_workers=min(cpu_count(), 8),
        pin_memory=False,
        homo=False,
        test_samples=1000,
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
        self.test_samples = test_samples

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

    @property
    def feature_dims(self) -> dict[NodeType, int]:
        if self.homo:
            raise NotImplementedError(
                "input_features is not implemented for homo graphs."
            )
        if self.graph is None:
            raise ValueError("Graph is not initialized. Call `setup()` first.")
        dataset = self.train_dataset or self.val_dataset or self.test_dataset
        assert isinstance(dataset, StaticHeteroDataset)
        return {k: v.shape[-1] for k, v in dataset[0].data.x_dict.items()}

    def build_dataset(self, load, additional_features):
        assert self.powerflow_parameters is not None
        if self.homo:
            assert isinstance(self.graph, Data)
            return StaticGraphDataset(
                load,
                self.graph,
                self.powerflow_parameters,
            )
        else:
            assert isinstance(self.graph, HeteroData)

            return StaticHeteroDataset(
                load,
                self.graph,
                self.powerflow_parameters.casefile,
                additional_features,
            )

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
                self.graph = build_dual_graph(self.powerflow_parameters)

        with h5py.File(self.data_dir / f"{self.case_name}.h5", "r") as f:
            # load bus voltage in rectangular coordinates
            bus_voltage = torch.from_numpy(f["bus"][:]).float()  # type: ignore
            bus_voltage = torch.view_as_real(
                torch.polar(bus_voltage[..., 0], bus_voltage[..., 1])
            )
            # gen power is already in rectangular coordinates
            gen_power = torch.from_numpy(f["gen"][:]).float()  # type: ignore
            # branch flow Sf and St
            branch = torch.from_numpy(f["branch"][:]).float()  # type: ignore
            Sf = branch[..., :2]
            St = branch[..., 2:]
            additional_features = {
                ("bus", "V"): bus_voltage,
                ("gen", "Sg"): gen_power,
                ("branch", "Sf"): Sf,
                ("branch", "St"): St,
            }
            load = torch.from_numpy(f["load"][:]).float()  # type: ignore

        n_samples = load.shape[0]
        n_bus = self.powerflow_parameters.n_bus

        bus_load = torch.zeros((n_samples, n_bus, 2))
        bus_load[:, self.powerflow_parameters.load_bus_ids, :] = load

        # figure out the number of samples for each set
        n_train = n_samples - 2 * self.test_samples
        n_val = self.test_samples
        n_test = self.test_samples

        if stage == "fit" or stage is None:
            self.train_dataset = self.build_dataset(
                bus_load[:n_train],
                {k: v[:n_train] for k, v in additional_features.items()},
            )
            self.val_dataset = self.build_dataset(
                bus_load[n_train : n_train + n_val],
                {
                    k: v[n_train : n_train + n_val]
                    for k, v in additional_features.items()
                },
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.build_dataset(
                bus_load[-n_test:],
                {k: v[-n_test:] for k, v in additional_features.items()},
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

    def transfer_batch_to_device(
        self,
        input: PowerflowBatch | PowerflowData,
        device,
        dataloader_idx: int,
    ) -> PowerflowBatch | PowerflowData:
        cls = PowerflowBatch if isinstance(input, PowerflowBatch) else PowerflowData
        return cls(
            input.data.to(device),  # type: ignore
            input.index.to(device),
        )
