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
from torch_geometric.data import Batch, HeteroData, download_url, extract_tar
from torch_geometric.typing import EdgeType, NodeType

import opf.powerflow as pf

PGLIB_VERSION = "21.07"


class PowerflowData(typing.NamedTuple):
    graph: HeteroData
    index: torch.Tensor


def build_graph(
    params: pf.PowerflowParameters,
) -> HeteroData:
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
    graph = HeteroData()
    bus_params = pf.BusParameters.from_pf_parameters(params)
    gen_params = pf.GenParameters.from_pf_parameters(params)
    branch_params = pf.BranchParameters.from_pf_parameters(params)
    graph["bus"].num_nodes = params.n_bus
    graph["gen"].num_nodes = params.n_gen
    graph["bus"].params = bus_params.to_tensor()
    graph["gen"].params = gen_params.to_tensor()
    gen_index = torch.arange(params.n_gen)
    # Edges betwen buses
    graph["bus", "branch", "bus"].edge_index = torch.stack(
        [params.fr_bus, params.to_bus], dim=0
    )
    graph["bus", "branch", "bus"].params = branch_params.to_tensor()
    # Edges between buses and generators
    graph["gen", "tie", "bus"].edge_index = torch.stack(
        [gen_index, params.gen_bus_ids], dim=0
    )
    graph.reference_cost = params.reference_cost
    return graph


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
    graph.reference_cost = params.reference_cost
    return graph


def static_collate(input: list[PowerflowData]) -> PowerflowData:
    """
    Collate function for static graphs.
    """

    data_list, indices = zip(*input)
    # cast to appropriate types
    data_list = typing.cast(tuple[HeteroData], data_list)
    batch: HeteroData = Batch.from_data_list(data_list)  # type: ignore
    return PowerflowData(
        batch,
        torch.cat(indices),
    )


class OPFDataset(Dataset[PowerflowData]):
    def __init__(
        self,
        load: torch.Tensor,
        V: torch.Tensor,
        Sg: torch.Tensor,
        Sf: torch.Tensor,
        St: torch.Tensor,
        graph: HeteroData,
        casefile: str,
        dual_graph: bool = False,
    ):
        """
        Wraps a torch_geometric.data.Data object with a single graph
        but multiple samples on the graph.

        Args:
            load: Load samples with shape (n_samples, n_bus, n_features_bus).
            V: Bus voltage samples with shape (n_samples, n_bus, 2).
            Sg: Generator power samples with shape (n_samples, n_gen, 2).
            Sf: Branch power flow samples with shape (n_samples, n_branch, 2).
            St: Branch power flow samples with shape (n_samples, n_branch, 2).
            graph: HeteroData object with the graph structure.
            casefile: Path to the casefile used to generate the data.
            dual_graph: If true, the provided graph is a dual graph (branches are nodes rather than edges).
        """
        super().__init__()
        self.load = load
        self.Sg = Sg
        self.V = V
        self.Sf = Sf
        self.St = St
        self.graph = graph
        self.graph.casefile = casefile
        self.dual_graph = dual_graph

    def __len__(self) -> int:
        return len(self.load)

    def __getitem__(self, index) -> PowerflowData:
        # shallow copy, to avoid copying tensors
        data = copy.copy(self.graph)
        data["bus"].load = self.load[index]
        data["bus"].V = self.V[index]
        data["gen"].Sg = self.Sg[index]
        if self.dual_graph:
            data["branch"].Sf = self.Sf[index]
            data["branch"].St = self.St[index]
        else:
            data["bus", "branch", "bus"].Sf = self.Sf[index]
            data["bus", "branch", "bus"].St = self.St[index]

        return PowerflowData(
            data,
            torch.tensor([index], dtype=torch.long),
        )

    @staticmethod
    def collate_fn(input: list[PowerflowData]) -> PowerflowData:
        return static_collate(input)


class CaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dual_graph: bool,
        case_name="case300_ieee",
        data_dir="./data",
        batch_size=32,
        num_workers=min(cpu_count(), 8),
        pin_memory=False,
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
        self.dual_graph = dual_graph
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
        if self.graph is None:
            raise ValueError("Graph is not initialized. Call `setup()` first.")
        dataset = self.train_dataset or self.val_dataset or self.test_dataset
        assert isinstance(dataset, OPFDataset)
        return dataset[0][0].metadata()

    @property
    def feature_dims(self) -> dict[NodeType, int]:
        if self.graph is None:
            raise ValueError("Graph is not initialized. Call `setup()` first.")
        dataset = self.train_dataset or self.val_dataset or self.test_dataset
        assert isinstance(dataset, OPFDataset)
        return {k: v.shape[-1] for k, v in dataset[0].graph.x_dict.items()}

    def build_dataset(self, load, V, Sg, Sf, St):
        if self.graph is None:
            raise ValueError("Graph is not initialized. Call `setup()` first.")
        if self.powerflow_parameters is None:
            raise ValueError(
                "Powerflow parameters are not initialized. Call `setup()` first."
            )
        return

    def setup(self, stage: Optional[str] = None):
        if self.powerflow_parameters is None:
            with open(self.case_path) as f:
                powermodels_dict = json.load(f)

            # parse the powermodels dict
            self.powerflow_parameters = pf.parameters_from_powermodels(
                powermodels_dict, self.case_path.as_posix()
            )
        with h5py.File(self.data_dir / f"{self.case_name}.h5", "r") as f:
            # load bus voltage in rectangular coordinates
            V = torch.from_numpy(f["bus"][:]).float()  # type: ignore
            V = torch.view_as_real(torch.polar(V[..., 0], V[..., 1]))
            load = torch.from_numpy(f["load"][:]).float()  # type: ignore
            # gen power is already in rectangular coordinates
            Sg = torch.from_numpy(f["gen"][:]).float()  # type: ignore
            branch = torch.from_numpy(f["branch"][:]).float()  # type: ignore
            Sf = branch[..., :2]
            St = branch[..., 2:]
            self.powerflow_parameters.reference_cost = torch.from_numpy(f["objective"][:]).mean().float()  # type: ignore

        if self.graph is None:
            if not self.dual_graph:
                self.graph = build_graph(self.powerflow_parameters)
            else:
                self.graph = build_dual_graph(self.powerflow_parameters)

        n_samples = load.shape[0]
        n_bus = self.powerflow_parameters.n_bus

        Sd = torch.zeros((n_samples, n_bus, 2))
        Sd[:, self.powerflow_parameters.load_bus_ids, :] = load

        # figure out the number of samples for each set
        n_train = n_samples - 2 * self.test_samples
        n_val = self.test_samples
        n_test = self.test_samples
        # Create a tuple of variables, each row defining a sample in the dataset
        # Will programatically split them into train, val and test
        variables = (Sd, V, Sg, Sf, St)
        if stage == "fit" or stage is None:
            self.train_dataset = OPFDataset(
                *(x[:n_train] for x in variables),
                graph=self.graph,
                casefile=self.powerflow_parameters.casefile,
                dual_graph=self.dual_graph,
            )
            self.val_dataset = OPFDataset(
                *(x[n_train : n_train + n_val] for x in variables),
                graph=self.graph,
                casefile=self.powerflow_parameters.casefile,
                dual_graph=self.dual_graph,
            )
        if stage == "test" or stage is None:
            self.test_dataset = OPFDataset(
                *(x[n_train + n_val : n_train + n_val + n_test] for x in variables),
                graph=self.graph,
                casefile=self.powerflow_parameters.casefile,
                dual_graph=self.dual_graph,
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
        input: PowerflowData | PowerflowData,
        device,
        dataloader_idx: int,
    ) -> PowerflowData | PowerflowData:
        cls = PowerflowData if isinstance(input, PowerflowData) else PowerflowData
        return cls(
            input.graph.to(device),  # type: ignore
            input.index.to(device),
        )
