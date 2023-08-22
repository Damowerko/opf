import json
import os
import typing
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data, download_url, extract_tar

import opf.powerflow as pf

PGLIB_VERSION = "21.07"


class PowerflowData(typing.NamedTuple):
    data: Data
    powerflow_parameters: pf.PowerflowParameters


class PowerflowBatch(typing.NamedTuple):
    data: Batch
    powerflow_parameters: pf.PowerflowParameters


def _graph_from_parameters(
    params: pf.PowerflowParameters,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    edge_attr = torch.cat(
        [params.forward_branch_parameters(), params.backward_branch_parameters()], dim=0
    )
    return edge_index, edge_attr.float()


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


class StaticGraphDataset(Dataset[PowerflowData]):
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
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
        self.edge_index = edge_index
        self.edge_attr = edge_attr
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
            # Output a list with one element, so that batches
            # have the same type as a single sample
            self.powerflow_parameters,
        )

    @staticmethod
    def collate_fn(input: list[PowerflowData]) -> PowerflowBatch:
        data_list, powerflow_parameters_list = zip(*input)
        batch: Batch = Batch.from_data_list(data_list)  # type: ignore
        powerflow_parameters_list = typing.cast(
            list[pf.PowerflowParameters], powerflow_parameters_list
        )
        # let's assume that all samples have the same powerflow parameters
        powerflow_parameters = powerflow_parameters_list[0]
        assert all(powerflow_parameters == p for p in powerflow_parameters_list)
        return PowerflowBatch(batch, powerflow_parameters)


class CaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        case_name="case1354_pegase__api",
        data_dir="./data",
        batch_size=32,
        n_train=10_000,
        load_distribution_width=0.2,
        num_workers=min(cpu_count(), 8),
        pin_memory=False,
        **kwargs,
    ):
        super().__init__()
        self.case_name = case_name
        self.data_dir = Path(data_dir)
        self.pglib_path = Path(self.data_dir, f"pglib-opf-{PGLIB_VERSION}")
        self.batch_size = batch_size
        self.n_train = n_train
        self.load_distribution_width = load_distribution_width
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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

    def setup(self, stage: Optional[str] = None):
        # load from json
        with open(self.data_dir / f"{self.case_name}.json") as f:
            powermodels_dict = json.load(f)

        # parse the powermodels dict
        powerflow_parameters = pf.parameters_from_powermodels(powermodels_dict)
        bus_parameters = powerflow_parameters.bus_parameters()
        edge_index, edge_attr = _graph_from_parameters(powerflow_parameters)

        def dataset_from_load(load: torch.Tensor):
            bus_load = (load.mT @ powerflow_parameters.load_matrix).mT
            features = _concat_features(
                bus_load,
                bus_parameters,
            ).to(bus_load.dtype)
            return StaticGraphDataset(
                features,
                edge_index,
                edge_attr,
                powerflow_parameters,
            )

        if stage in (None, "fit"):
            # Generate the training dataset by sampling the load from a uniform distribution
            reference_load = pf.powermodels_to_tensor(
                powermodels_dict["load"], ["pd", "qd"]
            )
            # For self.load_distribution_width = eps, the load is sampled from U(1-eps, 1+eps) * reference_load
            scale = 1 + self.load_distribution_width * (
                2 * torch.rand(self.n_train, *reference_load.shape) - 1
            )
            train_load = scale * reference_load[None, ...]
            self.train_dataset = dataset_from_load(train_load)

            # For validation, load the load from the validation json file
            # The file contains a list of dicts, each dict contains the load, solution to a case
            with open(self.data_dir / f"{self.case_name}.valid.json") as f:
                validation_dicts: list[dict] = json.load(f)
                validation_load = torch.stack(
                    [
                        pf.powermodels_to_tensor(d["load"], ["pd", "qd"])
                        for d in validation_dicts
                    ]
                )
                self.val_dataset = dataset_from_load(validation_load)

        if stage in (None, "test"):
            with open(self.data_dir / f"{self.case_name}.test.json") as f:
                test_dicts: list[dict] = json.load(f)
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
