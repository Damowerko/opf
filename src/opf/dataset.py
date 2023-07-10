import os
from multiprocessing import cpu_count
from typing import Optional, Tuple
from tempfile import TemporaryDirectory

import numpy as np
import json
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch_geometric.data import Data, download_url, extract_tar
from torch_geometric.utils import dense_to_sparse

from opf.powerflow import PowerflowParameters

PGLIB_VERSION = "21.07"


def create_graph(
    params: PowerflowParameters,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a PowerflowParameters object to a torch_geometric.data.Data object.

    Args:
        powerflow_parameters (PowerflowParameters): Powerflow parameters.

    Returns:
        torch_geometric.data.Data: Graph object.
    """
    Ybus = (
        params.Cf.T @ params.Yf
        + params.Ct.T @ params.Yt
        + params.Ybus_sh
    )
    edge_index, edge_admittance = dense_to_sparse(Ybus)
    Smax 

    edge_powerlimit = params.rate_a[edge_index[0, :]]
    edge_attr = torch.stack([edge_admittance.real, edge_admittance.imag], dim=1)

    return edge_index, edge_attr


class CaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        case_name="pglib_opf_case300_ieee__api",
        data_dir="./data",
        batch_size=32,
        ratio_train=0.95,
        num_workers=min(cpu_count(), 8),
        adj_scale=None,
        adj_threshold=0.01,
        pin_memory=False,
        **kwargs,
    ):
        super().__init__()
        self.case_name = case_name
        self.data_dir = data_dir
        self.pglib_dir = os.path.join(self.data_dir, f"pglib-opf-{PGLIB_VERSION}")
        self.batch_size = batch_size
        self.ratio_train = ratio_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.adj_scale = adj_scale
        self.adj_threshold = adj_threshold

        self.dims = (1, self.net_wrapper.n_buses, 2)

        self.net = self.load_network()

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        if not os.path.exists(self.pglib_dir):
            with TemporaryDirectory() as tmp_dir:
                download_url(
                    f"https://github.com/power-grid-lib/pglib-opf/archive/refs/tags/v{PGLIB_VERSION}.tar.gz",
                    tmp_dir,
                )
                extract_tar(
                    os.path.join(tmp_dir, f"v{PGLIB_VERSION}.tar.gz"), self.data_dir
                )

    def load_network(self):
        with open(os.path.join(self.pglib_dir, f"{self.case_name}.json")) as f:
            return json.load(f)

    def setup(self, stage: Optional[str] = None):
        data = np.load(os.path.join(self.data_dir, f"old/{self.case_name}.npz"))

        if stage in (None, "fit"):
            train_data = TensorDataset(torch.from_numpy(data["train_load"]).float())
            train_split = int(len(train_data) * self.ratio_train)
            val_split = len(train_data) - train_split
            self.train_data, self.val_data = random_split(
                train_data, [train_split, val_split]
            )

        if stage in (None, "test"):
            self.test_data = TensorDataset(
                torch.from_numpy(data["test_load"]).float(),
                torch.from_numpy(data["test_bus"]).float(),
            )

    def gso(self, normalize=True):
        adjacency = self.net_wrapper.impedence_matrix()
        if self.adj_scale is None:
            # Choose scaling factor so that the mean weight is 0.5
            self.adj_scale = (
                2 * np.exp(-1) / np.mean(self.net_wrapper.impedence_matrix().data)
            )
        np.exp(-self.adj_scale * np.abs(adjacency.data), out=adjacency.data)
        adjacency.data[adjacency.data < self.adj_threshold] = 0
        adjacency = adjacency.toarray()
        # Normalize GSO by dividing by larget eigenvalue
        if normalize:
            adjacency /= np.max(np.real(np.linalg.eigh(adjacency)[0]))
        return adjacency

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
