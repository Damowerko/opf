import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

from opf.power import NetWrapper, load_case


class CaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        case_name,
        data_dir="./data",
        batch_size=32,
        ratio_train=0.95,
        num_workers=0,
        adj_scale=None,
        adj_threshold=0.01,
        pin_memory=False,
    ):
        super().__init__()
        self.case_name = case_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.ratio_train = ratio_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.adj_scale = adj_scale
        self.adj_threshold = adj_threshold

        self.net_wrapper = NetWrapper(load_case(case_name))
        self.dims = (1, self.net_wrapper.n_buses, 2)

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: Optional[str] = None):
        data = np.load(os.path.join(self.data_dir, f"{self.case_name}.npz"))

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

    def gso(self):
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
        return adjacency / np.max(np.real(np.linalg.eigh(adjacency)[0]))

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
             pin_memory=self.pin_memory
        )
