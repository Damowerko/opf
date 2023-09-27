import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.transforms import ToSparseTensor

from opf.hetero import HeteroParametricGNN


def test_hetero():
    dataset = FakeHeteroDataset(avg_num_nodes=100)
    data = dataset[0]
    assert isinstance(data, HeteroData)
    transform = ToSparseTensor()
    data = transform(data)
    assert isinstance(data, HeteroData)

    model = HeteroParametricGNN(
        data.metadata(), in_channels=-1, out_channels=10, n_taps=4, mlp_per_gnn_layers=2
    )
    y_dict = model(data.x_dict, data.adj_t_dict)
    for node_type in y_dict:
        assert y_dict[node_type].shape[-1] == 10
