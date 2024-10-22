import pytest
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.transforms import ToSparseTensor

from opf.hetero import HeteroGCN


@pytest.fixture
def data():
    data = FakeHeteroDataset()[0]
    assert isinstance(data, HeteroData)
    data = ToSparseTensor()(data)
    assert isinstance(data, HeteroData)
    return data


def check_shape(y_dict, out_channels):
    for node_type in y_dict:
        assert y_dict[node_type].shape[-1] == out_channels


def test_hetero(data: HeteroData):
    model = HeteroGCN(
        data.metadata(), in_channels=-1, out_channels=10, n_taps=4, mlp_per_gnn_layers=2
    )
    y_dict = model(data.x_dict, data.adj_t_dict)
    check_shape(y_dict, 10)


def test_hetero_no_mlp(data: HeteroData):
    model = HeteroGCN(
        data.metadata(), in_channels=-1, out_channels=10, n_taps=4, mlp_per_gnn_layers=0
    )
    y_dict = model(data.x_dict, data.adj_t_dict)
    check_shape(y_dict, 10)


def test_hetero_single_layer(data: HeteroData):
    model = HeteroGCN(
        data.metadata(), in_channels=-1, out_channels=10, n_taps=4, n_layers=1
    )
    y_dict = model(data.x_dict, data.adj_t_dict)
    check_shape(y_dict, 10)


def test_hetero_single_tap(data: HeteroData):
    model = HeteroGCN(data.metadata(), in_channels=-1, out_channels=10, n_taps=1)
    y_dict = model(data.x_dict, data.adj_t_dict)
    check_shape(y_dict, 10)


def test_sort():
    x = torch.arange(10)
    edge_index = torch.stack([torch.arange(10), (torch.arange(10) + 1) % 10], dim=0)
    node_type = torch.arange(10) % 2
    edge_type = torch.arange(10) % 2

    x1, _, _, _, node_perm = HeteroGCN.sort(x, edge_index, node_type, edge_type)
    assert torch.any(x1 != x)
    x2 = HeteroGCN.unsort(x1, node_perm)
    assert torch.all(x2 == x)
