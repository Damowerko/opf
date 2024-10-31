import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.transforms import ToSparseTensor

from opf.dataset import CaseDataModule, PowerflowData
from opf.hetero import HeteroGCN


@pytest.fixture
def data():
    data = FakeHeteroDataset()[0]
    assert isinstance(data, HeteroData)
    data = ToSparseTensor()(data)
    assert isinstance(data, HeteroData)
    return data


def test_hetero(data: HeteroData):
    model = HeteroGCN(
        data.metadata(),
        in_channels=-1,
        out_channels=10,
        mlp_per_gnn_layers=2,
    )
    homo = data.to_homogeneous()
    y = model(homo.x, homo.edge_index, homo.node_type, homo.edge_type)
    assert y.shape[-1] == 10


def test_hetero_no_mlp(data: HeteroData):
    model = HeteroGCN(
        data.metadata(),
        in_channels=-1,
        out_channels=10,
        mlp_per_gnn_layers=0,
    )
    homo = data.to_homogeneous()
    y = model(homo.x, homo.edge_index, homo.node_type, homo.edge_type)
    assert y.shape[-1] == 10


def test_hetero_single_layer(data: HeteroData):
    model = HeteroGCN(
        data.metadata(),
        in_channels=-1,
        out_channels=10,
        n_layers=1,
    )
    homo = data.to_homogeneous()
    y = model(homo.x, homo.edge_index, homo.node_type, homo.edge_type)
    assert y.shape[-1] == 10


@pytest.mark.parametrize("case_name", ["case30_ieee", "case118_ieee"])
def test_hetero_opf(case_name):
    datamodule = CaseDataModule(
        case_name, data_dir="test", batch_size=8, test_samples=1
    )
    datamodule.setup()
    dataloder = datamodule.train_dataloader()
    batch: PowerflowData = next(iter(dataloder))
    data, params = batch.data, batch.powerflow_parameters

    model = HeteroGCN(
        data.metadata(),
        in_channels=-1,
        n_channels=32,
        n_layers=4,
        out_channels=4,
        mlp_per_gnn_layers=2,
        mlp_read_layers=2,
    )

    homo = data.to_homogeneous()
    # set requires_grad=True for the input data
    assert homo.x is not None
    homo.x.requires_grad_(True)

    # verify that we do not have extraneous connections
    for edge_type_idx, edge_type in enumerate(data.edge_types):
        fr_type = data.node_types.index(edge_type[0])
        to_type = data.node_types.index(edge_type[2])

        fr_nodes = set(torch.nonzero(homo.node_type == fr_type).squeeze().tolist())
        to_nodes = set(torch.nonzero(homo.node_type == to_type).squeeze().tolist())

        i, j = homo.edge_index[:, homo.edge_type == edge_type_idx]  # type: ignore
        assert fr_nodes.issuperset(i.tolist())
        assert to_nodes.issuperset(j.tolist())

    y = model(homo.x, homo.edge_index, homo.node_type, homo.edge_type)

    homo.y = y
    y_dict = homo.to_heterogeneous().y_dict
    loss = sum([y.view(8, -1)[0].sum() for y in y_dict.values()])
    loss.backward()  # type: ignore

    # should have gradients only on the first batch of the input data
    for i in range(homo.num_node_types):
        # if i != 0:
        #     continue

        mask = homo.node_type == i
        assert homo.x.grad is not None
        grad = homo.x.grad[mask].view(8, -1, homo.num_node_features)
        assert grad[0, ...].abs().sum() > 0
        for j in range(1, 8):
            assert grad[j, ...].abs().sum() == 0

        # check that node idx is correct
        num_node_of_type = [params.n_bus, params.n_branch, params.n_gen][i]
        assert torch.all(
            homo.idx[homo.node_type == i].view(8, -1)
            == torch.arange(num_node_of_type)[None, :].expand(8, num_node_of_type)
        )
