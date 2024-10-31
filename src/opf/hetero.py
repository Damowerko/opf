import argparse
import typing
from typing import Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import Adj, EdgeType, NodeType, Tensor
from torch_geometric.utils import index_sort


class HeteroMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        node_types: list[NodeType],
        dropout: float = 0.0,
        act: nn.Module = nn.LeakyReLU(),
        norm: bool = True,
        plain_last: bool = True,
        is_sorted: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.plain_last = plain_last
        self.act = act

        n_channels = (
            [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        )
        self.lins = nn.ModuleList(
            [
                gnn.HeteroLinear(
                    n_channels[i],
                    n_channels[i + 1],
                    len(node_types),
                    is_sorted=is_sorted,
                )
                for i in range(num_layers)
            ]
        )
        self.norms = (
            nn.ModuleList(
                [
                    gnn.HeteroBatchNorm(
                        hidden_channels, len(node_types), n_channels[i + 1]
                    )
                    for i in range(num_layers - 1 if plain_last else num_layers)
                ]
            )
            if norm
            else None
        )

    def forward(self, x: Tensor, node_type: Tensor):
        """
        Args:
            x: Node feature tensor.
            node_type: Type of each node.
        """
        for i in range(self.num_layers):
            last_layer = i == self.num_layers - 1
            x = self.lins[i](x, node_type)
            if last_layer and self.plain_last:
                continue
            if self.norms:
                x = self.norms[i](x, node_type)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HeteroResidualBlock(nn.Module):
    def __init__(
        self,
        conv: Callable,
        conv_norm: Callable,
        mlp: Callable,
        mlp_norm: Callable,
        act: Callable,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Residual block with a RES+ connection.
            Norm -> Activation -> Dropout -> Module -> Residual

        Args:
            conv: Convolutional layer with input arguments (x, adj_t, edge_type).
            conv_norm: Normalization layer with input arguments (x, node_type).
            mlp: MLP layer with input arguments (x, node_type).
            mlp_norm: Normalization layer with input arguments (x, node_type).
            act: Activation function. Input arguments (x,).
            dropout: Dropout probability.
        """
        super().__init__(**kwargs)
        self.conv = conv
        self.conv_norm = conv_norm
        self.mlp = mlp
        self.mlp_norm = mlp_norm
        self.act = act or nn.Identity()
        self.dropout = float(dropout)

    def forward(self, x: Tensor, adj_t: Adj, node_type: Tensor, edge_type: Tensor):
        # Convolutional layer and residual connection
        y = x
        y = self.conv_norm(y, node_type)
        y = self.act(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        x = x + self.conv(y, adj_t, edge_type)
        # MLP layer and residual connection
        y = x
        y = self.mlp_norm(y, node_type)
        y = self.act(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        x = x + self.mlp(y, node_type)
        return x


class HeteroGCN(nn.Module):
    activation_choices: typing.Dict[str, Type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
    }

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(HeteroGCN.__name__)
        group.add_argument(
            "--n_taps",
            type=int,
            default=4,
            help="Number of filter taps per layer.",
        )
        group.add_argument(
            "--n_channels",
            type=int,
            default=32,
            help="Number of hidden features on each layer.",
        )
        group.add_argument(
            "--n_layers", type=int, default=2, help="Number of GNN layers."
        )
        group.add_argument(
            "--activation",
            type=str,
            default="leaky_relu",
            choices=list(HeteroGCN.activation_choices),
        )
        group.add_argument(
            "--mlp_read_layers",
            type=int,
            default=2,
            help="Number of MLP layers to use for readin/readout.",
        )
        group.add_argument(
            "--mlp_per_gnn_layers",
            type=int,
            default=2,
            help="Number of MLP layers to use per GNN layer.",
        )
        group.add_argument(
            "--mlp_hidden_channels",
            type=int,
            default=256,
            help="Number of hidden features to use in the MLP layers.",
        )
        group.add_argument(
            "--dropout", type=float, default=0.0, help="Dropout probability."
        )
        group.add_argument(
            "--aggr", type=str, default="mean", help="Aggregation scheme to use."
        )

    def __init__(
        self,
        metadata: tuple[list[NodeType], list[EdgeType]],
        in_channels: int,
        out_channels: int | nn.Module,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        aggr: str = "mean",
        **kwargs,
    ):
        """
        A simple GNN model with a readin and readout MLP. The structure of the architecture is expressed using hyperparameters. This allows for easy hyperparameter search.

        Args:
            in_channels: Number of input features. Can be a dictionary with node types as keys.
            out_channels: Number of output features. Can instead be a module that takes takes in the output of the last GNN layer and returns the final output.
            n_layers: Number of GNN layers.
            n_channels: Number of hidden features on each layer.
            activation: Activation function to use.
            read_layers: Number of MLP layers to use for readin/readout.
            read_hidden_channels: Number of hidden features to use in the MLP layers.
            residual: Type of residual connection to use: "res", "res+", "dense", "plain".
                https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html
            normalization: Type of normalization to use: "batch" or "layer".
        """
        super().__init__()
        if isinstance(activation, str):
            self.activation = HeteroGCN.activation_choices[activation]()
        else:
            self.activation = activation

        if mlp_read_layers < 1:
            raise ValueError("mlp_read_layers must be >= 1.")

        self.node_types, self.edge_types = metadata

        # ensure that dropout is a float
        self.dropout = float(dropout)

        # Readin MLP: Changes the number of features from in_channels to n_channels
        self.readin = HeteroMLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels,
            num_layers=mlp_read_layers,
            node_types=self.node_types,
            dropout=self.dropout,
            act=self.activation,
            plain_last=True,
            is_sorted=True,
        )
        # Readout MLP: Changes the number of features from n_channels to out_channels
        if isinstance(out_channels, int):
            self.readout = HeteroMLP(
                in_channels=n_channels,
                hidden_channels=mlp_hidden_channels,
                out_channels=(
                    out_channels if isinstance(out_channels, int) else n_channels
                ),
                num_layers=mlp_read_layers,
                node_types=self.node_types,
                dropout=self.dropout,
                act=self.activation,
                plain_last=True,
                is_sorted=True,
            )
        else:
            self.readout = out_channels

        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            conv = gnn.RGCNConv(
                in_channels=n_channels,
                out_channels=n_channels,
                num_relations=len(self.edge_types),
                aggr=aggr,
                is_sorted=True,
            )
            mlp = HeteroMLP(
                in_channels=n_channels,
                hidden_channels=mlp_hidden_channels,
                out_channels=n_channels,
                num_layers=mlp_per_gnn_layers,
                node_types=self.node_types,
                dropout=dropout,
                act=self.activation,
                plain_last=True,
                is_sorted=True,
            )
            self.residual_blocks.append(
                HeteroResidualBlock(
                    conv=conv,
                    conv_norm=gnn.HeteroBatchNorm(n_channels, len(self.node_types)),
                    mlp=mlp,
                    mlp_norm=gnn.HeteroBatchNorm(n_channels, len(self.node_types)),
                    act=self.activation,
                    dropout=dropout,
                )
            )

    def forward(self, x, edge_index, node_type, edge_type):
        x = self.readin(x, node_type)
        for block in self.residual_blocks:
            x = block(x, edge_index, node_type.clone(), edge_type)
        x = self.readout(x, node_type)
        return x
