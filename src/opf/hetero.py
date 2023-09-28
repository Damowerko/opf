import argparse
import typing
from typing import Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, EdgeType, NodeType


class HeteroGraphFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_taps: int,
        metadata: tuple[list[NodeType], list[EdgeType]],
        aggr: str | Aggregation = "sum",
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_taps: Number of taps in the filter.
            aggr: Aggregation scheme to use. For example, "add", "sum" "mean", "min", "max" or "mul".
                In addition, can be any Aggregation module (or any string that automatically resolves to it).
        """
        super().__init__()
        node_types, edge_types = metadata
        self.shift = gnn.HeteroConv(
            convs={
                edge_type: gnn.SimpleConv(
                    aggr=aggr,
                    combine_root=None,
                )
                for edge_type in edge_types
            }
        )
        self.taps = nn.ModuleList(
            [
                gnn.HeteroDictLinear(in_channels, out_channels, types=node_types)
                for _ in range(n_taps + 1)
            ]
        )

    def forward(self, x: dict[NodeType, torch.Tensor], adj_t: dict[EdgeType, Adj]):
        z: dict[NodeType, torch.Tensor] = self.taps[0](x)
        for i in range(1, len(self.taps)):
            x = self.shift(x, adj_t)
            y = self.taps[i](x)
            z = {t: z[t] + y[t] for t in z}
        return z


class HeteroMap(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: dict[NodeType, torch.Tensor]):
        return {t: self.module(x[t]) for t in x}


class HeteroDictWrapper(nn.Module):
    def __init__(
        self,
        node_types: list[NodeType],
        module: typing.Type[nn.Module],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.module = module
        self.module_dict = nn.ModuleDict(
            {t: module(*args, **kwargs) for t in node_types}
        )

    def forward(self, x: dict[NodeType, torch.Tensor]):
        for t in x:
            x[t] = self.module_dict[t](x[t])
        return x


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
                gnn.HeteroDictLinear(
                    n_channels[i],
                    n_channels[i + 1],
                    node_types,
                )
                for i in range(num_layers)
            ]
        )
        self.norms = (
            nn.ModuleList(
                [
                    HeteroDictWrapper(node_types, gnn.BatchNorm, n_channels[i + 1])
                    for i in range(num_layers - 1 if plain_last else num_layers)
                ]
            )
            if norm
            else None
        )

    def forward(self, x: dict[NodeType, torch.Tensor]):
        for i in range(self.num_layers):
            last_layer = i == self.num_layers - 1
            x = self.lins[i](x)

            if last_layer and self.plain_last:
                continue

            if self.norms:
                x = self.norms[i](x)
            x = {t: self.act(x[t]) for t in x}
            x = {t: F.dropout(x[t], p=self.dropout, training=self.training) for t in x}
        return x


class HeteroResidualBlock(nn.Module):
    def __init__(
        self,
        conv: Callable,
        act: Callable | None = None,
        norm: HeteroDictWrapper | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Residual block with a RES+ connection.
            Norm -> Activation -> Dropout -> Conv -> Residual

        Args:
            conv: Convolutional layer with input arguments (x, adj_t).
            dropout: Dropout probability.
            act: Activation function.
            norm: Normalization function with input arguments (x, type_vec).
        """
        super().__init__(**kwargs)
        self.conv = conv
        self.act = act or nn.Identity()
        self.norm = norm
        self.dropout = float(dropout)

    def forward(self, x, adj_t):
        y_dict = {t: x[t] for t in x}
        if self.norm:
            y_dict = self.norm(y_dict)
        y_dict = {t: self.act(y_dict[t]) for t in y_dict}
        y_dict = {
            t: F.dropout(y_dict[t], p=self.dropout, training=self.training)
            for t in y_dict
        }
        y_dict = self.conv(y_dict, adj_t)
        for t in x:
            x[t] = x[t] + y_dict[t]
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
            default=1,
            help="Number of MLP layers to use for readin/readout.",
        )
        group.add_argument(
            "--mlp_per_gnn_layers",
            type=int,
            default=0,
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
            "--aggr", type=str, default="sum", help="Aggregation scheme to use."
        )

    def __init__(
        self,
        metadata: tuple[list[NodeType], list[EdgeType]],
        in_channels: int,
        out_channels: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        aggr: str | Aggregation = "sum",
        **kwargs,
    ):
        """
        A simple GNN model with a readin and readout MLP. The structure of the architecture is expressed using hyperparameters. This allows for easy hyperparameter search.

        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
            n_layers: Number of GNN layers.
            n_channels: Number of hidden features on each layer.
            n_taps: Number of filter taps per layer.
            activation: Activation function to use.
            read_layers: Number of MLP layers to use for readin/readout.
            read_hidden_channels: Number of hidden features to use in the MLP layers.
            residual: Type of residual connection to use: "res", "res+", "dense", "plain".
                https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html
            normalization: Type of normalization to use: "batch" or "layer".
        """
        super().__init__()
        if isinstance(activation, str):
            activation = HeteroGCN.activation_choices[activation]()

        if mlp_read_layers < 1:
            raise ValueError("mlp_read_layers must be >= 1.")

        self.node_types, self.edge_types = metadata

        # ensure that dropout is a float
        dropout = float(dropout)

        # Readin MLP: Changes the number of features from in_channels to n_channels
        self.readin = HeteroMLP(
            node_types=self.node_types,
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=True,
        )

        # Readout MLP: Changes the number of features from n_channels to out_channels
        self.readout = HeteroMLP(
            node_types=self.node_types,
            in_channels=n_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=True,
        )
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            conv: list[tuple[Callable, str] | Callable] = [
                (
                    HeteroGraphFilter(
                        in_channels=n_channels,
                        out_channels=n_channels,
                        n_taps=n_taps,
                        metadata=metadata,
                        aggr=aggr,
                    ),
                    "x, adj_t -> x",
                )
            ]
            if mlp_per_gnn_layers > 0:
                conv += [
                    (HeteroMap(activation), "x -> x"),
                    (
                        HeteroMLP(
                            in_channels=n_channels,
                            hidden_channels=mlp_hidden_channels,
                            out_channels=n_channels,
                            num_layers=mlp_per_gnn_layers,
                            node_types=self.node_types,
                            dropout=dropout,
                            act=activation,
                            plain_last=True,
                        ),
                        "x -> x",
                    ),
                ]
            norm = HeteroDictWrapper(
                self.node_types,
                gnn.BatchNorm,
                n_channels,
            )
            self.residual_blocks += [
                HeteroResidualBlock(
                    gnn.Sequential("x, adj_t", conv),
                    dropout=dropout,
                    act=activation,
                    norm=norm,
                )
            ]

    def forward(self, x, adj_t):
        x = self.readin(x)
        for block in self.residual_blocks:
            x = block(x, adj_t)
        x = self.readout(x)
        return x
