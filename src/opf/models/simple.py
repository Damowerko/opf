import torch
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.data import HeteroData

from opf.models.base import ModelRegistry, OPFModel


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.dropout = dropout

        layers = []
        if in_channels == -1:
            layers.append(nn.LazyLinear(hidden_channels))
        else:
            layers.append(nn.Linear(in_channels, hidden_channels))
        for i in range(1, self.n_layers):
            layers += [
                gnn.BatchNorm(self.hidden_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(
                    self.hidden_channels,
                    (
                        self.hidden_channels
                        if i < self.n_layers - 1
                        else self.out_channels
                    ),
                ),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPIO(OPFModel):
    def __init__(
        self,
        n_channels: int,
        mlp_hidden_channels: int,
        mlp_read_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.readin = nn.ModuleDict(
            {
                k: MLP(
                    in_channels=-1,
                    hidden_channels=mlp_hidden_channels,
                    out_channels=n_channels,
                    n_layers=mlp_read_layers,
                    dropout=dropout,
                )
                for k in ["gen", "bus", "branch_from", "branch_to"]
            }
        )
        self.readout = nn.ModuleDict(
            {
                k: MLP(
                    in_channels=(n_channels if "branch" not in k else 3 * n_channels),
                    hidden_channels=mlp_hidden_channels,
                    out_channels=2,
                    n_layers=mlp_read_layers,
                    dropout=dropout,
                )
                for k in ["gen", "bus", "branch_from", "branch_to"]
            }
        )

    def _forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        # check that there is at most one generator per bus
        gen_bus_ids = graph["gen", "tie", "bus"].edge_index[0]
        if (
            not torch.compiler.is_compiling()
            and gen_bus_ids.unique().shape[0] != gen_bus_ids.shape[0]
        ):
            raise AssertionError("There should be exactly one generator per bus.")
        # readin, is somewhat complex since we start with a heterogenous graph
        # and we need to convert it to a homogenous graph
        # I assume that each bus has at most one generator, so the graph features there represent both (added together)
        x_gen = self.readin["gen"](graph["bus"].params[gen_bus_ids])
        x_bus = self.readin["bus"](
            torch.cat([graph["bus"].load, graph["bus"].params], dim=-1)
        )
        x = x_bus.index_add(0, gen_bus_ids, x_gen)

        # Branches have transformers on ONE end, so its assymetric
        # Therefore I need edges going in the opposite direction
        edge_index_from, edge_index_to = graph["bus", "branch", "bus"].edge_index
        edge_index = torch.cat(
            [
                torch.stack([edge_index_from, edge_index_to], dim=0),
                torch.stack([edge_index_to, edge_index_from], dim=0),
            ],
            dim=1,
        )
        # Since branches are assymetric, I use two different MLPs to embed the edge parameters
        # I concatenate the embeddings of the two directions, just like I did the edge indices
        branch_params = graph["bus", "branch", "bus"].params
        branch_embed_from = self.readin["branch_from"](branch_params)
        branch_embed_to = self.readin["branch_to"](branch_params)
        edge_attr = torch.cat([branch_embed_from, branch_embed_to], dim=0)

        x = self._backbone(x, edge_index, edge_attr)

        # in the homogenous graph, all nodes are buses
        y_bus = self.readout["bus"](x)
        # some of the nodes have a generator
        y_gen = self.readout["gen"](x[gen_bus_ids])
        # I compute any branch variable outputs by hand
        x_from = x[edge_index_from]
        x_to = x[edge_index_to]
        y_from = self.readout["branch_from"](
            torch.cat([x_from, x_to, branch_embed_from], dim=1)
        )
        y_to = self.readout["branch_to"](
            torch.cat([x_from, x_to, branch_embed_to], dim=1)
        )
        y_branch = torch.cat([y_from, y_to], dim=1)
        return {
            "bus": y_bus,
            "gen": y_gen,
            "branch": y_branch,
        }

    def _backbone(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


@ModelRegistry.register("simplegat", False)
class SimpleGAT(MLPIO):
    def __init__(
        self,
        n_channels: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_hidden_channels: int = 512,
        mlp_per_gnn_layers: int = 2,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__(
            n_channels=n_channels,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_read_layers=mlp_per_gnn_layers,
            dropout=dropout,
        )
        self.n_layers = n_layers
        self.enable_mlp = mlp_per_gnn_layers > 0

        if n_channels % n_heads != 0:
            raise AssertionError(
                f"hidden_channels should be divisible by n_heads. Got {n_channels} and {n_heads}"
            )
        self.conv = nn.ModuleList(
            [
                gnn.GATv2Conv(
                    in_channels=n_channels,
                    out_channels=n_channels // n_heads,
                    heads=n_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=n_channels,
                )
                for _ in range(n_layers)
            ]
        )
        self.conv_norm = nn.ModuleList(
            [gnn.BatchNorm(n_channels) for _ in range(n_layers)]
        )
        self.head_proj = nn.ModuleList(
            [nn.Linear(n_channels, n_channels) for _ in range(n_layers)]
        )
        if self.enable_mlp:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_channels=n_channels,
                        hidden_channels=mlp_hidden_channels,
                        out_channels=n_channels,
                        n_layers=mlp_per_gnn_layers,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp_norm = nn.ModuleList(
                [gnn.BatchNorm(n_channels) for _ in range(n_layers)]
            )

    def _backbone(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # graph transformer
        for i in range(self.n_layers):
            residual = x
            x = self.conv[i](x, edge_index, edge_attr)
            # since we are using multi-head attention, we need to project the output
            x = self.head_proj[i](x)
            x = self.conv_norm[i](x + residual)
            residual = x
            # apply MLP
            if self.enable_mlp:
                x = self.mlp[i](x)
                x = self.mlp_norm[i](x + residual)
        return x


@ModelRegistry.register("simplegated", False)
class SimpleGated(MLPIO):
    def __init__(
        self,
        n_channels: int = 128,
        n_layers: int = 4,
        mlp_hidden_channels: int = 512,
        mlp_per_gnn_layers: int = 2,
        mlp_read_layers: int = 2,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__(
            n_channels=n_channels,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_read_layers=mlp_read_layers,
            dropout=dropout,
        )
        self.n_layers = n_layers
        self.enable_mlp = mlp_per_gnn_layers > 0
        self.conv = nn.ModuleList(
            [
                gnn.ResGatedGraphConv(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    edge_dim=n_channels,
                )
                for _ in range(n_layers)
            ]
        )
        self.conv_norm = nn.ModuleList(
            [gnn.BatchNorm(n_channels) for _ in range(n_layers)]
        )
        if self.enable_mlp:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_channels=n_channels,
                        hidden_channels=mlp_hidden_channels,
                        out_channels=n_channels,
                        n_layers=mlp_per_gnn_layers,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp_norm = nn.ModuleList(
                [gnn.BatchNorm(n_channels) for _ in range(n_layers)]
            )

    def _backbone(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        for i in range(self.n_layers):
            residual = x
            x = self.conv_norm[i](x).relu()
            x = self.conv[i](x, edge_index, edge_attr)
            x = x + residual
            if self.enable_mlp:
                residual = x
                x = self.mlp_norm[i](x).relu()
                x = self.mlp[i](x)
                x = x + residual
        return x


@ModelRegistry.register("mlp", False)
class OPFMLP(OPFModel):
    def __init__(
        self,
        n_nodes: tuple[int, int, int],
        n_channels: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__()
        self.mlp = MLP(
            in_channels=-1,
            hidden_channels=n_channels,
            out_channels=2,
            n_layers=n_layers,
            dropout=dropout,
        )
