import torch
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.data import HeteroData

from opf.models.base import ModelRegistry, OPFModel
from torch.utils.checkpoint import checkpoint
from functools import partial


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
        out_channels: dict[str, int] | int,
        combine_branch: bool,
    ):
        """
        Args:
            n_channels: The number of channels in the backbone.
            mlp_hidden_channels: The number of hidden channels in the MLPs.
            mlp_read_layers: The number of layers in the MLPs.
            dropout: The dropout rate.
            out_channels: The number of output channels for each node type. Can either be int or a dict with keys "gen", "bus", "branch".
            combine_branch: Whether to combine the branches into a single edge type.
        """

        super().__init__()
        self.combine_branch = combine_branch
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

        if isinstance(out_channels, int):
            out_channels = {
                "gen": out_channels,
                "bus": out_channels,
                "branch": 2 * out_channels,
                "branch_from": out_channels,
                "branch_to": out_channels,
            }
        if not combine_branch:
            self.readout = nn.ModuleDict(
                {
                    k: MLP(
                        in_channels=(
                            n_channels if "branch" not in k else 3 * n_channels
                        ),
                        hidden_channels=mlp_hidden_channels,
                        out_channels=out_channels[k],
                        n_layers=mlp_read_layers,
                        dropout=dropout,
                    )
                    for k in ["gen", "bus", "branch_from", "branch_to"]
                }
            )
        else:
            self.readout = nn.ModuleDict(
                {
                    k: MLP(
                        in_channels=(
                            n_channels if "branch" not in k else 4 * n_channels
                        ),
                        hidden_channels=mlp_hidden_channels,
                        out_channels=out_channels[k],
                        n_layers=mlp_read_layers,
                        dropout=dropout,
                    )
                    for k in ["gen", "bus", "branch"]
                }
            )

    def _forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        # check that there is at most one generator per bus
        gen_bus_ids = graph["gen", "tie", "bus"].edge_index[1]
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
        if not self.combine_branch:
            y_from = self.readout["branch_from"](
                torch.cat([x_from, x_to, branch_embed_from], dim=1)
            )
            y_to = self.readout["branch_to"](
                torch.cat([x_from, x_to, branch_embed_to], dim=1)
            )
            y_branch = torch.cat([y_from, y_to], dim=1)
        else:
            y_branch = self.readout["branch"](
                torch.cat([x_from, x_to, branch_embed_from, branch_embed_to], dim=1)
            )
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
        mlp_read_layers: int = 2,
        dropout: float = 0.0,
        out_channels: dict[str, int] | int = 2,
        combine_branch: bool = True,
        checkpoint_conv: bool = False,
        checkpoint_mlp: bool = False,
        **_,
    ):
        super().__init__(
            n_channels=n_channels,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_read_layers=mlp_read_layers,
            dropout=dropout,
            out_channels=out_channels,
            combine_branch=combine_branch,
        )
        self.checkpoint_conv = checkpoint_conv
        self.checkpoint_mlp = checkpoint_mlp
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

    def _layer_conv(
        self, i: int, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.conv[i](x, edge_index, edge_attr)
        # since we are using multi-head attention, we need to project the output
        x = self.head_proj[i](x)
        x = self.conv_norm[i](x + residual)
        return x

    def _layer_mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp[i](x)
        x = self.mlp_norm[i](x + residual)
        return x

    def _backbone(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # graph transformer
        for i in range(self.n_layers):
            if self.checkpoint_conv:
                x = checkpoint(
                    partial(self._layer_conv, i),
                    x,
                    edge_index,
                    edge_attr,
                    use_reentrant=False,
                )  # type: ignore
            else:
                x = self._layer_conv(i, x, edge_index, edge_attr)
            # apply MLP
            if self.enable_mlp:
                if self.checkpoint_mlp:
                    x = checkpoint(
                        partial(self._layer_mlp, i),
                        x,
                        use_reentrant=False,
                    )  # type: ignore
                else:
                    x = self._layer_mlp(i, x)
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
        combine_branch: bool = True,
        **_,
    ):
        super().__init__(
            n_channels=n_channels,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_read_layers=mlp_read_layers,
            dropout=dropout,
            out_channels=2,
            combine_branch=combine_branch,
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


class SimpleGraphTransformer(MLPIO):
    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        **_,
    ):
        """
        Args:
            n_layers: The number of transformer layers.
            embed_dim: The dimensionality of the embedding. Must be divisible by n_heads.
            n_heads: The number of attention heads. Each head will have dimension embed_dim // n_heads.
            dropout: The dropout rate.
        """

        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = float(dropout)
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.attention_window = attention_window

        # will have n_layers for encoder and n_layers for decoder
        self.Wq = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )
        self.Wk = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )
        self.Wv = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )
        self.Wo = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )

        self.norm1 = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(n_layers)]
        )
        self.norm2 = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(n_layers)]
        )
        self.linear1 = nn.ModuleList(
            [nn.Linear(self.embed_dim, 2 * self.embed_dim) for _ in range(n_layers)]
        )
        self.linear2 = nn.ModuleList(
            [nn.Linear(2 * self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )

    @staticmethod
    def _window_mask(pos: torch.Tensor, attention_window: float) -> torch.Tensor:
        """
        Create a window mask for the transformer. The mask is a matrix of shape (B, N, N) where N is the number of
        positions and B is the batch size.

        Args:
            pos: The positions of the input tensor with shape (B, N, n_dimensions).
            attention_window: The window size.

        Returns:
            A tensor of shape (B, 1, N, N) with the window mask.
        """
        distance = torch.cdist(pos, pos)
        # divide by 2 because we want to have the window size be the full width
        mask = distance <= attention_window / 2
        return mask.unsqueeze(1)

    @staticmethod
    def _connected_mask(components: torch.Tensor) -> torch.Tensor:
        """
        Compute a mask that is zero between positions that are not connected in the graph.

        Args:
            components: (B, N) tensor with integers representing the ID of connected components. If components[i] == components[j], then i and j are connected.

        Returns:
            A tensor of shape (B, 1, N, N) with the connected mask.
        """
        mask = components.unsqueeze(1) == components.unsqueeze(2)
        return mask.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        components: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (B, N, embed_dim).
            pos: The positions of the input tensor with shape (B, N, n_dimensions).
            padding_mask: (Optional) The padding mask with shape (B, N). If None, no mask is applied.
            components: (Optional) (B, N) tensor with IDs of connected components. If i and j are connected, then components[i] == components[j]. If provided, the attention matrix will be zero between different components.
        """
        B, N = x.shape[:2]
        if isinstance(self.encoding, (AbsolutePositionalEncoding, gnn.MLP)):
            x = x + self.encoding(pos.reshape(B * N, -1)).reshape(B, N, -1)

        # We need an attention mask if there is padding, windowing or we are masking by connected components.
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        if components is not None:
            connected_mask = self._connected_mask(components)
            attn_mask = (
                connected_mask if attn_mask is None else attn_mask & connected_mask
            )
        if self.attention_window > 0:
            window_mask = self._window_mask(pos, self.attention_window)
            attn_mask = window_mask if attn_mask is None else attn_mask & window_mask

        for i in range(self.n_layers):
            # using pre-norm transformer, so we apply layer norm before attention
            x_norm = self.norm1[i](x.reshape(B * N, -1)).reshape(B, N, -1)
            q = self.Wq[i](x_norm)
            k = self.Wk[i](x_norm)
            v = self.Wv[i](x_norm)
            # rotary positional encoding needs to be applied to the query and keys
            if i == 0 and isinstance(self.encoding, RotaryPositionalEncoding):
                q = self.encoding(q, pos)
                k = self.encoding(k, pos)
            # need to reshape k,q,v to have shape (B, n_heads, N, head_dim)
            # right now we have (B, N, n_heads * head_dim)
            q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

            # compute attention and reshape to "concate" heads
            x_attention = (
                F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout, attn_mask=attn_mask
                )
                .transpose(1, 2)
                .reshape(B, N, self.embed_dim)
            )
            x_attention = self.Wo[i](x_attention)
            x = x + x_attention
            # compute the feed forward layer and apply residual connection, as before
            x_fc = self.norm1[i](x_attention.reshape(B * N, -1)).reshape(B, N, -1)
            x_fc = self.linear1[i](x_fc).relu()
            x_fc = self.linear2[i](x_fc)
            x = x + x_fc
        return x


@ModelRegistry.register("mlp", True)
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
        n_bus, n_branch, n_gen = n_nodes
        self.n_nodes = {
            "bus": n_bus,
            "branch": n_branch,
            "gen": n_gen,
        }
        self.n_output = {
            "bus": 2,
            "branch": 4,
            "gen": 2,
        }

        self.mlp = MLP(
            in_channels=-1,
            hidden_channels=n_channels,
            out_channels=sum(self.n_nodes[k] * self.n_output[k] for k in self.n_nodes),
            n_layers=n_layers,
            dropout=dropout,
        )

    def _forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        graph["bus"].x = torch.cat([graph["bus"].load, graph["bus"].params], dim=-1)
        graph["branch"].x = graph["branch"].params
        graph["gen"].x = graph["gen"].params

        node_types = ["bus", "branch", "gen"]

        x_dict = graph.x_dict
        x_dict = {k: v.reshape(graph.batch_size, -1) for k, v in x_dict.items()}
        x = torch.cat([x_dict[nt] for nt in node_types], dim=1)
        y = self.mlp(x)
        chunk_sizes = [self.n_nodes[nt] * self.n_output[nt] for nt in node_types]
        y_dict = dict(zip(node_types, torch.split(y, chunk_sizes, dim=1)))
        y_dict = {
            k: v.reshape(graph.batch_size * self.n_nodes[k], self.n_output[k])
            for k, v in y_dict.items()
        }
        return y_dict
