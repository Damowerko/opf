import torch
import torch.nn

class Readout(torch.nn.Module):
    def __init__(self, nodes: int, F_in: int, F_out: int, use_bias: bool = True):
        super().__init__()
        self.use_bias = use_bias
        self.input_dims = (-1, nodes, F_in)
        self.output_dims = (-1, nodes, F_out)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(nodes, F_out))

    def forward(self, x: torch.Tensor):
        x = x.reshape(self.input_dims)
        x = self._apply_weight(x)
        x = x.reshape(self.output_dims)
        x = self._apply_bias(x)
        return x

    def _apply_weight(self, x: torch.Tensor):
        return x

    def _apply_bias(self, x: torch.Tensor):
        if self.use_bias:
            return x + self.bias
        return x

class ReadoutMLP(Readout):
    def __init__(self, nodes: int, F_in: int, F_out: int, use_bias: bool = True):
        super().init(nodes, F_in, F_out, use_bias)
        self.weight = torch.nn.Parameter(torch.empty(nodes * F_out, nodes * F_in))

    def _apply_weight(self, x: torch.Tensor):
        return self.weight @ x.reshape(x.shape[0], -1)

class ReadoutMulti(Readout):
    def __init__(self, nodes: int, F_in: int, F_out: int, use_bias: bool = True):
        super().init(nodes, F_in, F_out, use_bias)
        self.weight = torch.nn.Parameter(torch.empty(nodes, F_out, F_in))

    def _apply_weight(self, x):
        return torch.einsum("njk,bnk -> bnj", self.weight, x)

class ReadoutLocal(Readout):
    def __init__(self, nodes: int, F_in: int, F_out: int, use_bias: bool = True):
        super().init(nodes, F_in, F_out, use_bias)
        self.weight = torch.nn.Parameter(torch.empty())

    def _apply_weight(self, x: torch.Tensor):
        return self.weight @ x