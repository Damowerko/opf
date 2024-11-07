from dataclasses import asdict, dataclass
from typing import Callable, Dict

import numpy as np
import torch
from torch_geometric.data import HeteroData


@dataclass
class PowerflowVariables(torch.nn.Module):
    """
    V: Bus voltage
    S: But new power injected, satisfying S = y_shunt * V**2 + Cf.T Sf + Ct.T St
    Sd: Load at each bus (n_bus, 2), from data
    Sg: Generated power at each generator (n_gen, 2)
    Sg_bus: Generated power at each bus (n_bus, 2) calculated from Sg
    Sf: Power flow from each branch (n_branch, 2)
    St: Power flow to each branch (n_branch, 2)
    """

    V: torch.Tensor
    S: torch.Tensor
    Sd: torch.Tensor
    Sg: torch.Tensor
    Sg_bus: torch.Tensor
    Sf: torch.Tensor
    St: torch.Tensor

    def __post_init__(self):
        super().__init__()

    def _apply(self, fn):
        super()._apply(fn)
        self.V = fn(self.V)
        self.S = fn(self.S)
        self.Sd = fn(self.Sd)
        self.Sg = fn(self.Sg)
        self.Sg_bus = fn(self.Sg_bus)
        self.Sf = fn(self.Sf)
        self.St = fn(self.St)
        return self

    def __getitem__(self, idx: int):
        return PowerflowVariables(
            self.V[None, idx],
            self.S[None, idx],
            self.Sd[None, idx],
            self.Sg[None, idx],
            self.Sg_bus[None, idx],
            self.Sf[None, idx],
            self.St[None, idx],
        )


@dataclass(eq=False, repr=False)
class Constraint(torch.nn.Module):
    isBus: bool  # True if bus constraint. False if branch constraint.
    isAngle: bool  # Is the constraint an angle.
    augmented: bool  # Is the constraint augmented.

    @property
    def isBranch(self) -> bool:
        return not self.isBus


@dataclass(eq=False, repr=False)
class PowerflowParameters(torch.nn.Module):
    n_bus: int
    n_branch: int
    n_gen: int
    # branch admittance parameters
    Y: torch.Tensor
    Yc_fr: torch.Tensor
    Yc_to: torch.Tensor
    ratio: torch.Tensor
    # bus shunt admittance
    Ybus_sh: torch.Tensor
    # generator cost
    cost_coeff: torch.Tensor
    # index from branch to bus
    fr_bus: torch.Tensor
    to_bus: torch.Tensor
    # index from load to bus
    load_bus_ids: torch.Tensor
    # index from gen to bus
    gen_bus_ids: torch.Tensor
    # base voltage at each bus
    base_kv: torch.Tensor
    # constraint parameters
    vm_min: torch.Tensor
    vm_max: torch.Tensor
    Sg_min: torch.Tensor
    Sg_max: torch.Tensor
    vad_min: torch.Tensor
    vad_max: torch.Tensor
    rate_a: torch.Tensor
    # path to casefile
    casefile: str
    is_ref: torch.Tensor
    # reference cost
    reference_cost: float = 1.0

    def __post_init__(self):
        super().__init__()

    def tensor_dict(self) -> Dict[str, torch.Tensor]:
        """Return a dict of all the tensors. The key is the variable name."""
        tensors = {}
        for k, v in asdict(self).items():
            if torch.is_tensor(v):
                tensors[k] = v
        return tensors

    def _apply(self, fn):
        super()._apply(fn)
        self.Y = fn(self.Y)
        self.Yc_fr = fn(self.Yc_fr)
        self.Yc_to = fn(self.Yc_to)
        self.ratio = fn(self.ratio)
        self.Ybus_sh = fn(self.Ybus_sh)
        self.cost_coeff = fn(self.cost_coeff)
        self.fr_bus = fn(self.fr_bus)
        self.to_bus = fn(self.to_bus)
        self.load_bus_ids = fn(self.load_bus_ids)
        self.gen_bus_ids = fn(self.gen_bus_ids)
        self.base_kv = fn(self.base_kv)
        self.vm_min = fn(self.vm_min)
        self.vm_max = fn(self.vm_max)
        self.Sg_min = fn(self.Sg_min)
        self.Sg_max = fn(self.Sg_max)
        self.vad_min = fn(self.vad_min)
        self.vad_max = fn(self.vad_max)
        self.rate_a = fn(self.rate_a)
        self.is_ref = fn(self.is_ref)
        return self

    def bus_parameters(self) -> torch.Tensor:
        return torch.stack(
            [
                self.Ybus_sh.real,
                self.Ybus_sh.imag,
                self.base_kv,
                self.vm_min,
                self.vm_max,
                self.is_ref,
            ],
            dim=1,
        )

    def branch_parameters(self) -> torch.Tensor:
        """
        Return a list of graph edge signals for the forward direction.

        Returns:
            (n_branch, 12) tensor.
        """
        return torch.stack(
            [
                self.Y.real,
                self.Y.imag,
                self.Yc_fr.real,
                self.Yc_fr.imag,
                self.Yc_to.real,
                self.Yc_to.imag,
                self.ratio.real,
                self.ratio.imag,
                self.vad_min,
                self.vad_max,
                self.rate_a,
            ],
            dim=1,
        )

    def gen_parameters(self) -> torch.Tensor:
        return torch.stack(
            [
                self.vm_min[self.gen_bus_ids],
                self.vm_max[self.gen_bus_ids],
                self.Sg_min.real,
                self.Sg_min.imag,
                self.Sg_max.real,
                self.Sg_max.imag,
                *self.cost_coeff.T,
            ],
            dim=1,
        )

    # @staticmethod
    # def from_graph(graph: HeteroData) -> PowerflowParameters:

    #     bus = graph["bus"].parameters
    #     Ybus_sh = torch.view_as_complex(bus[:, 0:2])
    #     base_kv = bus[:, 2]
    #     vm_min = bus[:, 3]
    #     vm_max = bus[:, 4]
    #     is_ref = bus[:, 5]
    #     # load bus ids
    #     load_bus_idx, load_index = graph["bus", "tie", "load"].edge_index

    #     branch = graph["branch"].parameters
    #     Y = torch.view_as_complex(branch[:, 0:2])
    #     Yc_fr = torch.view_as_complex(branch[:, 2:4])
    #     Yc_to = torch.view_as_complex(branch[:, 4:6])
    #     ratio = torch.view_as_complex(branch[:, 6:8])
    #     vad_min = branch[:, 8]
    #     vad_max = branch[:, 9]
    #     rate_a = branch[:, 10]
    #     # from bus indices, make sure they are sorted by branch index
    #     fr_bus, branch_index = graph["bus", "from", "branch"].edge_index
    #     fr_bus = fr_bus[branch_index.argsort()]
    #     # to bus indices, make sure they are sorted by branch index
    #     to_bus, branch_index = graph["bus", "to", "branch"].edge_index
    #     to_bus = to_bus[branch_index.argsort()]

    #     gen = graph["gen"].parameters
    #     Sg_min = torch.view_as_complex(gen[:, 2:4])
    #     Sg_max = torch.view_as_complex(gen[:, 4:6])
    #     cost_coeff = gen[:, 6:].T

    #     return PowerflowParameters(
    #         bus.shape[0],
    #         branch.shape[0],
    #         gen.shape[0],
    #         Y,
    #         Yc_fr,
    #         Yc_to,
    #         ratio,
    #         Ybus_sh,
    #         cost_coeff,
    #         fr_bus,
    #         to_bus,


@dataclass(eq=False, repr=False)
class InequalityConstraint(Constraint):
    variable: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super()._apply(fn)
        if isinstance(self.min, torch.Tensor):
            self.min = fn(self.min)
        if isinstance(self.max, torch.Tensor):
            self.max = fn(self.max)


@dataclass(eq=False, repr=False)
class EqualityConstraint(Constraint):
    value: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor | None = None


def powermodels_to_tensor(data: dict, attributes: list[str]):
    n = len(data)
    tensor = torch.zeros(n, len(attributes))
    for index, element in data.items():
        i = int(index) - 1
        for j, attribute in enumerate(attributes):
            tensor[i, j] = element[attribute]
    return tensor


def power_from_solution(load: dict, solution: dict, parameters: PowerflowParameters):
    """
    Parse a PowerModels.jl solution into PowerflowVariables.

    Args:
        load: A dict of load data with attributes "pd" and "qd".
        solution: A dict of PowerModels.jl solution.
        parameters: A PowerflowParameters object.
    Returns:
        V: Bus voltage
        S: Bus net power injected S = Sg - Sd
        Sd: Bus load power injected
    """
    # Load
    load_tensor = torch.zeros((parameters.n_bus, 2))
    load_tensor[parameters.load_bus_ids] = powermodels_to_tensor(load, ["pd", "qd"])
    Sd = torch.complex(*load_tensor.T)
    # Voltages
    V = powermodels_to_tensor(solution["bus"], ["vm", "va"])
    V = torch.polar(V[:, 0], V[:, 1])
    # Generators
    assert len(parameters.gen_bus_ids.unique()) == len(parameters.gen_bus_ids)
    Sg = torch.complex(*powermodels_to_tensor(solution["gen"], ["pg", "qg"]).T)
    return V, Sg, Sd


def build_constraints(d: PowerflowVariables, p: PowerflowParameters):
    return {
        "equality/bus_active_power": EqualityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            value=d.S.real,
            target=d.Sg_bus.real - d.Sd.real,
        ),
        "equality/bus_reactive_power": EqualityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            value=d.S.imag,
            target=d.Sg_bus.imag - d.Sd.imag,
            mask=None,
        ),
        "equality/bus_reference": EqualityConstraint(
            isBus=True,
            isAngle=True,
            augmented=True,
            value=d.V.angle(),
            target=torch.zeros(p.n_bus, device=d.V.device),
            mask=p.is_ref,
        ),
        "inequality/voltage_magnitude": InequalityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            variable=d.V.abs(),
            min=p.vm_min,
            max=p.vm_max,
        ),
        "inequality/active_power": InequalityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            variable=d.Sg.real,
            min=p.Sg_min.real,
            max=p.Sg_max.real,
        ),
        "inequality/reactive_power": InequalityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            variable=d.Sg.imag,
            min=p.Sg_min.imag,
            max=p.Sg_max.imag,
        ),
        "inequality/forward_rate": InequalityConstraint(
            isBus=False,
            isAngle=False,
            augmented=True,
            variable=d.Sf.abs(),
            min=torch.zeros_like(p.rate_a),
            max=p.rate_a,
        ),
        "inequality/backward_rate": InequalityConstraint(
            isBus=False,
            isAngle=False,
            augmented=True,
            variable=d.St.abs(),
            min=torch.zeros_like(p.rate_a),
            max=p.rate_a,
        ),
        # "inequality/voltage_angle_difference": InequalityConstraint(
        #     isBus=False,
        #     isAngle=True,
        #     augmented=True,
        #     variable=(d.V[..., p.fr_bus] * d.V[..., p.to_bus].conj()).angle(),
        #     min=p.vad_min,
        #     max=p.vad_max,
        # ),
    }


def parameters_from_powermodels(pm, casefile: str, precision=32) -> PowerflowParameters:
    if precision == 32:
        dtype = torch.float32
        cdtype = torch.complex64
    elif precision == 64:
        dtype = torch.float64
        cdtype = torch.complex128
    else:
        raise ValueError(f"Precision must be 32 or 64, got {precision}.")

    # init bus
    n_bus = len(pm["bus"])
    vm_min = torch.zeros(n_bus, dtype=dtype)
    vm_max = torch.zeros(n_bus, dtype=dtype)
    base_kv = torch.zeros(n_bus, dtype=dtype)
    is_ref = torch.zeros(n_bus, dtype=torch.bool)
    for bus in pm["bus"].values():
        i = bus["bus_i"] - 1
        vm_min[i] = bus["vmin"]
        vm_max[i] = bus["vmax"]
        base_kv[i] = bus["base_kv"]
        if bus["bus_type"] == 3:
            is_ref[i] = 1

    # init gen
    n_gen = len(pm["gen"])
    n_cost = 3  # max number of cost coefficients (c0, c1, c2), which is quadratic
    Sg_min = torch.zeros(n_gen, dtype=cdtype)
    Sg_max = torch.zeros(n_gen, dtype=cdtype)
    cost_coeff = torch.zeros((n_gen, n_cost))
    gen_bus_ids = torch.zeros(n_gen, dtype=torch.long)

    for gen in pm["gen"].values():
        i = gen["index"] - 1
        Sg_min[i] = gen["pmin"] + 1j * gen["qmin"]
        Sg_max[i] = gen["pmax"] + 1j * gen["qmax"]
        assert gen["model"] == 2  # cost is polynomial
        assert len(gen["cost"]) <= n_cost  # only real cost
        n_cost_i = int(gen["ncost"])
        # Cost is polynomial c0 + c1 x + c2 x**2
        # gen["cost"][::-1] reverses the order
        cost_coeff[i, :n_cost_i] = torch.as_tensor(gen["cost"][::-1])
        gen_bus_ids[i] = gen["gen_bus"] - 1

    # init load
    n_load = len(pm["load"])
    load_bus_ids = torch.zeros(n_load, dtype=torch.long)
    for load in pm["load"].values():
        i = load["index"] - 1
        load_bus_ids[i] = load["load_bus"] - 1

    # init branch
    n_branch = len(pm["branch"])
    fr_bus = torch.zeros((n_branch,), dtype=torch.long)
    to_bus = torch.zeros((n_branch,), dtype=torch.long)
    rate_a = torch.full((n_branch,), float("inf"), dtype=dtype)
    vad_max = torch.full((n_branch,), float("inf"), dtype=dtype)
    vad_min = torch.full((n_branch,), -float("inf"), dtype=dtype)

    Y = torch.zeros((n_branch,), dtype=cdtype)
    Yc_fr = torch.zeros((n_branch,), dtype=cdtype)
    Yc_to = torch.zeros((n_branch,), dtype=cdtype)
    ratio = torch.zeros((n_branch,), dtype=cdtype)

    for branch in pm["branch"].values():
        index = branch["index"] - 1
        fr_bus[index] = branch["f_bus"] - 1
        to_bus[index] = branch["t_bus"] - 1
        Y[index] = 1 / (branch["br_r"] + 1j * branch["br_x"])
        Yc_fr[index] = branch["g_fr"] + 1j * branch["b_fr"]
        Yc_to[index] = branch["g_to"] + 1j * branch["b_to"]
        ratio[index] = branch["tap"] * np.exp(1j * branch["shift"])
        rate_a[index] = branch["rate_a"]
        vad_max[index] = branch["angmax"]
        vad_min[index] = branch["angmin"]
    # init shunt
    Ybus_sh = torch.zeros((n_bus,), dtype=cdtype)
    for shunt in pm["shunt"].values():
        i = shunt["shunt_bus"] - 1
        Ybus_sh[i] += shunt["gs"] + 1j * shunt["bs"]

    # init constraints
    parameters = PowerflowParameters(
        n_bus,
        n_branch,
        n_gen,
        Y,
        Yc_fr,
        Yc_to,
        ratio,
        Ybus_sh,
        cost_coeff,
        fr_bus,
        to_bus,
        load_bus_ids,
        gen_bus_ids,
        base_kv,
        vm_min,
        vm_max,
        Sg_min,
        Sg_max,
        vad_min,
        vad_max,
        rate_a,
        casefile,
        is_ref,
    )
    if precision == 32:
        return parameters.float()
    elif precision == 64:
        return parameters.double()
    else:
        raise ValueError("Precision must be 32 or 64.")


def powerflow(
    V: torch.Tensor,
    Sd: torch.Tensor,
    Sg: torch.Tensor,
    params: PowerflowParameters,
) -> PowerflowVariables:
    """
    Given the bus voltage and load, find all the other problem variables.
    The inputs and outputs should both be in the per unit system.

    Reference:
        https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
        https://matpower.org/docs/MATPOWER-manual.pdf
    """
    Vf = V[..., params.fr_bus]
    Vt = V[..., params.to_bus]
    # voltage after transformer
    Vf_trafo = Vf / params.ratio
    Sf = Vf_trafo * (params.Y + params.Yc_fr).conj() * Vf_trafo.conj() - (
        params.Y.conj() * Vf_trafo * Vt.conj()
    )
    St = Vt * (params.Y + params.Yc_to).conj() * Vt.conj() - (
        params.Y.conj() * Vf_trafo.conj() * Vt
    )
    S_sh = V * params.Ybus_sh.conj() * V.conj()
    S_branch = torch.zeros_like(Sd)
    S_branch = S_branch.index_add(1, params.fr_bus, Sf)
    S_branch = S_branch.index_add(1, params.to_bus, St)
    S = S_branch + S_sh
    Sg_bus = torch.zeros_like(Sd).index_add_(1, params.gen_bus_ids, Sg)
    return PowerflowVariables(V, S, Sd, Sg, Sg_bus, Sf, St)
