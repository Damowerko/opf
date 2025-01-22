from dataclasses import asdict, dataclass
from typing import Callable, Dict, NamedTuple

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


class BusParameters(NamedTuple):
    Ybus_sh: torch.Tensor
    base_kv: torch.Tensor
    vm_min: torch.Tensor
    vm_max: torch.Tensor
    is_ref: torch.Tensor

    @staticmethod
    def from_pf_parameters(p: PowerflowParameters) -> "BusParameters":
        return BusParameters(
            p.Ybus_sh,
            p.base_kv,
            p.vm_min,
            p.vm_max,
            p.is_ref,
        )

    def to_tensor(self) -> torch.Tensor:
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

    @staticmethod
    def from_tensor(x: torch.Tensor) -> "BusParameters":
        return BusParameters(
            torch.view_as_complex(x[:, 0:2].contiguous()),
            x[:, 2],
            x[:, 3],
            x[:, 4],
            x[:, 5],
        )


class BranchParameters(NamedTuple):
    Y: torch.Tensor
    Yc_fr: torch.Tensor
    Yc_to: torch.Tensor
    ratio: torch.Tensor
    vad_min: torch.Tensor
    vad_max: torch.Tensor
    rate_a: torch.Tensor

    @staticmethod
    def from_pf_parameters(p: PowerflowParameters) -> "BranchParameters":
        return BranchParameters(
            p.Y,
            p.Yc_fr,
            p.Yc_to,
            p.ratio,
            p.vad_min,
            p.vad_max,
            p.rate_a,
        )

    def to_tensor(self) -> torch.Tensor:
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

    @staticmethod
    def from_tensor(x: torch.Tensor) -> "BranchParameters":
        return BranchParameters(
            torch.view_as_complex(x[:, 0:2].contiguous()),
            torch.view_as_complex(x[:, 2:4].contiguous()),
            torch.view_as_complex(x[:, 4:6].contiguous()),
            torch.view_as_complex(x[:, 6:8].contiguous()),
            x[:, 8],
            x[:, 9],
            x[:, 10],
        )


class GenParameters(NamedTuple):
    Sg_min: torch.Tensor
    Sg_max: torch.Tensor
    cost_coeff: torch.Tensor

    @staticmethod
    def from_pf_parameters(p: PowerflowParameters) -> "GenParameters":
        return GenParameters(
            p.Sg_min,
            p.Sg_max,
            p.cost_coeff,
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.stack(
            [
                self.Sg_min.real,
                self.Sg_min.imag,
                self.Sg_max.real,
                self.Sg_max.imag,
                *self.cost_coeff.T,
            ],
            dim=1,
        )

    @staticmethod
    def from_tensor(x: torch.Tensor) -> "GenParameters":
        return GenParameters(
            torch.view_as_complex(x[:, 0:2].contiguous()),
            torch.view_as_complex(x[:, 2:4].contiguous()),
            x[:, 4:].T,
        )


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


def build_constraints(
    d: PowerflowVariables,
    graph: HeteroData,
) -> Dict[str, Constraint]:
    bus_params = BusParameters.from_tensor(graph["bus"]["params"])
    branch_params = BranchParameters.from_tensor(graph["branch"]["params"])
    gen_params = GenParameters.from_tensor(graph["gen"]["params"])

    # get indices from graph edges
    fr_bus = graph["bus", "from", "branch"].edge_index[0]
    to_bus = graph["bus", "to", "branch"].edge_index[0]

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
            target=torch.zeros_like(d.V.angle(), device=d.V.device),
            mask=bus_params.is_ref > 0,
        ),
        "inequality/voltage_magnitude": InequalityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            variable=d.V.abs(),
            min=bus_params.vm_min,
            max=bus_params.vm_max,
        ),
        "inequality/active_power": InequalityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            variable=d.Sg.real,
            min=gen_params.Sg_min.real,
            max=gen_params.Sg_max.real,
        ),
        "inequality/reactive_power": InequalityConstraint(
            isBus=True,
            isAngle=False,
            augmented=True,
            variable=d.Sg.imag,
            min=gen_params.Sg_min.imag,
            max=gen_params.Sg_max.imag,
        ),
        "inequality/forward_rate": InequalityConstraint(
            isBus=False,
            isAngle=False,
            augmented=True,
            variable=d.Sf.abs(),
            min=torch.zeros_like(branch_params.rate_a),
            max=branch_params.rate_a,
        ),
        "inequality/backward_rate": InequalityConstraint(
            isBus=False,
            isAngle=False,
            augmented=True,
            variable=d.St.abs(),
            min=torch.zeros_like(branch_params.rate_a),
            max=branch_params.rate_a,
        ),
        "inequality/voltage_angle_difference": InequalityConstraint(
            isBus=False,
            isAngle=True,
            augmented=True,
            variable=(d.V[..., fr_bus] * d.V[..., to_bus].conj()).angle(),
            min=branch_params.vad_min,
            max=branch_params.vad_max,
        ),
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


def powerflow_from_graph(
    V: torch.Tensor, Sd: torch.Tensor, Sg: torch.Tensor, graph: HeteroData
):
    """

    Args:
        V: Bus voltage
        Sd: Load at each bus (n_bus, 2), from data
        Sg: Generated power at each generator (n_gen, 2)
        graph: HeteroData with parameters for the powerflow problem.
    """
    bus_parameters = BusParameters.from_tensor(graph["bus"]["params"])
    branch_parameters = BranchParameters.from_tensor(graph["branch"]["params"])
    fr_bus = graph["bus", "from", "branch"].edge_index[0]
    to_bus = graph["bus", "to", "branch"].edge_index[0]
    gen_bus_ids = graph["bus", "tie", "gen"].edge_index[0]
    return _powerflow(
        V,
        Sd,
        Sg,
        bus_parameters.Ybus_sh,
        branch_parameters.Y,
        branch_parameters.Yc_fr,
        branch_parameters.Yc_to,
        branch_parameters.ratio,
        fr_bus,
        to_bus,
        gen_bus_ids,
    )


def powerflow(
    V: torch.Tensor,
    Sd: torch.Tensor,
    Sg: torch.Tensor,
    params: PowerflowParameters,
) -> PowerflowVariables:
    """
    Given the bus voltage and load, find all the other problem variables.
    The inputs and outputs should both be in the per unit system.

    Args:
        V: Bus voltage
        Sd: Load at each bus (n_bus, 2), from data
        Sg: Generated power at each generator (n_gen, 2)
        params: PowerflowParameters object.

    Reference:
        https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
        https://matpower.org/docs/MATPOWER-manual.pdf
    """
    return _powerflow(
        V,
        Sd,
        Sg,
        params.Ybus_sh,
        params.Y,
        params.Yc_fr,
        params.Yc_to,
        params.ratio,
        params.fr_bus,
        params.to_bus,
        params.gen_bus_ids,
    )


def _powerflow(
    V: torch.Tensor,
    Sd: torch.Tensor,
    Sg: torch.Tensor,
    Ybus_sh: torch.Tensor,
    Y: torch.Tensor,
    Yc_fr: torch.Tensor,
    Yc_to: torch.Tensor,
    ratio: torch.Tensor,
    fr_bus: torch.Tensor,
    to_bus: torch.Tensor,
    gen_bus_ids: torch.Tensor,
):
    Vf = V[..., fr_bus]
    Vt = V[..., to_bus]
    # voltage after transformer
    Vf_trafo = Vf / ratio
    Sf = Vf_trafo * (Y + Yc_fr).conj() * Vf_trafo.conj() - (
        Y.conj() * Vf_trafo * Vt.conj()
    )
    St = Vt * (Y + Yc_to).conj() * Vt.conj() - (Y.conj() * Vf_trafo.conj() * Vt)
    S_sh = V * Ybus_sh.conj() * V.conj()
    S_branch = torch.zeros_like(Sd)
    S_branch = S_branch.index_add(-1, fr_bus, Sf)
    S_branch = S_branch.index_add(-1, to_bus, St)
    S = S_branch + S_sh
    Sg_bus = torch.zeros_like(Sd).index_add_(-1, gen_bus_ids, Sg)
    return PowerflowVariables(V, S, Sd, Sg, Sg_bus, Sf, St)
