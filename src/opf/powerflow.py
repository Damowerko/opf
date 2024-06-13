from dataclasses import asdict, dataclass
from typing import Callable, Dict

import numpy as np
import torch


@dataclass
class PowerflowVariables:
    V: torch.Tensor
    S: torch.Tensor
    Sd: torch.Tensor
    Sg: torch.Tensor
    Sf: torch.Tensor
    St: torch.Tensor
    Sbus: torch.Tensor

"""
    !!! MAY NEED CHANGES
    - add an isGen
"""
@dataclass(eq=False, repr=False)
class Constraint(torch.nn.Module):
    isBus: bool  # True if bus constraint. False if branch constraint.
    isAngle: bool  # Is the constraint an angle.

    @property
    def isBranch(self) -> bool:
        return not self.isBus

    def __post_init__(self):
        super().__init__()


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
    # matrix from branch to bus
    Cf: torch.Tensor
    Ct: torch.Tensor
    # bus shunt admittance
    Ybus_sh: torch.Tensor
    # generator cost
    cost_coeff: torch.Tensor
    # generator and load to bus mapping
    gen_matrix: torch.Tensor
    load_matrix: torch.Tensor
    # index from branch to bus
    fr_bus: torch.Tensor
    to_bus: torch.Tensor
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

    """
    may need changes (gen_matrix, gen_bus_ids)
    """
    def _apply(self, fn):
        super()._apply(fn)
        self.Y = fn(self.Y)
        self.Yc_fr = fn(self.Yc_fr)
        self.Yc_to = fn(self.Yc_to)
        self.ratio = fn(self.ratio)
        self.Cf = fn(self.Cf)
        self.Ct = fn(self.Ct)
        self.Ybus_sh = fn(self.Ybus_sh)
        self.cost_coeff = fn(self.cost_coeff)
        self.gen_matrix = fn(self.gen_matrix)
        self.load_matrix = fn(self.load_matrix)
        self.fr_bus = fn(self.fr_bus)
        self.to_bus = fn(self.to_bus)
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
                # self.vm_min[self.gen_bus_ids],
                # self.vm_max[self.gen_bus_ids],
                self.Sg_min.real,
                self.Sg_max.real,
                self.Sg_min.imag,
                self.Sg_max.imag,
                *self.cost_coeff.T,
            ],
            dim=1,
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

"""
literally only used in a test case what
- may need changes
"""
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
    Sd = torch.complex(
        *powermodels_to_tensor(load, ["pd", "qd"]).T @ parameters.load_matrix
    )
    # Voltages
    V = powermodels_to_tensor(solution["bus"], ["vm", "va"])
    V = torch.polar(V[:, 0], V[:, 1])
    # Gen

    # could I get rid of gen_matrix???
    # I am sure that I could replace this with gen_bus_ids somehow
    # then again this is just for test cases(?), so I am not sure
    # that there is a point to making changes here
    Sg_unfiltered = torch.complex(
        *powermodels_to_tensor(solution["gen"], ["pg", "qg"]).T @ parameters.gen_matrix
    )
    Sg = Sg_unfiltered[parameters.gen_bus_ids]
    return V, Sg, Sd


def build_constraints(d: PowerflowVariables, p: PowerflowParameters):
    return {
        "equality/bus_power": EqualityConstraint(True, False, d.Sbus, d.S),
        "equality/bus_reference": EqualityConstraint(
            True, True, d.V.angle(), torch.zeros(p.n_bus).to(d.V.device), p.is_ref
        ),
        "inequality/voltage_magnitude": InequalityConstraint(
            True, False, d.V.abs(), p.vm_min, p.vm_max
        ),
        "inequality/active_power": InequalityConstraint(
            True, False, d.Sg.real, p.Sg_min.real, p.Sg_max.real
        ),
        "inequality/reactive_power": InequalityConstraint(
            True, False, d.Sg.imag, p.Sg_min.imag, p.Sg_max.imag
        ),
        "inequality/forward_rate": InequalityConstraint(
            False, False, d.Sf.abs(), torch.zeros_like(p.rate_a), p.rate_a
        ),
        "inequality/backward_rate": InequalityConstraint(
            False, False, d.St.abs(), torch.zeros_like(p.rate_a), p.rate_a
        ),
        "inequality/voltage_angle_difference": InequalityConstraint(
            False,
            True,
            ((d.V @ p.Cf.T) * (d.V @ p.Ct.T).conj()).angle(),
            p.vad_min,
            p.vad_max,
        ),
    }

"""
    !!! NEEDS CHANGES !!!
    - depends on changes to params
"""
def parameters_from_powermodels(pm, casefile: str, precision=32) -> PowerflowParameters:
    dtype = torch.complex128 if precision == 64 else torch.complex64

    # init bus
    n_bus = len(pm["bus"])
    vm_min = torch.zeros(n_bus)
    vm_max = torch.zeros(n_bus)
    base_kv = torch.zeros(n_bus)
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
    Sg_min = torch.zeros(n_gen, dtype=dtype)
    Sg_max = torch.zeros(n_gen, dtype=dtype)
    gen_matrix = torch.zeros(n_gen, n_bus)
    cost_coeff = torch.zeros((n_gen, n_cost))
    gen_bus_ids = torch.zeros(n_gen)

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
        gen_matrix[i, gen["gen_bus"] - 1] = 1

    # init load
    n_load = len(pm["load"])
    # load_matrix = torch.zeros((n_load, (n_bus+n_gen)))
    load_matrix = torch.zeros((n_load, n_bus))
    for load in pm["load"].values():
        i = load["load_bus"] - 1
        load_matrix[load["index"] - 1, i] = 1

    # init branch
    n_branch = len(pm["branch"])
    fr_bus = torch.zeros((n_branch,), dtype=torch.long)
    to_bus = torch.zeros((n_branch,), dtype=torch.long)
    rate_a = torch.full((n_branch,), float("inf"))
    vad_max = torch.full((n_branch,), float("inf"))
    vad_min = torch.full((n_branch,), -float("inf"))

    Y = torch.zeros((n_branch,), dtype=dtype)
    Yc_fr = torch.zeros((n_branch,), dtype=dtype)
    Yc_to = torch.zeros((n_branch,), dtype=dtype)
    ratio = torch.zeros((n_branch,), dtype=dtype)
    Cf = torch.zeros((n_branch, n_bus), dtype=dtype)
    Ct = torch.zeros((n_branch, n_bus), dtype=dtype)
    for branch in pm["branch"].values():
        index = branch["index"] - 1
        fr_bus[index] = branch["f_bus"] - 1
        to_bus[index] = branch["t_bus"] - 1
        Y[index] = 1 / (branch["br_r"] + 1j * branch["br_x"])
        Yc_fr[index] = branch["g_fr"] + 1j * branch["b_fr"]
        Yc_to[index] = branch["g_to"] + 1j * branch["b_to"]
        ratio[index] = branch["tap"] * np.exp(1j * branch["shift"])
        Cf[index, fr_bus[index]] = 1
        Ct[index, to_bus[index]] = 1
        rate_a[index] = branch["rate_a"]
        vad_max[index] = branch["angmax"]
        vad_min[index] = branch["angmin"]

    # init shunt
    Ybus_sh = torch.zeros((n_bus,), dtype=dtype)
    for shunt in pm["shunt"].values():
        i = shunt["shunt_bus"] - 1
        Ybus_sh[i] += shunt["gs"] + 1j * shunt["bs"]

    # init constraints
    return PowerflowParameters(
        n_bus,
        n_branch,
        n_gen,
        Y,
        Yc_fr,
        Yc_to,
        ratio,
        Cf,
        Ct,
        Ybus_sh,
        cost_coeff,
        gen_matrix,
        load_matrix,
        fr_bus,
        to_bus,
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


def powerflow(
    V: torch.Tensor, Sg: torch.Tensor, Sd: torch.Tensor, params: PowerflowParameters
) -> PowerflowVariables:
    """
    Find the branch variables given the bus voltages. The inputs and outputs should both be
    in the per unit system.

    Reference:
        https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
        https://matpower.org/docs/MATPOWER-manual.pdf
    """
    Vf = V[..., params.fr_bus]
    Vt = V[..., params.to_bus]
    Sf = (params.Y + params.Yc_fr).conj() * Vf.abs() ** 2 / params.ratio.abs() ** 2 - (
        params.Y.conj() * Vf * Vt.conj() / params.ratio
    )
    St = (params.Y + params.Yc_to).conj() * Vt.abs() ** 2 - (
        params.Y.conj() * Vf.conj() * Vt / params.ratio.conj()
    )
    Sbus_sh = params.Ybus_sh.conj() * V.abs() ** 2
    Sbus_branch = Sf @ params.Cf + St @ params.Ct
    # alternate method
    # Sbus_branch = torch.zeros_like(Sbus_sh)
    # for i in range(len(Sbus_branch)):
    #     Sbus_branch[i] += torch.sum(Sf[params.fr_bus == i])
    #     Sbus_branch[i] += torch.sum(St[params.to_bus == i])
    Sbus = Sbus_branch + Sbus_sh
    Sg_unfiltered = torch.zeros_like(Sd)
    Sg_unfiltered[params.gen_bus_ids.int()] = Sg
    S = Sg_unfiltered - Sd
    return PowerflowVariables(V, S, Sd, Sg, Sf, St, Sbus)
