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
    If: torch.Tensor
    It: torch.Tensor
    Sf: torch.Tensor
    St: torch.Tensor
    Sbus: torch.Tensor


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
    # branch admittance parameters
    Yf: torch.Tensor
    Yff: torch.Tensor
    Yft: torch.Tensor
    Yt: torch.Tensor
    Ytf: torch.Tensor
    Ytt: torch.Tensor
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
        self.Yf = fn(self.Yf)
        self.Yff = fn(self.Yff)
        self.Yft = fn(self.Yft)
        self.Yt = fn(self.Yt)
        self.Ytf = fn(self.Ytf)
        self.Ytt = fn(self.Ytt)
        self.Cf = fn(self.Cf)
        self.Ct = fn(self.Ct)
        self.Ybus_sh = fn(self.Ybus_sh)
        self.cost_coeff = fn(self.cost_coeff)
        self.gen_matrix = fn(self.gen_matrix)
        self.load_matrix = fn(self.load_matrix)
        self.fr_bus = fn(self.fr_bus)
        self.to_bus = fn(self.to_bus)
        self.base_kv = fn(self.base_kv)
        self.vm_min = fn(self.vm_min)
        self.vm_max = fn(self.vm_max)
        self.Sg_min = fn(self.Sg_min)
        self.Sg_max = fn(self.Sg_max)
        self.vad_min = fn(self.vad_min)
        self.vad_max = fn(self.vad_max)
        self.rate_a = fn(self.rate_a)

    def bus_parameters(self) -> torch.Tensor:
        """
        A tesor representing the parameters of the bus.

        Returns:
            (n_bus, 10) tensor.
        """
        return torch.stack(
            [
                self.Ybus_sh.real,
                self.Ybus_sh.imag,
                self.base_kv,
                self.vm_min,
                self.vm_max,
                self.Sg_min.real,
                self.Sg_max.real,
                self.Sg_min.imag,
                self.Sg_min.imag,
                *self.cost_coeff.T,
            ],
            dim=1,
        )

    def forward_branch_parameters(self) -> torch.Tensor:
        """
        Return a list of graph edge signals for the forward direction.

        Returns:
            (n_branch, 7) tensor.
        """
        return torch.stack(
            [
                self.Yff.real,
                self.Yff.imag,
                self.Ytf.real,
                self.Ytf.imag,
                self.vad_min,
                self.vad_max,
                self.rate_a,
            ],
            dim=1,
        )

    def backward_branch_parameters(self) -> torch.Tensor:
        """
        Return a list of graph edge signals for the backward direction.

        Returns:
            (n_branch, 7) tensor.
        """
        return torch.stack(
            [
                self.Ytt.real,
                self.Ytt.imag,
                self.Yft.real,
                self.Yft.imag,
                self.vad_min,
                self.vad_max,
                self.rate_a,
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


def powermodels_to_tensor(data: dict, attributes: list[str]):
    n = len(data)
    tensor = torch.zeros(n, len(attributes))
    for element in data.values():
        i = element["index"] - 1
        for j, attribute in enumerate(attributes):
            tensor[i, j] = element[attribute]
    return tensor


def power_from_solution(solution: dict, network: PowerflowParameters):
    """
    Parse a PowerModels.jl solution into PowerflowVariables.

    Args:
        solution: A PowerModels.jl solution as outputted form `generate_test.jl`

    Returns:
        V: Bus voltage
        S: Bus net power injected S = Sg - Sd
        Sd: Bus load power injected
    """
    # Voltages
    V = powermodels_to_tensor(solution["bus"], ["vm", "va"])
    V = torch.polar(V[:, 0], V[:, 1])
    # Load
    Sd = (
        torch.complex(*powermodels_to_tensor(solution["gen"], ["pg", "qg"]).T)
        @ network.load_matrix
    )
    # Gen
    Sg = (
        torch.complex(*powermodels_to_tensor(solution["gen"], ["pg", "qg"]).T)
        @ network.gen_matrix
    )
    S = Sg - Sd
    return V, S, Sd


def build_constraints(d: PowerflowVariables, p: PowerflowParameters):
    return {
        "equality/bus_power": EqualityConstraint(True, False, d.Sbus, d.S),
        "inequality/voltage_magnitude": InequalityConstraint(
            True, True, d.V.abs(), p.vm_min, p.vm_max
        ),
        "inequality/active_power": InequalityConstraint(
            True, False, d.Sg.real, p.Sg_min.real, p.Sg_min.real
        ),
        "inequality/reactive_power": InequalityConstraint(
            True, False, d.Sg.imag, p.Sg_min.imag, p.Sg_min.imag
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


def parameters_from_powermodels(pm) -> PowerflowParameters:
    # init bus
    n_bus = len(pm["bus"])
    vm_min = torch.zeros(n_bus)
    vm_max = torch.zeros(n_bus)
    base_kv = torch.zeros(n_bus)
    reference_buses = []
    for bus in pm["bus"].values():
        i = bus["bus_i"] - 1
        vm_min[i] = bus["vmin"]
        vm_max[i] = bus["vmax"]
        base_kv[i] = bus["base_kv"]
        if bus["bus_type"] == 3:
            reference_buses.append(i)

    # init gen
    n_gen = len(pm["gen"])
    n_cost = 3  # max number of cost coefficients (c0, c1, c2), which is quadratic
    Sg_min = torch.zeros(n_bus, dtype=torch.complex64)
    Sg_max = torch.zeros(n_bus, dtype=torch.complex64)
    gen_matrix = torch.zeros((n_gen, n_bus))
    cost_coeff = torch.zeros((n_bus, n_cost))

    for gen in pm["gen"].values():
        i = gen["gen_bus"] - 1
        Sg_min[i] = gen["pmin"] + 1j * gen["qmin"]
        Sg_max[i] = gen["pmax"] + 1j * gen["qmax"]
        assert gen["model"] == 2  # cost is polynomial
        assert len(gen["cost"]) <= n_cost  # only real cost
        n_cost_i = int(gen["ncost"])
        # Cost is polynomial c0 + c1 x + c2 x**2
        cost_coeff[i, :n_cost_i] = torch.as_tensor(gen["cost"][::-1])
        gen_matrix[gen["index"] - 1, i] = 1

    # init load
    n_load = len(pm["load"])
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

    Yff = np.zeros((n_branch,), dtype=np.complex64)
    Yft = np.zeros((n_branch,), dtype=np.complex64)
    Ytf = np.zeros((n_branch,), dtype=np.complex64)
    Ytt = np.zeros((n_branch,), dtype=np.complex64)
    Cf = np.zeros((n_branch, n_bus), dtype=np.complex64)
    Ct = np.zeros((n_branch, n_bus), dtype=np.complex64)
    for branch in pm["branch"].values():
        index = branch["index"] - 1
        fr_bus[index] = branch["f_bus"] - 1
        to_bus[index] = branch["t_bus"] - 1
        y = 1 / (branch["br_r"] + 1j * branch["br_x"])
        yc_fr = branch["g_fr"] + 1j * branch["b_fr"]
        yc_to = branch["g_to"] + 1j * branch["b_to"]
        ratio = branch["tap"] * np.exp(1j * branch["shift"])
        Yff[index] = (y + yc_fr) / np.abs(ratio) ** 2
        Yft[index] = -y / np.conj(ratio)
        Ytt[index] = y + yc_to
        Ytf[index] = -y / ratio
        Cf[index, fr_bus[index]] = 1
        Ct[index, to_bus[index]] = 1
        rate_a[index] = branch["rate_a"]
        vad_max[index] = branch["angmax"]
        vad_min[index] = branch["angmin"]
    Yf = torch.from_numpy(np.diag(Yff).dot(Cf) + np.diag(Yft).dot(Ct))
    Yt = torch.from_numpy(np.diag(Ytf).dot(Cf) + np.diag(Ytt).dot(Ct))
    Cf = torch.from_numpy(Cf)
    Ct = torch.from_numpy(Ct)

    # init shunt
    Ybus_sh = np.zeros((n_bus), dtype=np.complex64)
    for shunt in pm["shunt"].values():
        i = shunt["shunt_bus"] - 1
        Ybus_sh[i] += shunt["gs"] + 1j * shunt["bs"]
    Ybus_sh = torch.from_numpy(Ybus_sh)

    # init constraints
    return PowerflowParameters(
        n_bus,
        n_branch,
        Yf,
        torch.from_numpy(Yff),
        torch.from_numpy(Yft),
        Yt,
        torch.from_numpy(Ytf),
        torch.from_numpy(Ytt),
        Cf,
        Ct,
        Ybus_sh,
        cost_coeff,
        gen_matrix,
        load_matrix,
        fr_bus,
        to_bus,
        base_kv,
        vm_min,
        vm_max,
        Sg_min,
        Sg_max,
        vad_min,
        vad_max,
        rate_a,
    )


def powerflow(
    V: torch.Tensor, S: torch.Tensor, Sd: torch.Tensor, params: PowerflowParameters
) -> PowerflowVariables:
    """
    Find the branch variables given the bus voltages. The inputs and outputs should both be
    in the per unit system.

    Reference:
        https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
        https://matpower.org/docs/MATPOWER-manual.pdf
    """
    Sg = S + Sd
    If = V @ params.Yf.T  # Current from
    It = V @ params.Yt.T  # Current to
    Sf = (V @ params.Cf.T) * If.conj()  # [Cf V] If* = Power from branch
    St = (V @ params.Ct.T) * It.conj()  # [Cf V] It* = Power to branch
    Sbus_sh = (
        V * params.Ybus_sh[None].conj() * V.conj()
    )  # [V][Ybus_shunt*] V* = shunt bus power
    Sbus_branch = Sf @ params.Cf + St @ params.Ct
    Sbus = Sbus_branch + Sbus_sh
    return PowerflowVariables(V, S, Sd, Sg, If, It, Sf, St, Sbus)
