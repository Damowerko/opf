import torch
from dataclasses import dataclass, asdict
import numpy as np
from typing import Optional, Callable, Dict


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
    Yf: torch.Tensor
    Yt: torch.Tensor
    Cf: torch.Tensor
    Ct: torch.Tensor
    Ybus_sh: torch.Tensor
    cost_coeff: torch.Tensor
    gen_matrix: torch.Tensor
    load_matrix: torch.Tensor
    constraints: Dict[str, Constraint]
    rate_a: torch.Tensor
    fr_bus: torch.Tensor
    to_bus: torch.Tensor
    base_kv: torch.Tensor

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
        self.Yt = fn(self.Yt)
        self.Cf = fn(self.Cf)
        self.Ct = fn(self.Ct)
        self.Ybus_sh = fn(self.Ybus_sh)
        self.cost_coeff = fn(self.cost_coeff)
        self.gen_matrix = fn(self.gen_matrix)
        self.load_matrix = fn(self.load_matrix)
        for constraint in self.constraints.values():
            constraint._apply(fn)


@dataclass(eq=False, repr=False)
class InequalityConstraint(Constraint):
    variable: Callable[[PowerflowParameters, PowerflowVariables], torch.Tensor]
    min: Callable[[PowerflowParameters, PowerflowVariables], torch.Tensor]
    max: Callable[[PowerflowParameters, PowerflowVariables], torch.Tensor]

    def _apply(self, fn):
        super()._apply(fn)
        if self.min is not None:
            self.min = fn(self.min)
        if self.max is not None:
            self.max = fn(self.max)


@dataclass(eq=False, repr=False)
class EqualityConstraint(Constraint):
    value: Callable[[PowerflowParameters, PowerflowVariables], torch.Tensor]
    target: Callable[[PowerflowParameters, PowerflowVariables], torch.Tensor]


def parameters_from_pm(pm) -> PowerflowParameters:
    constraints = {}

    # init bus
    n_bus = len(pm["bus"])
    vm_min = torch.zeros((n_bus, 1))
    vm_max = torch.zeros((n_bus, 1))
    base_kv = torch.zeros((n_bus, 1))
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
    p_min = torch.zeros((n_bus, 1))
    q_min = torch.zeros((n_bus, 1))
    p_max = torch.zeros((n_bus, 1))
    q_max = torch.zeros((n_bus, 1))
    gen_matrix = torch.zeros((n_gen, n_bus))
    cost_coeff = torch.zeros((n_bus, 3))

    for gen in pm["gen"].values():
        i = gen["gen_bus"] - 1
        p_min[i] = gen["pmin"]
        q_min[i] = gen["qmin"]
        p_max[i] = gen["pmax"]
        q_max[i] = gen["qmax"]
        assert gen["model"] == 2  # cost is polynomial
        assert len(gen["cost"]) == 3  # only real cost
        cost_coeff[i, :] = torch.as_tensor(
            gen["cost"][::-1]
        )  # Cost is polynomial c0 x^2 + c1x
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
    rate_a = torch.full((n_branch, 1), float("inf"))
    vad_max = torch.full((n_branch, 1), float("inf"))
    vad_min = torch.full((n_branch, 1), -float("inf"))

    Yff = np.zeros((n_branch,), dtype=np.cdouble)
    Yft = np.zeros((n_branch,), dtype=np.cdouble)
    Ytf = np.zeros((n_branch,), dtype=np.cdouble)
    Ytt = np.zeros((n_branch,), dtype=np.cdouble)
    Cf = np.zeros((n_branch, n_bus), dtype=np.cdouble)
    Ct = np.zeros((n_branch, n_bus), dtype=np.cdouble)
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
    Ybus_sh = np.zeros((n_bus, 1), dtype=np.cdouble)
    for shunt in pm["shunt"].values():
        i = shunt["shunt_bus"] - 1
        Ybus_sh[i] += shunt["gs"] + 1j * shunt["bs"]
    Ybus_sh = torch.from_numpy(Ybus_sh)

    # init constraints
    constraints = {
        "equality/bus_power": EqualityConstraint(
            True, False, lambda p, d: d.Sbus, lambda p, d: d.S
        ),
        "inequality/voltage_magnitude": InequalityConstraint(
            True, True, lambda p, d: d.V.abs(), vm_min, vm_max
        ),
        "inequality/active_power": InequalityConstraint(
            True, False, lambda p, d: d.Sg.real, p_min, p_max
        ),
        "inequality/reactive_power": InequalityConstraint(
            True, False, lambda p, d: d.Sg.imag, q_min, q_max
        ),
        "inequality/forward_rate": InequalityConstraint(
            False, False, lambda p, d: d.Sf.abs(), torch.zeros_like(rate_a), rate_a
        ),
        "inequality/backward_rate": InequalityConstraint(
            False, False, lambda p, d: d.St.abs(), torch.zeros_like(rate_a), rate_a
        ),
        "inequality/voltage_angle_difference": InequalityConstraint(
            False,
            True,
            lambda p, d: ((p.Cf @ d.V) * (p.Ct @ d.V).conj()).angle(),
            vad_min,
            vad_max,
        ),
    }

    return PowerflowParameters(
        n_bus,
        n_branch,
        Yf,
        Yt,
        Cf,
        Ct,
        Ybus_sh,
        cost_coeff,
        gen_matrix,
        load_matrix,
        constraints,
        rate_a,
        fr_bus,
        to_bus,
        base_kv,
    )


def powerflow(
    V: torch.Tensor, S: torch.Tensor, Sd: torch.Tensor, network: PowerflowParameters
) -> PowerflowVariables:
    """
    Find the branch variables given the bus voltages. The inputs and outputs should both be
    in the per unit system.

    Reference:
        https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/
        https://matpower.org/docs/MATPOWER-manual.pdf
    """
    Sg = S + Sd
    If = network.Yf @ V  # Current from
    It = network.Yt @ V  # Current to
    Sf = (network.Cf @ V) * If.conj()  # [Cf V] If* = Power from branch
    St = (network.Ct @ V) * It.conj()  # [Cf V] It* = Power to branch
    Sbus_sh = (
        V * network.Ybus_sh.unsqueeze(0).conj() * V.conj()
    )  # [V][Ybus_shunt*] V* = shunt bus power
    Sbus_branch = network.Cf.T @ Sf + network.Ct.T @ St
    Sbus = Sbus_branch + Sbus_sh
    return PowerflowVariables(V, S, Sd, Sg, If, It, Sf, St, Sbus)
