from dataclasses import asdict, dataclass
from typing import Callable, Dict

import numpy as np
import torch

@dataclass
class PowerFlowVariables():
    V: torch.Tensor
    # S is target
    S: torch.Tensor
    Sg: torch.Tensor
    Sd: torch.Tensor
    Sf: torch.Tensor
    St: torch.Tensor
    # Sbus is value
    Sbus: torch.Tensor
    theta: torch.Tensor

@dataclass
class PowerFlowParameters():
    # indexing
    n_bus: int
    n_branch: int
    fr_bus: int
    to_bus: int
    # power and voltage params
    Sg_min: torch.Tensor
    Sg_max: torch.Tensor
    v_min: torch.Tensor
    v_max: torch.Tensor
    theta_min: torch.Tensor
    theta_max: torch.Tensor
    # cost
    cost_coeff: torch.Tensor
    # admittance
    Y: torch.Tensor
    Yc_fr: torch.Tensor
    Yc_to: torch.Tensor
    Ybus_sh: torch.Tensor
    # shortened
    ff_max: torch.Tensor
    ff_min: torch.Tensor
    ft_min: torch.Tensor
    ft_min: torch.Tensor
    # is everything REALLY a tensor

    # WHAT THE HELL IS THIS?????
    Cf: torch.Tensor
    Ct: torch.Tensor


@dataclass
class Constraint():
    isBus: bool
    isBranch: bool

@dataclass
class EqualityConstraint(Constraint):
    value: torch.Tensor
    target: torch.Tensor

@dataclass
class InequalityConstraint(Constraint):
    value: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor

def build_constraints(var: PowerFlowVariables, param: PowerFlowParameters):
    """
    EqualityConstraint(isBus, isBranch, value, target)
    InequalityConstraint(isBus, isBranch, value, lower_bound, upper_bound)
    """

    return {
        "equality/bus_power": EqualityConstraint(True, False, var.Sbus, var.S),
        "inequality/Sg": InequalityConstraint(True, False, var.Sg, param.Sg_min, param.Sg_max),
        "inequality/V": InequalityConstraint(True, False, var.V, param.v_min, param.v_max),
        "inequality/Sij": InequalityConstraint(False, True, var.Sf, param.ff_min, param.ff_max),
        "inequality/Sji": InequalityConstraint(False, True, var.St, param.ft_min, param.ft_max),
        "inequality/theta": InequalityConstraint(False, True, var.theta, param.theta_min, param.theta_max)
    }

# # STOLEN!!!
# class PowerflowData(typing.NamedTuple):
#     data: Data | HeteroData
#     powerflow_parameters: PowerFlowParameters

# # STOLEN!!!
# class PowerflowBatch(typing.NamedTuple):
#     data: Batch
#     powerflow_parameters: PowerFlowParameters

def powerflow(V, Sg, Sd, params: PowerFlowParameters) -> PowerFlowVariables:
    Vf = V[..., params.fr_bus]
    Vt = V[..., params.to_bus]
    Sf = (params.Y + params.Yc_fr).conj() * Vf.abs() ** 2 / params.ratio.abs() ** 2 - (
        params.Y.conj() * Vf * Vt.conj() / params.ratio
    )
    St = (params.Y + params.Yc_to).conj() * Vt.abs() ** 2 - (
        params.Y.conj() * Vf.conj() * Vt / params.ratio.conj()
    )

    Sbus_sh = params.Ybus_sh.conj() * V.abs() ** 2
    # you don't have Cf and Ct defined
    Sbus_branch = Sf @ params.Cf + St @ params.Ct
    Sbus = Sbus_branch + Sbus_sh
    S = Sg - Sd
    return PowerFlowVariables(V, S, Sd, Sg, Sf, St, Sbus)    
