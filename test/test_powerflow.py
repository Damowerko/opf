import json
import random
from pathlib import Path

import pytest
import torch

import opf.powerflow as pf

case_path = Path(__file__).parent / "case30_ieee.json"
solutions_path = Path(__file__).parent / "case30_ieee.train.json"


@pytest.mark.parametrize("execution_number", range(10))
def test_powerflow(execution_number):
    # set the precision to 32 bit floats
    precision = 32
    dtype = torch.complex64
    eps = 1e-4

    # load network
    dataset = json.loads(solutions_path.read_text())

    parameters = pf.parameters_from_powermodels(
        json.loads(case_path.read_text()), precision
    )
    # get a random sample
    data = dataset[execution_number]
    solution = data["result"]["solution"]

    V, S, Sd = pf.power_from_solution(data["load"], solution, parameters)
    V = V.to(dtype)
    S = S.to(dtype)
    Sd = Sd.to(dtype)
    variables = pf.powerflow(V, S, Sd, parameters)

    # branch current
    Sf = torch.complex(
        *pf.powermodels_to_tensor(solution["branch"], ["pf", "qf"]).T
    ).to(dtype)
    St = torch.complex(
        *pf.powermodels_to_tensor(solution["branch"], ["pt", "qt"]).T
    ).to(dtype)
    assert torch.allclose(Sf, variables.Sf, atol=eps)
    assert torch.allclose(St, variables.St, atol=eps)

    constraints = pf.build_constraints(variables, parameters)
    for constraint_name, constraint in constraints.items():
        if isinstance(constraint, pf.InequalityConstraint):
            assert (
                constraint.variable - constraint.min >= -eps
            ).all(), f"Failed {constraint_name}"
            assert (
                constraint.max - constraint.variable >= -eps
            ).all(), f"Failed {constraint_name}"
        elif isinstance(constraint, pf.EqualityConstraint):
            difference = (constraint.value - constraint.target).abs()
            assert (difference < eps).all(), f"Failed {constraint_name}"
