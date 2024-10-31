import json
from pathlib import Path

import h5py
import pytest
import torch

import opf.powerflow as pf


@pytest.mark.parametrize("execution_number", range(20))
def test_powerflow(execution_number):
    eps = 1e-4
    precision: int = 32
    if precision == 32:
        dtype = torch.float32
        cdtype = torch.complex64
    elif precision == 64:
        dtype = torch.float64
        cdtype = torch.complex128

    if execution_number < 10:
        case_path = Path(__file__).parent / "case30_ieee.json"
        data_path = Path(__file__).parent / "case30_ieee.h5"
        idx = execution_number
    else:
        case_path = Path(__file__).parent / "case118_ieee.json"
        data_path = Path(__file__).parent / "case118_ieee.h5"
        idx = execution_number - 10

    parameters = pf.parameters_from_powermodels(
        json.loads(case_path.read_text()), case_path.as_posix(), precision=precision
    )

    # get a random sample
    with h5py.File(data_path, "r") as f:
        bus = torch.tensor(f["bus"][idx], dtype=dtype)  # type: ignore
        load = torch.tensor(f["load"][idx], dtype=dtype)  # type: ignore
        gen = torch.tensor(f["gen"][idx], dtype=dtype)  # type: ignore
        branch = torch.tensor(f["branch"][idx], dtype=dtype)  # type: ignore

    V = torch.polar(*bus.T).to(cdtype)
    Sg = torch.complex(*gen.T).to(cdtype)
    Sd = torch.zeros_like(V).index_put_(
        (parameters.load_bus_ids,), torch.complex(*load.T)
    )
    # add batch dimensions
    V = V.unsqueeze(0)
    Sg = Sg.unsqueeze(0)
    Sd = Sd.unsqueeze(0)

    variables = pf.powerflow(V, Sd, Sg, parameters)

    Sf = torch.complex(*branch[:, :2].T).to(cdtype)[None, :]
    St = torch.complex(*branch[:, 2:].T).to(cdtype)[None, :]

    assert (Sf - variables.Sf).abs().max() < eps
    assert (St - variables.St).abs().max() < eps

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
            if constraint.mask is not None:
                difference = difference[:, constraint.mask]
            assert (difference < eps).all(), f"Failed {constraint_name}"
