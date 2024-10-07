import json
from pathlib import Path

import h5py
import pytest
import torch

import opf.powerflow as pf

case_path = Path(__file__).parent / "case30_ieee.json"
data_path = Path(__file__).parent / "case30_ieee.h5"


@pytest.mark.parametrize("execution_number", range(10))
def test_powerflow(execution_number):
    # set the precision to 32 bit floats
    precision: int = 32
    dtype = torch.complex64
    eps = 1e-4

    parameters = pf.parameters_from_powermodels(
        json.loads(case_path.read_text()), case_path.as_posix(), precision
    )
    # get a random sample
    with h5py.File(data_path, "r") as f:
        bus = torch.tensor(f["bus"][execution_number], dtype=torch.float32)  # type: ignore
        load = torch.tensor(f["load"][execution_number], dtype=torch.float32)  # type: ignore
        gen = torch.tensor(f["gen"][execution_number], dtype=torch.float32)  # type: ignore
        branch = torch.tensor(f["branch"][execution_number], dtype=torch.float32)  # type: ignore

    V = torch.polar(*bus.T).to(dtype)
    Sg = torch.complex(*gen.T).to(dtype)
    Sd = torch.zeros_like(V).index_put_(
        (parameters.load_bus_ids,), torch.complex(*load.T)
    )
    # add batch dimensions
    V = V.unsqueeze(0)
    Sg = Sg.unsqueeze(0)
    Sd = Sd.unsqueeze(0)

    variables = pf.powerflow(V, Sd, Sg, parameters)

    Sf = torch.complex(*branch[:, :2].T).to(dtype)[None, :]
    St = torch.complex(*branch[:, 2:].T).to(dtype)[None, :]

    torch.testing.assert_close(Sf, variables.Sf, atol=eps, rtol=eps)
    torch.testing.assert_close(St, variables.St, atol=eps, rtol=eps)

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
