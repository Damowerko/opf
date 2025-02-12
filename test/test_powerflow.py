import json
from itertools import product
from pathlib import Path

import h5py
import pytest
import torch

import opf.powerflow as pf
from opf.dataset import build_graph


@pytest.mark.parametrize(
    "from_graph, execution_number", product([True, False], range(10))
)
def test_powerflow(from_graph, execution_number):
    eps = 1e-4
    precision: int = 32
    if precision == 32:
        dtype = torch.float32
        cdtype = torch.complex64
    elif precision == 64:
        dtype = torch.float64
        cdtype = torch.complex128

    if execution_number < 5:
        case_path = Path(__file__).parent / "case30_ieee.json"
        data_path = Path(__file__).parent / "case30_ieee.h5"
        idx = execution_number
    else:
        case_path = Path(__file__).parent / "case118_ieee.json"
        data_path = Path(__file__).parent / "case118_ieee.h5"
        idx = execution_number - 5

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

    if from_graph:
        graph = build_graph(parameters)
        constraints = pf.build_constraints(
            variables,
            graph,
            is_dual=False,
        )
    else:
        bus_parameters = pf.BusParameters.from_pf_parameters(parameters)
        gen_parameters = pf.GenParameters.from_pf_parameters(parameters)
        branch_parameters = pf.BranchParameters.from_pf_parameters(parameters)
        constraints = pf._build_constraints(
            variables,
            bus_parameters,
            branch_parameters,
            gen_parameters,
            parameters.fr_bus,
            parameters.to_bus,
        )
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
