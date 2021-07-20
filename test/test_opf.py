from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier, GNN
import torch
import numpy as np
from itertools import islice
import pytest


class Modules:
    def __init__(self):
        param = dict(
            case_name="case30",
            adj_scaling="auto",
            adj_threshold=0.01,
            batch_size=1024,
            max_epochs=100,
            K=8,
            F=16,
            gnn_layers=4,
            MLP=4,
            mlp_layers=1,
            t=10,
            s=500,
            cost_weight=0.1,
            lr=3e-4,
            constraint_features=False,
        )

        dm = CaseDataModule(
            param["case_name"],
            data_dir="../data",
            batch_size=param["batch_size"],
            num_workers=0,
            pin_memory=False,
        )

        gnn = GNN(
            dm.gso(),
            [2] + [param["F"]] * param["gnn_layers"],
            [param["K"]] * param["gnn_layers"],
            [dm.net_wrapper.n_buses * param["MLP"]] * param["mlp_layers"],
        )

        barrier: OPFLogBarrier = OPFLogBarrier(
            dm.net_wrapper,
            gnn,
            t=param["t"],
            s=param["s"],
            cost_weight=param["cost_weight"],
            lr=param["lr"],
            constraint_features=param["constraint_features"],
            eps=1e-3,
        )

        self.barrier: OPFLogBarrier = barrier.double()
        self.net_wrapper = self.barrier.net_wrapper
        self.net = self.net_wrapper.net
        dm.setup("test")
        self.dataloader = dm.test_dataloader()
        self.decimals = int(-np.log10(barrier.eps))


@pytest.fixture
def modules():
    return Modules()


def test_pandapower_reference(modules):
    Sd = np.zeros((2, modules.net_wrapper.n_buses))
    Sg = np.zeros((2, modules.net_wrapper.n_buses))
    (
        Sd[0, modules.net_wrapper.load_indices],
        Sd[1, modules.net_wrapper.load_indices],
    ) = modules.net_wrapper.get_load()
    bus, gen, ext = modules.net_wrapper.powerflow()
    Sg[:, modules.net_wrapper.gen_indices] = gen
    Sg[:, modules.net_wrapper.ext_indices] = ext
    S = bus[2:4, :]
    error = S + Sd - Sg
    assert np.all(error <= 1e-8)


def test_gen(modules):
    for i, (load, acopf_bus) in enumerate(islice(modules.dataloader, 3)):
        load = load.double() @ modules.barrier.load_matrix.T
        modules.barrier.net_wrapper.set_load_sparse(load[0, 0, :], load[0, 1, :])
        bus, gen, ext = modules.barrier.net_wrapper.optimal_ac(powermodels=False)
        bus = torch.from_numpy(bus).unsqueeze(0).double()
        V, S, Sg, Sd = modules.barrier.bus(modules.barrier.bus_from_polar(bus), load)
        gen_torch = torch.view_as_real(
            Sg[0, modules.barrier.net_wrapper.gen_indices, :]
        ).squeeze()
        assert (gen_torch.T - gen).abs().max() < 1e-8


def test_convert_bus(modules):
    for i, (_, original) in enumerate(islice(modules.dataloader, 10)):
        rect = modules.barrier.bus_from_polar(original.double())
        polar = modules.barrier.bus_to_polar(rect.double())
        assert (original - polar).abs().max() < 1e-8


def test_project_pandapower(modules):
    for i, (load, acopf_bus) in enumerate(islice(modules.dataloader, 10)):
        load = load.double() @ modules.barrier.load_matrix.T
        acopf_bus = modules.barrier.bus_from_polar(acopf_bus.double())
        acopf_bus_proj = modules.barrier.project_pandapower(acopf_bus, load)
        assert (acopf_bus - acopf_bus_proj).abs().max() < 1e-5


def test_powerflow(modules):
    for i, (load, acopf_bus) in enumerate(islice(modules.dataloader, 10)):
        load = load.double() @ modules.barrier.load_matrix.T
        acopf_bus = modules.barrier.bus_from_polar(acopf_bus.double())
        V, S, Sg, Sd = modules.barrier.bus(acopf_bus, load)
        If, It, Sf, St, Sbus = modules.barrier.power_flow(V)

        modules.barrier.net_wrapper.set_load_sparse(load[0, 0, :], load[0, 1, :])
        modules.barrier.net_wrapper.set_gen_sparse(Sg.real.squeeze(), Sg.imag.squeeze())
        modules.barrier.net_wrapper.powerflow()
        res_from = modules.barrier.net_wrapper.net.res_line[
            ["p_from_mw", "q_from_mvar"]
        ].to_numpy()
        res_to = modules.barrier.net_wrapper.net.res_line[
            ["p_to_mw", "q_to_mvar"]
        ].to_numpy()

        errors = (
            torch.stack(
                (
                    torch.view_as_real(Sf.squeeze()) - res_from,
                    torch.view_as_real(St.squeeze()) - res_to,
                )
            ).abs()
            / torch.max(Sf.abs().max(), St.abs().max())
        )

        assert errors.max() <= 1e-3
        assert (S - Sbus).abs().max() < 1e-8


def test_acopf_feasible(modules):
    for load, acopf_bus in modules.dataloader:
        load = load.double() @ modules.barrier.load_matrix.T
        acopf_bus = modules.barrier.bus_from_polar(acopf_bus.double())
        _, constraints = modules.barrier.optimal_power_flow(acopf_bus, load)
        for constraint, values in constraints.items():
            assert values["rate"] < 1e-8

def test_cost(modules):
    for load, acopf_bus in islice(modules.dataloader, 10):
        load = load.double() @ modules.barrier.load_matrix.T
        acopf_bus = modules.barrier.bus_from_polar(acopf_bus.double())
        acopf_bus = modules.barrier.project_pandapower(acopf_bus)