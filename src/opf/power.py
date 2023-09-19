import glob
from copy import deepcopy

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks
import pandapower.topology
import pandas as pd
from pandapower.converter.matpower.to_mpc import _ppc2mpc
from pandapower.converter.powermodels.to_pm import convert_pp_to_pm
from pandapower.converter.pypower.to_ppc import to_ppc


def load_case(case_name, reindex=True):
    if hasattr(pp.networks, case_name):
        net = getattr(pp.networks, case_name)()
    else:
        raise ValueError("Network name {} is undefined.".format(case_name))
    if reindex:
        pp.create_continuous_elements_index(net, start=0)
    return net


def simplify_net(net):
    """
    Simplifies the net:
        * one generator per bus
        * replaces static generators (PQ) with generators (PV)

    It will move generators onto new buses which are connected to the original node with a low impedence line.

    TODO: Consider merging generators together instead of adding new buses.

    :param net: The pandapower network.
    :return: The modified pandapower network.
    """
    net = deepcopy(net)
    pp.replace_gen_by_sgen(net)

    gen_bus = set()  # keep track of the set of all buses
    # external grid should be considered a generator
    assert len(net.ext_grid == 1)
    gen_bus.add(net.ext_grid.iloc[0].bus)

    for gen_index, row in net.sgen.iterrows():
        bus_index = row["bus"]
        # check if bus already exists
        if bus_index in gen_bus:
            # Create new bus with unique index
            new_bus_index = net.bus.shape[0]
            bus_row = net.bus.loc[bus_index]
            bus_row["name"] = new_bus_index
            net.bus = net.bus.append(bus_row, ignore_index=True)
            # Change the generator bus
            net.sgen.at[gen_index, "bus"] = new_bus_index
            # Connect original bus with low impedence line
            pp.create_line_from_parameters(
                net,
                bus_index,
                new_bus_index,
                0.01,
                1,
                1,
                0.01,
                1e8,
                r0_ohm_per_km=1,
                x0_ohm_per_km=1,
                c0_nf_per_km=1,
                max_loading_percent=100,
            )
        gen_bus.add(bus_index)
    return net


class NetWrapper:
    def __init__(self, net, per_unit=True):
        """
        Wrapper around a PandaPower network.
        :param net: PandaPower network.
        :param per_unit: Should the input/output values be in the per unit system?
        """
        self.net = simplify_net(net)
        self.per_unit = per_unit

        self.bus_indices = self.net.bus.index.to_numpy()
        self.gen_indices = self.net.sgen.bus.to_numpy()
        self.load_indices = self.net.load.bus.to_numpy()
        self.ext_indices = self.net.ext_grid.bus.to_numpy()
        self.shunt_indices = self.net.shunt.bus.to_numpy()

        self.base_mva = self.net.sn_mva
        self.base_kv = self.net.bus["vn_kv"].to_numpy()
        self.base_ka = self.base_mva / self.base_kv
        self.base_ohm = self.base_kv / self.base_ka
        assert len(set(self.gen_indices)) == len(self.gen_indices)

    def impedence_matrix(self):
        """
        Creates a matrix of the branch impedance values from a PandaPower network.
        Includes all branch elements.

        :param net: Pandapower network.
        :param per_unit: If true the output will be in per unit else in ohms.
        :param
        """
        unit = "pu" if self.per_unit else "ohm"
        graph = pp.topology.create_nxgraph(
            self.net,
            calc_branch_impedances=True,
            multi=False,
            branch_impedance_unit=unit,
        )
        # Check that node indices matrch the network bus indices.
        assert set(graph.nodes) == set(self.bus_indices)
        return nx.linalg.graphmatrix.adjacency_matrix(
            graph, weight=f"z_{unit}", nodelist=self.bus_indices
        )

    @property
    def n_buses(self):
        return self.net.bus.shape[0]

    def set_gen_sparse(self, p_gen: np.ndarray, q_gen: np.ndarray):
        p_gen = p_gen[self.gen_indices]
        q_gen = q_gen[self.gen_indices]
        self.set_gen(p_gen, q_gen)

    def set_gen(self, p_gen: np.ndarray, q_gen: np.ndarray):
        self.net["sgen"]["p_mw"] = p_gen * self.base_mva if self.per_unit else p_gen
        self.net["sgen"]["q_mvar"] = q_gen * self.base_mva if self.per_unit else q_gen

    def get_gen(self):
        p_gen = self.net["sgen"]["p_mw"].to_numpy()
        q_gen = self.net["sgen"]["q_mvar"].to_numpy()
        if self.per_unit:
            p_gen /= self.base_mva
        return p_gen, q_gen

    def set_load_sparse(self, p, q):
        self.set_load(p[self.load_indices], q[self.load_indices])

    def set_load(self, p, q):
        self.net["load"]["p_mw"] = p * self.base_mva if self.per_unit else p
        self.net["load"]["q_mvar"] = q * self.base_mva if self.per_unit else q

    def get_load(self):
        p_load = self.net["load"]["p_mw"].to_numpy()
        q_load = self.net["load"]["q_mvar"].to_numpy()
        if self.per_unit:
            p_load /= self.base_mva
            q_load /= self.base_mva
        return p_load, q_load

    def powerflow(self):
        try:
            pp.runpp(self.net, calculate_voltage_angles=True, trafo_model="pi")  # type: ignore
            return self._results()
        except pp.LoadflowNotConverged:
            pp.diagnostic(net=self.net, report_style="detailed")
            return None

    def optimal_ac(self, powermodels=True):
        """
        Run optimal power flow.
        :return: A tuple of bus, generator, and external grid results as numpy arrays. Each column represents a
        different value. Returns None if not converged.
            Bus columns: [|V| angle(V) Re(S) Im(S)].
            Generator columns: [Re(S) Im(S)]
            External grid columns: [Re(S) Im(S)]
        """
        try:
            if powermodels:
                pp.runpm_ac_opf(
                    self.net, calculate_voltage_angles=True, trafo_model="pi"
                )
            else:
                pp.runopp(self.net, calculate_voltage_angles=True)
        except pp.OPFNotConverged:
            return None
        return self._results()

    def optimal_dc(self):
        try:
            pp.rundcopp(self.net)
        except pp.OPFNotConverged:
            return None
        return self._results()

    def _opf_converged(self):
        return self.net["OPF_converged"]

    def _results(self):
        bus = self.net["res_bus"]
        gen = self.net["res_sgen"]
        ext = self.net["res_ext_grid"]
        sh = self.net["res_shunt"]

        vm = bus["vm_pu"].to_numpy()
        va = bus["va_degree"].to_numpy() * np.pi / 180
        # Note the negative sign. We are converting from PandaPower convention to MatPower/PowerModels.jl.
        p_bus = -bus["p_mw"].to_numpy()
        q_bus = -bus["q_mvar"].to_numpy()
        p_gen = gen["p_mw"].to_numpy()
        q_gen = gen["q_mvar"].to_numpy()
        p_ext = ext["p_mw"].to_numpy()
        q_ext = ext["q_mvar"].to_numpy()

        p_bus[self.shunt_indices] += sh["p_mw"].to_numpy()
        q_bus[self.shunt_indices] += sh["q_mvar"].to_numpy()

        if self.per_unit:
            p_bus /= self.base_mva
            q_bus /= self.base_mva
            p_gen /= self.base_mva
            q_gen /= self.base_mva
            p_ext /= self.base_mva
            q_ext /= self.base_mva

        bus = np.stack((vm, va, p_bus, q_bus), axis=1).T
        gen = np.stack((p_gen, q_gen), axis=1).T
        ext = np.stack((p_ext, q_ext), axis=1).T
        return bus, gen, ext

    def to_powermodels(self):
        return convert_pp_to_pm(self.net, trafo_model="pi")

    def to_pypower(self):
        return to_ppc(self.net, trafo_model="pi")

    def to_matpower(self):
        ppc = self.to_pypower()
        return _ppc2mpc(ppc)

    def cost_coefficients(self, element_type):
        assert element_type in ["gen", "sgen", "ext_grid", "load", "dcline", "storage"]
        assert len(self.net.pwl_cost.index) == 0

        poly: pd.DataFrame = self.net.poly_cost
        poly = poly.loc[poly["et"] == element_type].astype({"element": int})
        elements: pd.DataFrame = self.net[element_type]
        joined = elements.merge(poly, left_index=True, right_on="element")

        p_coeff = np.zeros((3, self.n_buses))
        q_coeff = np.zeros((3, self.n_buses))
        indices = joined["bus"].to_numpy().flatten()

        p_coeff[:, indices] = (
            joined[["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]].to_numpy().T
        )
        q_coeff[:, indices] = (
            joined[["cq0_eur", "cq1_eur_per_mvar", "cq2_eur_per_mvar2"]].to_numpy().T
        )
        return p_coeff, q_coeff

    def cost(self):
        res_powerflow = self.powerflow()
        if res_powerflow is None:
            return None
        _, g = res_powerflow
        cost = 0
        for _, row in self.net["poly_cost"].iterrows():
            index = int(row["element"])
            p = self.net["res_" + row["et"]]["p_mw"][index]
            q = self.net["res_" + row["et"]]["q_mvar"][index]
            cost += (
                row["cp0_eur"]
                + p * row["cp1_eur_per_mw"]
                + p * p * row["cp2_eur_per_mw2"]
                + row["cq0_eur"]
                + q * row["cq1_eur_per_mvar"]
                + q * q * row["cq2_eur_per_mvar2"]
            )
        return cost

    def to_vector(self, element_type: str, constraint_type: str):
        element: pd.DataFrame = self.net[element_type]
        if constraint_type not in element:
            return np.zeros((self.n_buses, 1))
        series: pd.Series = (
            element[constraint_type].reindex(pd.RangeIndex(self.n_buses)).fillna(0)
        )
        return series.to_numpy().reshape(self.n_buses, 1)

    def get_constrains(self):
        element_types = ["gen", "sgen", "ext_grid", "load", "storage"]
        constraint_types = ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]
        constraints = []
        for element_type in element_types:
            for constraint_type in constraint_types:
                constraints.append(self.to_vector(element_type, constraint_type))

        dc_line_contrant_types = [
            "max_p_mw",
            "min_q_from_mvar",
            "max_q_from_mvar",
            "min_q_to_mvar",
            "max_q_to_mvar",
        ]
        for constraint_type in dc_line_contrant_types:
            constraints.append(self.to_vector("dcline", constraint_type))

        return np.concatenate(constraints, 1)

    def count_violations(self):
        """:return True if there is at least one constraint violated.
        Does not take into account branch constraintrs.
        """

        def count(element, column):
            if self.net[element].shape[0] == 0:
                return 0
            val = self.net["res_" + element].sort_index()[column].to_numpy()
            max = self.net[element].sort_index()["max_" + column].to_numpy()
            min = self.net[element].sort_index()["min_" + column].to_numpy()
            return np.sum(val > max) + np.sum(val < min)

        bus = count("bus", "vm_pu")
        gen = count("gen", "p_mw") + count("gen", "q_mvar")
        sgen = count("sgen", "p_mw") + count("sgen", "q_mvar")
        ext_grid = count("ext_grid", "p_mw") + count("ext_grid", "q_mvar")
        line = len(pp.overloaded_lines(self.net, 100))
        return bus + gen + sgen + ext_grid + line

    def count_contraints(self):
        def count(element):
            return self.net[element].shape[0] * 2

        return (
            count("bus")
            + count("gen")
            + count("sgen")
            + count("ext_grid")
            + count("line")
        )


class LoadGenerator:
    def __init__(self, input_dir):
        self.commercial_data = pd.read_pickle(input_dir + "commercial.pickle")
        self.residential_data = pd.read_pickle(input_dir + "residential.pickle")

    def size(self):
        """
        :return: A pair of integers. The number of source commercial templates and the number of source residential
        templates.
        """
        return len(self.commercial_data.columns), len(self.residential_data.columns)

    @staticmethod
    def generate_load_from_random(
        average: np.ndarray, num_samples: int, delta: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate loads by sampling from the uniform distribution on [average*(1-delta),average*(1+delta)].
        :param average: A vector with average load for each node.
        :param num_samples: The number of samples to generate.
        :return: A series of hourly power demanded in kW. num_samples x N, where N is the number of nodes.
        """
        N = average.size
        average = average.reshape((1, average.size))
        average = np.tile(average, (num_samples, 1))
        low = average * (1 - delta)
        high = average * (1 + delta)
        random = np.random.uniform(low, high, (num_samples, N))
        return pd.DataFrame(random)

    def generate_load_from_profiles(self, composition: np.ndarray) -> pd.DataFrame:
        """
        Generate a series representing the load in kW based on composition. Linearly combines several sample load
        profiles in order to create unique profiles on the energy grid.
        :param composition: A matrix representing how the linear combination should be performed. Should be of size,
        size()[0] + size()[1] by K, where K is the number of series to generate.
        :return: A series of hourly power demanded in kW.
        """
        data: np.ndarray = pd.concat(
            [self.commercial_data, self.residential_data], axis=1
        ).to_numpy()
        composition: np.ndarray = np.asarray(composition)
        return data @ composition

    def random_composition(self, average, portion_commercial):
        """
        :param average: A vector with average load for each node.
        :param portion_commercial: Vector ofHow much of the load should be commercial. This vector should be same
        length as average.
        :return: The composition matrix.
        """
        if isinstance(portion_commercial, float):
            portion_commercial = np.full(np.size(average), portion_commercial)
        commercial = np.random.rand(self.size()[0], average.size)
        residential = np.random.rand(self.size()[1], average.size)
        commercial *= average * portion_commercial / np.linalg.norm(commercial, 1)
        residential *= (
            average * (1 - portion_commercial) / np.linalg.norm(residential, 1)
        )
        return np.vstack((commercial, residential))

    @staticmethod
    def parse_data(input_dir, state):
        commercial_paths = glob.glob(
            "{}/commercial/USA_{}*/*.csv".format(input_dir, state)
        )
        residential_paths = glob.glob(
            "{}/residential/*/USA_{}*.csv".format(input_dir, state)
        )

        def aggregate_electricity(path):
            return pd.read_csv(path).filter(regex="Electricity", axis=1).sum(axis=1)

        commercial_data: list = []
        residential_data: list = []
        for path in commercial_paths:
            commercial_data.append(aggregate_electricity(path))
        for path in residential_paths:
            residential_data.append(aggregate_electricity(path))
        commercial_data: pd.DataFrame = pd.concat(commercial_data, axis=1)
        residential_data: pd.DataFrame = pd.concat(residential_data, axis=1)

        # normalizes ST the series average is 1
        def normalize(series: pd.Series):
            return series / series.mean()

        commercial_data: pd.DataFrame = commercial_data.apply(normalize)
        residential_data: pd.DataFrame = residential_data.apply(normalize)

        commercial_data.to_pickle(input_dir + "commercial.pickle")
        residential_data.to_pickle(input_dir + "residential.pickle")
