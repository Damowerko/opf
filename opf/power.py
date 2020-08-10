import glob
import warnings
import math
import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology
import pandas as pd
import pandapower.networks
import os
from importlib import reload
from functools import lru_cache

OPFNotConverged = pp.OPFNotConverged


def load_case(case_name, data_dir, reindex=True):
    custom_cases = ['denmark']
    if hasattr(pp.networks, case_name):
        net = getattr(pp.networks, case_name)()
        if case_name == 'iceland':
            net['line']['x_ohm_per_km'][80] = -net['line']['x_ohm_per_km'][80]
    elif case_name in custom_cases:
        net = pp.from_json(os.path.join(data_dir, case_name, "train.json"))
    else:
        raise ValueError("Network name {} is undefined.".format(case_name))
    if not os.path.exists(os.path.join(data_dir, case_name)):
        os.mkdir(os.path.join(data_dir, case_name))
    if reindex:
        pp.create_continuous_bus_index(net, start=0)
    return net


def adjacency_from_net(net, scaling=0.001, threshold=0.01):
    nxgraph = pp.topology.create_nxgraph(net, calc_branch_impedances=True, multi=False)
    index_attribute = "index"
    nxgraph = nx.convert_node_labels_to_integers(nxgraph, label_attribute=index_attribute)
    N = len(nxgraph.nodes)
    G = np.zeros((N, N))
    for u, v, data in nxgraph.edges(data=True):
        value = math.exp(-data['z_ohm'] * scaling)
        if value > threshold:
            G[u, v] = value
            G[v, u] = value
    indicies = [index for _, index in nxgraph.nodes(data=index_attribute)]

    nxg = nx.from_numpy_array(G)
    num_connected_components = len(list(nx.connected_components(nxg)))
    if num_connected_components > 1:
        warnings.warn("The graph has more than 1 connected component ({}). Try decreasing scaling and/or threshold "
                      "parameters.".format(num_connected_components))
    return G, indicies


class NetworkManager:
    def __init__(self, net, A_scaling=0.001, A_threshold=0.01):
        self.net = net
        self.A_scaling = A_scaling
        self.A_threshold = A_threshold
        self._adjacency, self._bus_indices = adjacency_from_net(self.net, self.A_scaling, self.A_threshold)
        self.gen_indices = self._gen_indices()
        self.sgen_indices = self._sgen_indices()
        self.gen_default = self.get_gen()

        assert len(set(self.gen_indices)) == len(self.gen_indices)
        assert len(set(self.sgen_indices)) == len(self.sgen_indices)

    def _gen_indices(self):
        return self.net.gen.index.to_numpy()

    def _sgen_indices(self):
        return self.net.sgen.index.to_numpy()

    def get_adjacency(self):
        return np.copy(self._adjacency)

    @property
    def n_buses(self):
        return self._adjacency.shape[0]

    def get_bus_index(self, i):
        return self._bus_indices[i]

    def set_gen(self, g: np.ndarray):
        self.net["gen"]["p_mw"] = g[self.gen_indices]
        self.net["sgen"]["p_mw"] = g[self.sgen_indices]

    def reset_gen(self):
        self.set_gen(self.gen_default)

    def get_gen(self):
        g = np.zeros((self.n_buses, 2))
        g[:, 0].put(self.gen_indices, self.net["gen"]["p_mw"].to_numpy())
        g[:, 1].put(self.sgen_indices, self.net["sgen"]["p_mw"].to_numpy())
        return g

    def get_gen_index(self):
        gen_index = self.net['gen']['bus'].to_numpy()
        sgen_index = self.net['sgen']['bus'].to_numpy()
        return np.concatenate((gen_index, sgen_index))

    def get_gen_len(self) -> int:
        return len(self.net["gen"]) + len(self.net["sgen"])

    def set_load(self, l):
        l = np.transpose(l)
        self.net["load"]["p_mw"] = l[0]
        self.net["load"]["q_mvar"] = l[1]

    def get_load(self, reactive=False):
        p = self.net["load"]["p_mw"].to_numpy()
        q = self.net["load"]["q_mvar"].to_numpy()
        if reactive:
            return np.stack((p, q), axis=1)
        return p

    def cost(self):
        _, g = self.powerflow()
        if g is None:
            return None
        cost = 0
        for _, row in self.net["poly_cost"].iterrows():
            index = int(row["element"])
            p = self.net["res_" + row["et"]]["p_mw"][index]
            q = self.net["res_" + row["et"]]["q_mvar"][index]
            cost += row["cp0_eur"] + p * row["cp1_eur_per_mw"] + p * p * row["cp2_eur_per_mw2"] \
                    + row["cq0_eur"] + q * row["cq1_eur_per_mvar"] + q * q * row["cq2_eur_per_mvar2"]
        return cost

    def to_vector(self, element_type: str, constraint_type: str):
        element: pd.DataFrame = self.net[element_type]
        if constraint_type not in element:
            return np.zeros((self.n_buses, 1))
        series: pd.Series = element[constraint_type].reindex(pd.RangeIndex(self.n_buses)).fillna(0)
        return series.to_numpy().reshape(self.n_buses, 1)

    def get_constrains(self):
        element_types = ['gen', 'sgen', 'ext_grid', 'load', 'storage']
        constraint_types = ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']
        constraints = []
        for element_type in element_types:
            for constraint_type in constraint_types:
                constraints.append(self.to_vector(element_type, constraint_type))

        dc_line_contrant_types = ["max_p_mw", "min_q_from_mvar", "max_q_from_mvar", "min_q_to_mvar", "max_q_to_mvar"]
        for constraint_type in dc_line_contrant_types:
            constraints.append(self.to_vector('dcline', constraint_type))

        return np.concatenate(constraints, 1)

    def count_violations(self):
        """ :return True if there is at least one constraint violated.
        Does not take into account branch constraintrs.
        """

        def count(element, column):
            if self.net[element].shape[0] == 0:
                return 0
            val = self.net['res_' + element].sort_index()[column].to_numpy()
            max = self.net[element].sort_index()['max_' + column].to_numpy()
            min = self.net[element].sort_index()['min_' + column].to_numpy()
            return np.sum(val > max) + np.sum(val < min)

        bus = count('bus', 'vm_pu')
        gen = count('gen', 'p_mw') + count('gen', 'q_mvar')
        sgen = count('sgen', 'p_mw') + count('sgen', 'q_mvar')
        ext_grid = count('ext_grid', 'p_mw') + count('ext_grid', 'q_mvar')
        line = len(pp.overloaded_lines(self.net, 100))
        return bus + gen + sgen + ext_grid + line

    def count_contraints(self):
        def count(element):
            return self.net[element].shape[0] * 2

        return count('bus') + count('gen') + count('sgen') + count('ext_grid') + count('line')

    def powerflow(self):
        try:
            pp.runpp(self.net)
            return self._results()
        except pp.LoadflowNotConverged:
            pp.diagnostic(self.net)
            return None, None

    def optimal_dc(self):
        pp.rundcopp(self.net)
        if not self.opf_converged():
            raise pp.OPFNotConverged
        return self._results()

    def optimal_ac(self):
        try:
            try:
                pp.runopp(self.net, calculate_voltage_angles=True)
                # pp.runpm_ac_opf(self.net)
            except RuntimeError:
                reload(pandapower)
                pp.runopp(self.net, calculate_voltage_angles=True)
                # pp.runpm_ac_opf(self.net)
        except (RuntimeError, TypeError, NameError):
            raise pp.OPFNotConverged
        if not self.opf_converged():
            raise pp.OPFNotConverged
        return self._results()

    def opf_converged(self):
        return self.net['OPF_converged']

    def _results(self):
        b = self.net["res_bus"].to_numpy()
        g = np.concatenate((self.net["res_gen"]["p_mw"].to_numpy(), self.net["res_sgen"]["p_mw"].to_numpy()))
        return b, g


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
    def generate_load_from_random(average: np.ndarray, num_samples: int, delta: float = 0.1) -> pd.DataFrame:
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
        data: np.ndarray = pd.concat([self.commercial_data, self.residential_data], axis=1).to_numpy()
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
        residential *= average * (1 - portion_commercial) / np.linalg.norm(residential, 1)
        return np.vstack((commercial, residential))

    @staticmethod
    def parse_data(input_dir, state):
        commercial_paths = glob.glob("{}/commercial/USA_{}*/*.csv".format(input_dir, state))
        residential_paths = glob.glob("{}/residential/*/USA_{}*.csv".format(input_dir, state))

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
