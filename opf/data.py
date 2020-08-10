import os
import numpy as np
from multiprocessing import Pool, Value
from functools import partial
import torch
import torch.nn
import torch.nn.functional
import tqdm

from opf.power import NetworkManager, LoadGenerator, load_case, OPFNotConverged, adjacency_from_net
from GNN.Utils.dataTools import _data
import pandapower as pp


class OPFData(_data):
    def __init__(self, data_dir, case_name, ratio_train, ratio_valid, A_scaling=0.001, A_threshold=0.01,
                 dataType=np.float64, device='cpu', use_constraints=True):
        super().__init__()
        self.dataType = dataType
        self.device = device
        #   Dataset partition
        self.ratio_train = ratio_train
        self.ratio_valid = ratio_valid
        self.data_dir = data_dir

        self.net = load_case(case_name, data_dir)
        self.manager = NetworkManager(self.net)

        data = np.load(os.path.join(data_dir, case_name, "data.npz"))
        self.gen = data['gen']
        self.inputs = np.transpose(data['bus'], [0, 2, 1])
        nDataPoints = self.inputs.shape[0]

        if use_constraints:
            constraints = np.repeat(np.expand_dims(self.manager.get_constrains().T, axis=0), nDataPoints, axis=0)
            self.inputs = np.concatenate((self.inputs, constraints), axis=1)

        self.nTrain = round(ratio_train * nDataPoints)  # Total train set
        self.nValid = round(ratio_valid * nDataPoints)  # Validation set
        self.nTest = nDataPoints - self.nTrain - self.nValid
        assert self.nTest > 0

        randPerm = np.random.permutation(nDataPoints)
        # And choose the indices that will correspond to each dataset
        indexTrainPoints = randPerm[0:self.nTrain]
        indexValidPoints = randPerm[self.nTrain: self.nTrain + self.nValid]
        indexTestPoints = randPerm[self.nTrain + self.nValid: nDataPoints]
        # Finally get the corresponding samples and store them
        self.samples['train']['signals'] = self.inputs[indexTrainPoints, :, :]
        self.samples['train']['labels'] = self.gen[indexTrainPoints, :]
        self.samples['valid']['signals'] = self.inputs[indexValidPoints, :, :]
        self.samples['valid']['labels'] = self.gen[indexValidPoints, :]
        self.samples['test']['signals'] = self.inputs[indexTestPoints, :, :]
        self.samples['test']['labels'] = self.gen[indexTestPoints, :]
        # And update the index of the data points.
        self.indexDataPoints = {'train': indexTrainPoints, 'valid': indexValidPoints, 'test': indexTestPoints}

        self.adjacencyMatrix = adjacency_from_net(self.net, A_scaling, A_threshold)[0]

        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def case_info(self):
        return {
            'num_nodes': self.inputs.shape[2],
            'num_gen': self.gen.shape[1],
            'gen_index': self.manager.get_gen_index()
        }

    def getGraph(self):
        return np.copy(self.adjacencyMatrix)

    def rms(self, yHat, y):
        # Now, we compute the RMS
        yNorm = np.sum((y ** 2), axis=1)
        mse = np.sum((y - yHat) ** 2, axis=1)
        nmse = np.mean(mse / yNorm)
        return np.sqrt(nmse)

    def l1(self, yHat, y):
        return np.mean(np.sum(np.abs(y - yHat), axis=1), axis=0)

    def linf(self, yHat, y):
        return np.mean(np.max((y - yHat), axis=1), axis=0)

    def cost_percentage(self, yHat, y):
        cHat = []
        c = []
        violated_count = 0
        violations = 0
        constraint_count = 0

        print("Calculating violation rate.")
        for i in tqdm.trange(y.shape[0]):
            _yHat = yHat[i, :]
            _y = y[i, :]
            self.manager.set_gen(_y)
            _c = self.manager.cost()
            self.manager.set_gen(_yHat)
            _cHat = self.manager.cost()
            _violations = self.manager.count_violations()
            violated_count += _violations
            if _violations > 0:
                violations += 1
            constraint_count += self.manager.count_contraints()
            if _cHat is not None:
                cHat.append(_cHat)
                c.append(_c)
        violation_percentage = violated_count / constraint_count
        if len(cHat) == 0 or len(c) == 0:
            return 0, violation_percentage
        print(len(cHat))
        cHat = np.vstack(cHat)
        c = np.vstack(c)
        return np.mean(cHat / c), violations, violation_percentage

    def evaluate(self, yHat, y):
        # y and yHat should be of the same dimension, where dimension 0 is the
        # number of samples
        N = y.shape[0]  # number of samples
        assert yHat.shape[0] == N
        # And now, get rid of any extra '1' dimension that might appear
        # involuntarily from some vectorization issues.
        y = y.squeeze()
        yHat = yHat.squeeze()
        # Yet, if there was only one sample, then the sample dimension was
        # also get rid of during the squeeze, so we need to add it back
        if N == 1:
            y = y.unsqueeze(0)
            yHat = yHat.unsqueeze(0)

        if 'torch' in repr(self.dataType):
            yHat = yHat.data.cpu().numpy()
            y = y.data.cpu().numpy()
        return self.rms(yHat, y)

    def astype(self, dataType):
        # This changes the type for the incomplete and adjacency matrix.
        if repr(dataType).find('torch') == -1:
            self.adjacencyMatrix = dataType(self.adjacencyMatrix)
        else:
            self.adjacencyMatrix = torch.tensor(self.adjacencyMatrix).type(dataType)
        super().astype(dataType)

    def to(self, device):
        # If the dataType is 'torch'
        if repr(self.dataType).find('torch') >= 0:
            # Change the stored attributes that are not handled by the inherited
            # method to().
            self.adjacencyMatrix.to(device)
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)


def f(manager, length, data):
    n, l = data
    # if n % 100 == 0:
    # print("[{0:.0f}]".format(n / length * 100))
    try:
        _, gen_init = manager.optimal_dc()
        manager.set_load(l)
        manager.set_gen(gen_init)
        b = manager.powerflow()[0]
        g = manager.optimal_ac()[1]
        # manager.powerflow()
        # if not manager.net.converged:
        #     print("PF did not converge!")
        #     g = None
        # elif manager.check_violation():
        #     print("PF violation!")
        #     g = None
        manager.reset_gen()
        if g is None or b is None or l is None:
            return None, None, None
        return l, b, g
    except OPFNotConverged:
        manager.reset_gen()
        return None, None, None


if __name__ == '__main__':
    os.chdir("..")
    # Parameters
    case_name = "case118"
    state = "AK"  # state to use data from
    load_scale = 1.0  # scale the load by a factor
    portion_commercial = 0.5  # how much power should be commercial
    data_dir = "data/"

    case_dir = os.path.join(data_dir, case_name)
    profile_dir = data_dir + "load_profiles/"

    net = load_case(case_name, data_dir, reindex=True)
    manager = NetworkManager(net)

    load = manager.get_load(reactive=True) * load_scale
    p, q = np.split(load, 2, axis=1)
    p = LoadGenerator.generate_load_from_random(p, 20000, delta=0.1)
    q = LoadGenerator.generate_load_from_random(q, 20000, delta=0.1)
    load = np.stack((p, q), axis=2)

    results = None
    with Pool() as p:
        g = partial(f, manager, load.shape[0])
        results = list(tqdm.tqdm((p.imap(g, enumerate(load))), total=load.shape[0]))

    load, bus, gen = zip(*results)
    isNotNone = lambda x: x is not None
    load = np.stack(list(filter(isNotNone, load)))
    bus = np.stack(list(filter(isNotNone, bus)))
    gen = np.stack(list(filter(isNotNone, gen)))
    print(gen.shape)
    np.savez(os.path.join(case_dir, "data.npz"), load=load, bus=bus, gen=gen, constraints=constraints)
