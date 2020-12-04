from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from pyopf.power import NetworkManager, LoadGenerator, OPFNotConverged

def generate_samples(manager: NetworkManager, num_samples, load_scale=1.0):
    load = manager.get_load(reactive=True) * load_scale
    p, q = np.split(load, 2, axis=1)
    p = LoadGenerator.generate_load_from_random(p, num_samples, delta=0.1)
    q = LoadGenerator.generate_load_from_random(q, num_samples, delta=0.1)
    load_samples = np.stack((p, q), axis=2)
    return load_samples


def label_sample(manager: NetworkManager, load_sample):
    """
    Generate the optimal power flow label.
    :param manager: NetworkManger instance.
    :param load_sample: The active and reactive load at each node.
    :return:
    """
    try:
        _, gen_init = manager.optimal_dc()
        manager.set_load(load_sample)
        manager.set_gen(gen_init)
        return manager.optimal_ac()[0]
    except OPFNotConverged:
        return None


def label_samples(manager, load_samples):
    """

    :param manager: NetworkManager instance.
    :param load_samples: Active and reactive loads on each node per sample.
    :return: A list of tuples (load_samples, optimal_power_flow) where values of opf which did not converge were
        filtered out.
    """
    with Pool() as p:
        f = partial(label_sample, manager)
        opfs = list(tqdm.tqdm((p.imap(f, load_samples)), total=load_samples.shape[0]))
    data = zip(load_samples, opfs)
    data = list(filter(lambda x: x[1] is not None, data))
    return zip(*data)

