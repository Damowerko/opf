import numpy as np
import scipy.sparse
import torch.nn
from matplotlib import pyplot as plt

from opf.dataset import CaseDataModule
from opf.modules import OPFLogBarrier, SimpleGNN


def graph_info(gso, plot=False):
    print(f"Non-zero edges: {np.sum(np.abs(gso) > 0)}")
    print(f"Connected components: {scipy.sparse.csgraph.connected_components(gso)[0]}")

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gso)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.hist(gso[gso > 0].flat, bins=20, range=(0, 1))
        plt.title("Distribution of non-zero edge weights")
        plt.show()


def create_model(dm: CaseDataModule, params: dict):
    input_features = 8 if params["constraint_features"] else 2
    output_features = 2
    model = SimpleGNN(input_features, output_features, **params)
    barrier = OPFLogBarrier(dm.net_wrapper, model, **params)
    return barrier
