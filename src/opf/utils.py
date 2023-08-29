import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt
from torchcps.gnn import ParametricGNN

from opf.modules import OPFLogBarrier


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


def create_model(params: dict):
    input_features = 14
    output_features = 4
    n_edges = 10
    model = ParametricGNN(input_features, output_features, n_edges, **params).float()
    barrier = OPFLogBarrier(model, **params)
    return barrier
