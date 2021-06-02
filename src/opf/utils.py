import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt


def graph_info(gso, plot=False):
    print(f"Non-zero edges: {np.sum(np.abs(gso) > 0)}")
    print(f"Connected components: {scipy.sparse.csgraph.connected_components(gso)[0]}")

    if plot:
        plt.figure()
        plt.imshow(gso)
        plt.colorbar()

        plt.figure()
        plt.hist(gso[gso > 0].flat, bins=20, range=(0, 1))
        plt.title("Distribution of non-zero edge weights")
        plt.show()
