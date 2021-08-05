from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_equality(title, v1, v2):
    if isinstance(v1, torch.Tensor):
        v1 = v1.numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.numpy()
    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)

    plt.figure()
    plt.stairs(v1, baseline=None)
    plt.stairs(v2, baseline=None)
    plt.legend(("Value 1", "Value 2"))
    plt.title(title)


def plot_inequality(title, value, lower, upper):
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    if isinstance(lower, torch.Tensor):
        lower = lower.numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.numpy()
    value = np.squeeze(value)
    lower = np.squeeze(lower)
    upper = np.squeeze(upper)

    violation = (value > upper) | (lower > value)

    plt.figure()
    n = np.arange(len(value))
    plt.stairs(value, color='b', baseline=None)
    plt.stairs(lower, color='r', baseline=None)
    plt.stairs(upper, color='g', baseline=None)
    plt.scatter(n[violation]+0.5, value[violation], c="r", marker="|")
    plt.legend(("val", "min", "max", "violation"))
    plt.title(title)
