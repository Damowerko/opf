from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

import opf.powerflow as pf


def plot_equality(title, target, value):
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    target = np.squeeze(target)
    value = np.squeeze(value)

    fig = plt.figure()
    plt.stairs(target - value, baseline=None)
    # plt.stairs(value, baseline=None)
    plt.legend(["Target - Value"])
    plt.title(title)
    return fig


def plot_inequality(title, value, lower, upper, eps=1e-4):
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    if isinstance(lower, torch.Tensor):
        lower = lower.numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.numpy()
    value = np.squeeze(value)
    lower = np.squeeze(lower)
    upper = np.squeeze(upper)

    violation = (value > upper + eps) | (lower - eps > value)

    fig = plt.figure()
    n = np.arange(len(value))
    plt.stairs(value, color="b", baseline=None)
    plt.stairs(lower, color="r", baseline=None)
    plt.stairs(upper, color="g", baseline=None)
    plt.scatter(n[violation] + 0.5, value[violation], c="r", marker="|")
    plt.legend(("val", "min", "max", "violation"))
    plt.title(title)
    return fig


def plot_constraints(constraints: Dict[str, pf.Constraint]):
    plots = {}
    for name, constraint in constraints.items():
        if isinstance(constraint, pf.EqualityConstraint):
            if torch.is_complex(constraint.target):
                suffix = [" real", " imag"]
                plots[name + suffix[0]] = plot_equality(
                    name + suffix[0], constraint.target.real, constraint.value.real
                )
                plots[name + suffix[1]] = plot_equality(
                    name + suffix[1], constraint.target.imag, constraint.value.imag
                )
            else:
                plots[name] = plot_equality(name, constraint.target, constraint.value)
        elif isinstance(constraint, pf.InequalityConstraint):
            plots[name] = plot_inequality(
                name, constraint.variable, constraint.min, constraint.max
            )
        else:
            raise ValueError(
                f"Unknown constraint type: {type(constraint)} with value {constraint}"
            )
    return plots
