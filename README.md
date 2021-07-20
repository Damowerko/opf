# OPF

## Installation
This project uses [poetry](https://python-poetry.org/) for dependency managemnt. To install the project and all
the dependencies simply call use `poetry install`. Poetry will create a virtualenv for you.

If your GPU needs CUDA 11 then you will need to change the pytroch version manually. Each time 
after calling `poetry install` you will need to to the following.
1. Activate the virtualenv using `poetry shell`.
2. Run `poe cuda11`.

## Code Overview
### Files
   * `train.py` is the main file used for experiments. It uses Weights and Biases for cloud based logging. You can
    change the hyperparameters by editing the hyperparameter dict.
   * `power.py` wraps pandapower in two classes and provides several helper functions.
   * `generate.py` is a utility for generating a dataset for test cases.
### Classes
   * `load_case` loads cases from the [pandapower.networks](https://pandapower.readthedocs.io/en/v2.2.0/networks.html)
     module. 
   * `adjacency_from_net` creates an adjacency matrix based off of the imported network object. 
   * `NetWrapper` provides a set of functions to manipulate the PandaPower data structure. 
     Pandapower uses pandas to store information about the electrical network in tables,
     this class allows you to easily manipulate those as numpy matrices.
   * `LoadGenerator` is a utility for generating synthetic load profiles based on real data or sampled from a uniform distribution.

## Conventions

### Units
We follow the conventions from PowerModels.jl. All quantities are expressed in the
[per-unit system](https://en.wikipedia.org/wiki/Per-unit_system).
This simplifies computations and typically normalizes values.

All angles are expressed in radians where possible. This is contrary to the PandaPower convention.

### Bus Values
We can describe the state of the buses in the grid by two complex quantities:
* the voltage at each node
* and the power injected at the node (PandaPower values have the opposite sign).
  This can be represented by a Nx4 matrix containing the real and imaginary components of both quantities.

## Test Cases
Not of pandapower test cases converge during optimal power flow. Some have missing parameters, necessary for
OPF computation, others just fail to converge using PYPOWER. Below is a list of tested test cases.

Working:
* case6ww
* case30

Broken:
* case4gs
* case5

## Load profiles
Though currently unused, data for generating "realistic" load profile based on historical data is available 
[here](https://openei.org/doe-opendata/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations-in-the-united-states).

## Julia Dependencies (Optional)
PandaPower offers two backends for optimization: PYPOWER and Powermodels.jl. The latter is more reliable, but is
implemented in Julia and uses PyCall to execute Julia functions from Python. 
If you want to run dataset generation using PowerModels.jl as a backend for PandaPower you will need to install
the following Julia dependencies using the Julia package manager (press `]` in Julia REPL).
```
add Ipopt PowerModels PyCall JSON Cbc Juniper JuMP
``` 
Then, in a desired Python REPL you will need to run.
```
import julia
julia.install()
```
Now you can use PowerModels.jl as a Julia backend.




