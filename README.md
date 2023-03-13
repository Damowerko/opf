# OPF

If you found this work useful you can cite our conference paper and preprint.

```bibtex
@inproceedings{owerko2020opf,
  author={Owerko, Damian and Gama, Fernando and Ribeiro, Alejandro},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Optimal Power Flow Using Graph Neural Networks}, 
  year={2020},
  pages={5930-5934},
  doi={10.1109/ICASSP40776.2020.9053140}
}

@misc{owerko2022opf,
  doi = {10.48550/ARXIV.2210.09277},
  url = {https://arxiv.org/abs/2210.09277},
  author = {Owerko, Damian and Gama, Fernando and Ribeiro, Alejandro},
  title = {Unsupervised Optimal Power Flow Using Graph Neural Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Installation

### Python

You will need to have a python version installed that is not statically linked.
You can install such a version using `pyenv`.
```PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.9```
This will create a python version linked to a shared library.

This project uses [poetry](https://python-poetry.org/) for dependency managemnt. To install the project and all
the dependencies simply call use `poetry install`. Poetry will create a virtualenv for you or alternatively you can use,
```
poetry config virtualenvs.create false --local
```
to supress virtualenv creation for the project.

### Julia

You will need to install Julia. Follow the instructions on the (pandapower docs)[https://pandapower.readthedocs.io/en/develop/opf/powermodels.html#].
Instead of step 2, run the following in the python version installed above. This will install and configure PyCall in the julia installation. 
```
import julia
julia.install()
```

Make sure you install all the required packages in julia.
```
add Ipopt PowerModels PandaModels JSON Cbc Juniper JuMP
```

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
* case57
* case118

Broken:
* case4gs
* case5

## Load profiles
Though currently unused, data for generating "realistic" load profile based on historical data is available 
[here](https://openei.org/doe-opendata/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations-in-the-united-states).




