# OPF

Code used in [Optimal Power Flow Using Graph Neural Networks](https://doi.org/10.1109/ICASSP40776.2020.9053140) and [Unsupervised Optimal Power Flow Using Graph Neural Networks](https://arxiv.org/abs/2210.09277). To cite our work please use the following biblatex citation. The version used for ICASSP is archived in a different branch.

```bibtex
@misc{owerko2022opf,
  doi = {10.48550/ARXIV.2210.09277},
  url = {https://arxiv.org/abs/2210.09277},
  author = {Owerko, Damian and Gama, Fernando and Ribeiro, Alejandro},
  title = {Unsupervised Optimal Power Flow Using Graph Neural Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@inproceedings{owerko2020opf,
  author={Owerko, Damian and Gama, Fernando and Ribeiro, Alejandro},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Optimal Power Flow Using Graph Neural Networks}, 
  year={2020},
  pages={5930-5934},
  doi={10.1109/ICASSP40776.2020.9053140}
}
```

## Reproducing Results

The code on the `main` branch is much more recent than used in the paper. To make reproducing results easier, I have created a branch 

### Unsupervised Optimal Power Flow Using Graph Neural Networks
The code from the paper is archived in the `unsupervised` branch.
Below I provide instructions on how to reproduce the results for the IEEE-30 system.

#### Data
Generated data is available on Google Drive. Download the data and place in `data/`, relative to the repo root.
* [IEEE-30](https://drive.google.com/file/d/1bM7rwOdLN555zRkLBRLgxSUgH2VkEnQy/view?usp=sharing) 
* [IEEE-118](https://drive.google.com/file/d/1UyztAKuIYxhTJyJFhLxOe-0MKUuFgn2s/view?usp=drive_web)

Alternatively use `generate.py` to generate your own data.
```
python -m opf.generate --help
```

#### Training
To train the model with the hyperparameters from the paper, run the following command.
```
python scripts/train.py --F 32 --K 8 --L 2 --activation leaky_relu --adj_threshold 0.01 --batch_size 256 --case_name case30 --constraint_features 0 --cost_weight 0.01 --enforce_constraints 0 --eps 0.0001 --gpus 1 --gradient_clip_val 0 --log 1 --lr 0.0003 --max_epochs 1000 --patience 1000 --readout local --s 10 --t 500 --wandb 0
```

#### Using the provided checkpoint
The branch `unsupervided` contains a checkpoint trained on the IEEE-30 system. It is inside `models/`. There is a checkpoint file and a yaml file specifying all the hyperparameters needed to load the model. You can create an instance of `OPFLogBarrier` using the hyperparameters and load the weights using `load_state_dict`.

### Optimal Power Flow Using Graph Neural Networks
The code from the paper is archived in the `icassp` branch. I do not recommend using this branch as it is outdated.

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

You will need to install Julia 1.6 (as of December 2022). Run the following in the python version installed above.
```
python -c "import julia; julia.install()"
```
This will install and configure PyCall in the julia installation. There are couple dependencies for PandaPower-PowerModels.jl integration. Open up julia REPL and press `]` to manage the packages. Then run the following.
```
add Ipopt PowerModels PandaModels JSON Cbc Juniper JuMP ProgressMeter ArgParse
```
You might want to run `test PandaModels` in the julia pakcage manager to test the installation.

### HSL Solver (Optional)
The HSL linear solver for IPOPT dramatically speeds up solving the problem by taking advantage of BLAS and LAPACK. PowerModels.jl (benchmarks)[https://lanl-ansi.github.io/PowerModels.jl/stable/experiment-results/] use the HSL_MA57 solver, which I will use as well.

What I do is compile the (Coin-HSL Full)[https://www.hsl.rl.ac.uk/ipopt/] solver archive. You need a license to do so, but it is free for academic use.
Installing the dependencies can be simplified by using (HSL.jl)[https://github.com/JuliaSmoothOptimizers/HSL.jl].
Note that HSL.jl depends on the `gfortran` compiler.
To build HSL.jl you will need to run the following in julia.
```
ENV["HSL_ARCHIVES_PATH"] = "PATH_TO_FOLDER_CONTAINING_ARCHIVE"
import Pkg
Pkg.build("HSL")
Pkg.test("HSL")  # If needed
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




