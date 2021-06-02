# OPF
## Code Overview
`wrapper.py` is the main file used for experiments. Here you chose the models you want to experiment with and their hyperparameters.

`power.py` wraps pandapower in two classes and provides several helper functions. `load_case` loads cases from the [pandapower.networks](https://pandapower.readthedocs.io/en/v2.2.0/networks.html) module. `adjacency_from_net` creates an adjacency matrix based off of the imported network object. `NetworkManager` provides a set of functions to manipulate a pandapower network structure. Pandapower uses pandas to store information about the electrical network in tables, this class allows you to easily manipulate those as numpy matrices. Finally, `LoadGenerator` is a utility for generating synthetic load profiles based on real data or sampled from a uniform distribution.

## Test Cases
Not of pandapower test cases converge during optimal power flow. Some have missing parameters, necessary for
OPF computation, others just fail to converge using PYPOWER. Below is a list of tested test cases.

Working:
* case6ww
* case30

Broken:
* case4gs
* case5

## Load profile
Though currently unused, data for generating "realistic" load profile based on historical data is available 
[here](https://openei.org/doe-opendata/dataset/commercial-and-residential-hourly-load-profiles-for-all-tmy3-locations-in-the-united-states).

## Julia Dependencies (Optional)
If you want to run dataset generation using PowerModels.jl as a backend you will need to install
`Ipopt PowerModels PyCall JSON Cbc Juniper JuMP` julia dependencies. Then you will need to run
```
import julia
julia.install()
```
from REPL.