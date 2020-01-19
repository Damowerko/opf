# OPF
## Code Overview
`wrapper.py` is the main file used for experiments. Here you chose the models you want to experiment with and their hyperparameters.

`power.py` wraps pandapower in two classes and provides several helper functions. `load_case` loads cases from the [pandapower.networks](https://pandapower.readthedocs.io/en/v2.2.0/networks.html) module. `adjacency_from_net` creates an adjacency matrix based off of the imported network object. `NetworkManager` provides a set of functions to manipulate a pandapower network structure. Pandapower uses pandas to store information about the electrical network in tables, this class allows you to easily manipulate those as numpy matrices. Finally, `LoadGenerator` is a utility for generating synthetic load profiles based on real data or sampled from a uniform distribution.
