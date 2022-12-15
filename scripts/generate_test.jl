using PowerModels
using Ipopt
using Random
using JuMP
using ProgressMeter
using JSON

"""
Sample a network model based on the reference case.

Currently we only sample the load uniformly +- `width` around the reference case.
"""
function sample_load(network_data::Dict{String,Any}, width=0.2)::Dict{String,Any}
    load = deepcopy(network_data["load"])
    for (k, v) in load
        # sample load around [REF * (1 - width), REF * (1 + width)]
        v["pd"] = v["pd"] * (1 + width * (2 * rand() - 1))
        v["qd"] = v["qd"] * (1 + width * (2 * rand() - 1))
    end
    return load
end

"""
Get the solution to the OPF problem defined by the PowerModels.jl `network_data`.
# Return
- `solution`: the solution to the OPF problem
- `solved`: `true` if the problem was solved, `false` otherwise
"""
function label_network(network_data::Dict{String,Any}, load::Dict{String,Any})::Tuple{Dict{String,Any},Bool}
    network_data = deepcopy(network_data)
    network_data["load"] = load
    result = solve_ac_opf(network_data, JuMP.with_optimizer(Ipopt.Optimizer, print_level=0))
    solved = result["termination_status"] == LOCALLY_SOLVED
    return result, solved
end

"""
Generate `n_samples` from the power system network. Each sample is labeled with the optimal solution.
Samples that did not converge are discarded.
"""
function generate_samples(network_data, n_samples, width=0.2)::Array{Dict,1}
    samples = Array{Dict,1}(undef, n_samples)
    progress = Progress(n_samples)
    Threads.@threads for i = 1:n_samples
        solved = false
        while !solved
            load = sample_load(network_data, width)
            result, solved = label_network(network_data, load)
            if !solved
                continue
            end
            samples[i] = Dict("load" => load, "result" => result)
        end
        next!(progress)
    end
    return samples
end

## Main script

# arguments
casefile = "data/pglib-opf-21.07/api/pglib_opf_case300_ieee__api.m"
out_dir = "data/"
n_samples = 100

PowerModels.silence()
# load case file
network_data = PowerModels.parse_file(casefile)
# we assume that the data is in per unit
@assert network_data["per_unit"] == true

# suppress PowerModels.jl output
samples = generate_samples(network_data, n_samples, 0.2);

# save the data
casename = splitext(basename(casefile))[1]

# save network data
open(joinpath(out_dir, casename * ".json"), "w") do f
    JSON.print(f, network_data, 4)
end
# save samples
open(joinpath(out_dir, casename * ".test.json"), "w") do f
    JSON.print(f, samples)
end