using PowerModels
using Ipopt
using Random
using JuMP
using ProgressMeter
using JSON
using ArgParse

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
    solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0)
    result = solve_ac_opf(network_data, solver)
    solved = result["termination_status"] == LOCALLY_SOLVED
    return result, solved
end

"""
Generate `n_samples` from the power system network. Each sample is labeled with the optimal solution.
Samples that did not converge are discarded.
"""
function generate_samples(network_data, n_samples, width=0.2)::Array{Dict,1}
    count_atomic = Threads.Atomic{Int}(0)
    samples = Array{Dict,1}(undef, n_samples)
    progress = Progress(n_samples)
    Threads.@threads for i = 1:n_samples
        solved = false
        while !solved
            Threads.atomic_add!(count_atomic, 1) # count the number of samples taken
            load = sample_load(network_data, width)
            result, solved = label_network(network_data, load)
            if !solved
                continue
            end
            samples[i] = Dict("load" => load, "result" => result)
        end
        next!(progress)
    end
    count = count_atomic[]
    println("Generated $n_samples feasible samples using $count samples. Feasibility ratio: $(n_samples / count).")
    return samples
end

function check_assumptions(network_data)
    # we assume that the data is in per unit
    @assert network_data["per_unit"] == true
    # ensure one generator per bus
    gen_buses = Set{Int}()
    for (k, v) in network_data["gen"]
        @assert !(v["gen_bus"] in gen_buses)
        push!(gen_buses, v["gen_bus"])
    end
    @assert length(network_data["storage"]) == 0
    @assert length(network_data["switch"]) == 0
    @assert length(network_data["dcline"]) == 0
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--casefile"
        arg_type = String
        help = "MATPOWER case file path."
        required = true
        "--out"
        help = "Output directory path."
        arg_type = String
        default = "./data"
        "--samples"
        help = "Number of (labeled) samples to generate."
        arg_type = Int
        required = true
        "--width"
        help = "Width of the uniform distribution to sample the load."
        arg_type = Float64
        default = 0.2
    end

    args = parse_args(ARGS, s)
    casefile = args["casefile"]
    casename = splitext(basename(casefile))[1]
    out_dir = args["out"]
    n_samples = args["samples"]
    width = args["width"]

    println("There are $(Threads.nthreads()) threads available.")

    # load case file
    network_data = PowerModels.parse_file(casefile)
    check_assumptions(network_data)

    # save network data in JSON format
    open(joinpath(out_dir, casename * ".json"), "w") do f
        JSON.print(f, network_data, 4)
    end

    # generate the labeled samples
    samples = generate_samples(network_data, n_samples, width)

    #assuming 50/50 split for validation and test
    n_valid = Int(floor(n_samples / 2))
    samples_valid = @view samples[1:n_valid]
    samples_test = @view samples[(n_valid+1):end]
    open(joinpath(out_dir, casename * ".valid.json"), "w") do f
        JSON.print(f, samples_valid)
    end
    open(joinpath(out_dir, casename * ".test.json"), "w") do f
        JSON.print(f, samples_test)
    end
end

main()