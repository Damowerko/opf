using ArgParse

# Script to generate test data for the ACOPF problem.

# parse arguments
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
    "--n_samples"
    help = "Number of samples to test."
    arg_type = Int
    required = true
    "--min_load"
    help = "Width of the uniform distribution to sample the load."
    arg_type = Float64
    default = 0.9
    "--max_load"
    help = "Width of the uniform distribution to sample the load."
    arg_type = Float64
    default = 1.1
end
args = parse_args(ARGS, s)

# now the real script begins (so --help is fast)
using PowerModels
using Ipopt
using Random
using ProgressMeter
using NPZ

# load HSL if available
try
    using HSL_jll
    global use_hsl = true
catch
    global use_hsl = false
    println("HSL not available.")
end


"""
Sample a network model based on the reference case.

Currently we only sample the load uniformly +- `width` around the reference case.
"""
function sample_load(network_data::Dict{String,Any}, min_load=0.9, max_load=1.1)::Dict{String,Any}
    load = deepcopy(network_data["load"])
    for (k, v) in load
        # sample load around [REF * min_load, REF * max_load]
        width = max_load - min_load
        v["pd"] = v["pd"] * (width * rand() + min_load)
        v["qd"] = v["qd"] * (width * rand() + min_load)
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
    if use_hsl
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 1, "linear_solver" => "ma27", "hsllib" => HSL_jll.libhsl_path, "sb" => "yes")
    else
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 1, "sb" => "yes")
    end
    result = solve_ac_opf(network_data, solver)
    solved = result["termination_status"] == LOCALLY_SOLVED
    return result, solved
end

function check_assumptions!(network_data)
    # we assume that the data is in per unit
    @assert network_data["per_unit"] == true
    # ensure one generator per bus
    gen_buses = Set{Int}()
    for (k, v) in network_data["gen"]
        @assert !(v["gen_bus"] in gen_buses)
        @assert v["model"] == 2 # only consider the quadratic cost model
        push!(gen_buses, v["gen_bus"])
    end
    @assert length(network_data["storage"]) == 0
    @assert length(network_data["switch"]) == 0
    @assert length(network_data["dcline"]) == 0
end

function reindex_bus(data::Dict{String,Any})
    data = deepcopy(data)
    bus_ordered = sort([bus for (i, bus) in data["bus"]], by=(x) -> x["index"])
    bus_id_map = Dict{Int,Int}()
    for (i, bus) in enumerate(bus_ordered)
        bus_id_map[bus["index"]] = i
    end
    update_bus_ids!(data, bus_id_map)
    return data
end

function main()
    casefile = args["casefile"]
    casename = replace(splitext(basename(casefile))[1], "pglib_opf_" => "")
    out_dir = args["out"]
    n_samples = args["n_samples"]
    min_load = args["min_load"]
    max_load = args["max_load"]

    if max_load <= min_load
        error("max_load must be greater than min_load")
    end

    println("There are $(Threads.nthreads()) threads available.")

    # load case file
    network_data = PowerModels.parse_file(casefile)
    # reindex bus ids to be contiguous from 1 to N
    network_data = reindex_bus(network_data)

    check_assumptions!(network_data)

    times = Array{Float64,1}(undef, n_samples)
    progress = Progress(n_samples, desc="Generating labeled samples:")
    for i = 1:n_samples
        solved = false
        while !solved
            load = sample_load(network_data, min_load, max_load)
            time = @elapsed begin
                _, solved = label_network(network_data, load)
            end
            if !solved
                continue
            end
            times[i] = time
        end
        next!(progress)
    end
    mean_time = mean(times)
    std_time = std(times)
    println("Mean time: $mean_time, std: $std_time for $n_samples samples.")

    # write times to NPZ file
    npzwrite(joinpath(out_dir, "times_ipopt_$casename.npz"), times)
end

main()