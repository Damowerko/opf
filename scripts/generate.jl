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
    help = "Number of samples to generate if `--label_train` is provided the samples will be labeled."
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
using JSON
using HDF5

# load HSL if available
try
    using HSL_jll
    global use_hsl = true
    println("HSL available.")
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

function generate_samples_unlabeled(network_data, n_samples, min_load=0.9, max_load=1.1)::Array{Dict,1}
    samples = Array{Dict,1}(undef, n_samples)
    progress = Progress(n_samples, desc="Generating unlabeled samples:")
    Threads.@threads for i = 1:n_samples
        load = sample_load(network_data, min_load, max_load)
        samples[i] = Dict("load" => load)
        next!(progress)
    end
    return samples
end

"""
Generate `n_samples` from the power system network. Each sample is labeled with the optimal solution.
Samples that did not converge are discarded. Outputs an array of dictionaries with the following keys:
- `load`: a dictionary with the active and reactive load at each bus.
- `result`: the result of the optimization problem.
"""
function generate_samples(network_data, n_samples, min_load=0.9, max_load=1.1)::Tuple{Array{Float64,3},Array{Float64,3},Array{Dict,1}}
    count_atomic = Threads.Atomic{Int}(0)
    samples = Array{Dict,1}(undef, n_samples)
    progress = Progress(n_samples, desc="Generating labeled samples:")
    Threads.@threads for i = 1:n_samples
        solved = false
        while !solved
            Threads.atomic_add!(count_atomic, 1) # count the number of samples taken
            load = sample_load(network_data, min_load, max_load)
            result, solved = label_network(network_data, load)
            if !solved
                continue
            end
            samples[i] = Dict("load" => load, "result" => result)
        end
        next!(progress)
    end
    samples_memory = Base.summarysize(samples) / 1024^2
    println("Memory usage for $n_samples samples: $samples_memory MB.")

    count = count_atomic[]
    println("Generated $n_samples feasible samples using $count samples. Feasibility ratio: $(n_samples / count).")
    return samples
end

"""
Generate `n_samples` from the power system network. Each sample is labeled with the optimal solution.
Samples that did not converge are discarded. Outputs a dictionary with the following keys:
- `load`: an array of (n_samples, n_load, 2) with the active and reactive load at each bus.
- `gen`: an array of (n_samples, n_gen, 2) with the active and reactive power generated at each generator.
- `termination_status`: an array of (n_samples,) with the termination status of the optimization problem.
- `primal_status`: an array of (n_samples,) with the primal status of the optimization problem.
- `dual_status`: an array of (n_samples,) with the dual status of the optimization problem.
- `solve_time`: an array of (n_samples,) with the time taken to solve the optimization problem.
- `objective`: an array of (n_samples,) with the objective value of the optimization problem.
"""
function generate_samples_numpy(network_data, n_samples, min_load=0.9, max_load=1.1)
    count_atomic = Threads.Atomic{Int}(0)
    data = Dict(
        "bus" => Array{Float64,3}(undef, 2, length(network_data["bus"]), n_samples),
        "load" => Array{Float64,3}(undef, 2, length(network_data["load"]), n_samples),
        "gen" => Array{Float64,3}(undef, 2, length(network_data["gen"]), n_samples),
        "termination_status" => Array{String}(undef, n_samples),
        "primal_status" => Array{String}(undef, n_samples),
        "dual_status" => Array{String}(undef, n_samples),
        "solve_time" => Array{Float64}(undef, n_samples),
        "objective" => Array{Float64}(undef, n_samples),
    )
    progress = Progress(n_samples, desc="Generating labeled samples:")
    Threads.@threads for i = 1:n_samples
        solved = false
        while !solved
            Threads.atomic_add!(count_atomic, 1)
            load = sample_load(network_data, min_load, max_load)
            result, solved = label_network(network_data, load)
            if !solved
                continue
            end
            for j = 1:length(network_data["bus"])
                data["bus"][1, j, i] = network_data["bus"]["$(j)"]["vm"]
                data["bus"][2, j, i] = network_data["bus"]["$(j)"]["va"]
            end
            for j = 1:length(network_data["load"])
                data["load"][1, j, i] = load["$(j)"]["pd"]
                data["load"][2, j, i] = load["$(j)"]["qd"]
            end
            for j = 1:length(network_data["gen"])
                data["gen"][1, j, i] = result["solution"]["gen"]["$(j)"]["pg"]
                data["gen"][2, j, i] = result["solution"]["gen"]["$(j)"]["qg"]
            end
            data["termination_status"][i] = string(result["termination_status"])
            data["primal_status"][i] = string(result["primal_status"])
            data["dual_status"][i] = string(result["dual_status"])
            data["solve_time"][i] = result["solve_time"]
            data["objective"][i] = result["objective"]
        end
        next!(progress)
    end
    count = count_atomic[]
    println("Generated $n_samples feasible samples using $count samples. Feasibility ratio: $(n_samples / count).")
    return data
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

function reindex(data::Dict{String,Any})
    data = deepcopy(data)
    # reindex buses
    bus_ordered = sort(collect(values(data["bus"])), by=(x) -> x["index"])
    bus_id_map = Dict{Int,Int}()
    for (i, bus) in enumerate(bus_ordered)
        bus_id_map[bus["index"]] = i
    end
    update_bus_ids!(data, bus_id_map)
    # reindex generators
    new_gen = Dict{String,Any}()
    gen_ordered = sort(collect(values(data["gen"])), by=(x) -> x["index"])
    for (i, gen) in enumerate(gen_ordered)
        gen["index"] = i
        new_gen["$(i)"] = gen
    end
    data["gen"] = new_gen
    # reindex loads
    new_load = Dict{String,Any}()
    load_ordered = sort(collect(values(data["load"])), by=(x) -> x["index"])
    for (i, load) in enumerate(load_ordered)
        load["index"] = i
        new_load["$(i)"] = load
    end
    data["load"] = new_load
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
    network_data = reindex(network_data)

    check_assumptions!(network_data)

    # save network data in JSON format
    open(joinpath(out_dir, casename * ".json"), "w") do f
        JSON.print(f, network_data, 4)
    end

    data = generate_samples_numpy(network_data, n_samples, min_load, max_load)
    # write to h5 file
    h5file = joinpath(out_dir, casename * ".h5")
    h5open(h5file, "w") do file
        for (k, v) in data
            write(file, k, v)
        end
    end

end

main()