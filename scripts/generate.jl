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
    "--label_train"
    help = "If provided will label the training data else only generate the data."
    action = :store_true
    "--n_train"
    help = "Number of samples to generate if `--label_train` is provided the samples will be labeled."
    arg_type = Int
    required = true
    "--n_valid"
    help = "Number of (labeled) samples to generate."
    arg_type = Int
    required = true
    "--n_test"
    help = "Number of (labeled) samples to generate."
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
using JuMP
using ProgressMeter
using JSON
using Pkg
using ZipFile

# load HSL if available
try
    using HSL
    global use_hsl = true
catch
    global use_hsl = false
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
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 1, "linear_solver" => "ma57", "hsllib" => HSL.libcoinhsl)
    else
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 1)
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
Samples that did not converge are discarded.
"""
function generate_samples(network_data, n_samples, min_load=0.9, max_load=1.1)::Array{Dict,1}
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
    count = count_atomic[]
    println("Generated $n_samples feasible samples using $count samples. Feasibility ratio: $(n_samples / count).")
    return samples
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
    n_train = args["n_train"]
    n_valid = args["n_valid"]
    n_test = args["n_test"]
    min_load = args["min_load"]
    max_load = args["max_load"]
    label_train = args["label_train"]

    
    if max_load <= min_load
        error("max_load must be greater than min_load")
    end
    
    println("There are $(Threads.nthreads()) threads available.")
    
    # load case file
    network_data = PowerModels.parse_file(casefile)
    # reindex bus ids to be contiguous from 1 to N
    network_data = reindex_bus(network_data)
    
    check_assumptions!(network_data)
    
    # save network data in JSON format
    open(joinpath(out_dir, casename * ".json"), "w") do f
        JSON.print(f, network_data, 4)
    end

    if label_train
        # generate the labeled samples
        n_labeled = n_train + n_valid + n_test
        samples = generate_samples(network_data, n_labeled, min_load, max_load)
        samples_train = @view samples[1:n_train]
        samples_valid = @view samples[(n_train+1):(n_train+n_valid)]
        samples_test = @view samples[(n_train+n_valid+1):end]
    else
        samples_train = generate_samples_unlabeled(network_data, n_train, min_load, max_load)
        n_labeled = n_valid + n_test
        samples = generate_samples(network_data, n_labeled, min_load, max_load)
        samples_valid = @view samples[1:n_valid]
        samples_test = @view samples[(n_valid+1):end]
    end

    # write data to zip file
    writer = ZipFile.Writer(joinpath(out_dir, casename * ".zip"))
    if n_train > 0
        f = ZipFile.addfile(writer, casename * "_train.json")
        JSON.print(f, samples_train)
    end
    if n_valid > 0
        f = ZipFile.addfile(writer, casename * "_valid.json")
        JSON.print(f, samples_valid)
    end
    if n_test > 0
        f = ZipFile.addfile(writer, casename * "_test.json")
        JSON.print(f, samples_test)
    end
    close(writer)
end

main()