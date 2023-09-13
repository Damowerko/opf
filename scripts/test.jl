using ArgParse

# Script to generate test data for the ACOPF problem.

# parse arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--casename"
    help = "Path to the parsed casefile in json format."
    arg_type = String
    required = true
    "--data"
    help = "Path to the data directory."
    arg_type = String
    default = "./data"
    "--out"
    help = "Directory into which to write the result of the test."
    arg_type = String
    required = true
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
using NPZ

# load HSL if available
try
    using HSL
    global use_hsl = true
catch
    global use_hsl = false
end

function test(network_data::Dict{String,Any}, data::Dict{String,Any}, output::Array{Float32})
    println("Hello, World!")
end

function main()
    casename = args["casename"]
    out_dir = args["out"]
    data_dir = args["data"]

    casefile = joinpath(data_dir, casename * ".json")
    datafile = joinpath(data_dir, casename * ".test.json")
    model_outfile = joinpath(out_dir, "outputs" * ".npy")

    println("There are $(Threads.nthreads()) threads available.")

    # load case file
    network_data = JSON.parsefile(casefile)

    # load the test dataset
    dataset = JSON.parsefile(datafile)
    # load the model output matrix
    outputs = NPZ.npzread(model_outfile)
    for i in eachindex(dataset, outputs)
        test(network_data, dataset[i], outputs[i])
    end
end

main()