# Script to generate test data for the ACOPF problem.
using ArgParse

# parse arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--casefile"
    help = "Path to the parsed casefile in json format."
    arg_type = String
    required = true
    "--busfile"
    help = "Path to numpy archive containing the bus data V, Sg and Sd."
    arg_type = String
    required = true
end
args = parse_args(ARGS, s)

# import here so --help is faster
using OPFHelpers
using JSON
using NPZ
using PowerModels
using Memento

function main()
    casefile = args["casefile"]
    busfile = args["busfile"]

    # supress info and warning messages from PowerModels
    Memento.setlevel!(Memento.getlogger(PowerModels), "error")

    # load case file
    network_data = JSON.parsefile(casefile)

    # load the model output matrix
    bus_variables = NPZ.npzread(busfile)
    V, Sd, Sg = bus_variables["V"], bus_variables["Sd"], bus_variables["Sg"]
    if (size(V) != size(Sd))
        error("The size of the variables is not the same.")
    end

    # convert arrays to float32
    V = Float64.(V)
    Sd = Float64.(Sd)
    Sg = Float64.(Sg)

    n_bus = size(V, 1)
    Threads.@threads for i in 1:n_bus
        V[i, :, :], Sg[i, :, :] = OPFHelpers.project(network_data, V[i, :, :], Sg[i, :, :], Sd[i, :, :])
    end
    # write the projected variables to file
    NPZ.npzwrite(busfile, Dict("V" => V, "Sd" => Sd, "Sg" => Sg))
end

main()