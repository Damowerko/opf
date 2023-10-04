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

# Include libraries here so --help is faster.
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

"""
Run Powerflow given the load and the power generated at each bus.

Args:
    network_data: Dict{String,Any}
        A dictionary describing the power system.
    load: Dict{String,Any}
        A dictionary containing the load.
    Sg: Array{Float32}
        An array with the power generated at each bus.
"""
function project(network_data::Dict{String,Any}, V::Array{Float64,2}, Sg::Array{Float64,2}, Sd::Array{Float64,2})
    network_data = deepcopy(network_data)
    for (_, v) in network_data["bus"]
        i = v["bus_i"]
        voltage = V[i, 1] + 1im * V[i, 2]
        v["vm"] = abs(voltage)
        v["va"] = angle(voltage)
    end
    for (_, v) in network_data["load"]
        bus = v["load_bus"]
        v["pd"] = Sd[bus, 1]
        v["qd"] = Sd[bus, 2]
    end
    for (_, v) in network_data["gen"]
        bus = v["gen_bus"]
        pg = Sg[bus, 1]
        qg = Sg[bus, 2]
        # enforce constraints
        pg = max(pg, v["pmin"])
        pg = min(pg, v["pmax"])
        qg = max(qg, v["qmin"])
        qg = min(qg, v["qmax"])
        v["pg"] = Sg[bus, 1]
        v["qg"] = Sg[bus, 2]
    end

    # if possible use hsl to speed up computation
    # uncomment to use IPOPT
    if use_hsl
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0, "linear_solver" => "ma57", "hsllib" => HSL.libcoinhsl)
    else
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0)
    end
    result = PowerModels.solve_ac_pf(network_data, solver)
    if (result["termination_status"] != LOCALLY_SOLVED)
        println("The problem was not solved to optimality.")
    end

    # result = PowerModels.compute_ac_pf(network_data)
    # if (!result["termination_status"])
    #     println("The problem was not solved to optimality.")
    # end

    # parse the result as array
    n_bus = length(network_data["bus"])
    V = zeros(Float64, n_bus, 2)
    for (k, v) in result["solution"]["bus"]
        i = network_data["bus"][k]["bus_i"]
        V[i, 1] = v["vm"] * cos(v["va"])
        V[i, 2] = v["vm"] * sin(v["va"])
    end
    return V
end

function main()
    casefile = args["casefile"]
    busfile = args["busfile"]

    # load case file
    network_data = JSON.parsefile(casefile)

    # load the model output matrix
    bus_variables = NPZ.npzread(busfile)
    V, Sd, Sg = bus_variables["V"], bus_variables["Sd"], bus_variables["Sg"]
    if (size(V) != size(Sd) || size(V) != size(Sg))
        error("The size of the variables is not the same.")
    end

    # convert arrays to float32
    V = Float64.(V)
    Sd = Float64.(Sd)
    Sg = Float64.(Sg)

    n_bus = size(V, 1)
    Threads.@threads for i in 1:n_bus
        V[i, :, :] = project(network_data, V[i, :, :], Sg[i, :, :], Sd[i, :, :])
    end
    # write the projected variables to file
    NPZ.npzwrite(busfile, Dict("V" => V, "Sd" => Sd, "Sg" => Sg))
end

main()