module OPFHelpers

# Include libraries here so --help is faster.
using PrecompileTools
using PowerModels
using Ipopt

# load HSL if available
try
    import HSL_jll
    global use_hsl = true
catch
    global use_hsl = false
    println("HSL not available.")
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
        i = v["index"]
        pg = Sg[i, 1]
        qg = Sg[i, 2]
        v["pg"] = pg
        v["qg"] = qg
    end

    # if possible use hsl to speed up computation
    if use_hsl
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "max_iter" => 1000, "print_level" => 0, "linear_solver" => "ma27", "hsllib" => HSL_jll.libhsl_path, "sb" => "yes")
    else
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "max_iter" => 1000, "print_level" => 0, "sb" => "yes")
    end
    result = PowerModels.solve_ac_pf(network_data, solver)
    if (result["termination_status"] != LOCALLY_SOLVED)
        println("The problem was not solved to optimality: $(result["termination_status"])")
    end

    # parse the result as array
    n_bus = length(network_data["bus"])
    V = zeros(Float64, n_bus, 2)
    for (k, v) in result["solution"]["bus"]
        i = network_data["bus"][k]["bus_i"]
        V[i, 1] = v["vm"] * cos(v["va"])
        V[i, 2] = v["vm"] * sin(v["va"])
    end
    n_gen = length(network_data["gen"])
    Sg = zeros(Float64, n_gen, 2)
    for (k, v) in result["solution"]["gen"]
        i = network_data["gen"][k]["index"]
        Sg[i, 1] = v["pg"]
        Sg[i, 2] = v["qg"]
    end
    return V, Sg
end

@compile_workload begin
    # toy example to precompile the code

    if use_hsl
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0, "linear_solver" => "ma57", "hsllib" => HSL_jll.libhsl_path)
    else
        solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "print_level" => 0)
    end
    data = PowerModels.parse_file(normpath(joinpath(@__DIR__, "../data/case3.m")))

    # read matrices from networkdata
    n_bus = length(data["bus"])
    V = zeros(n_bus, 2)
    Sd = zeros(n_bus, 2)
    Sg = zeros(n_bus, 2)
    for (k, v) in data["bus"]
        i = v["bus_i"]
        V[i, 1] = v["vm"] * cos(v["va"])
        V[i, 2] = v["vm"] * sin(v["va"])
    end
    for (k, v) in data["load"]
        bus = v["load_bus"]
        Sd[bus, 1] = v["pd"]
        Sd[bus, 2] = v["qd"]
    end
    for (k, v) in data["gen"]
        bus = v["gen_bus"]
        Sg[bus, 1] = v["pg"]
        Sg[bus, 2] = v["qg"]
    end

    # call several methods to precompile
    project(data, V, Sg, Sd)
    solve_ac_opf(data, solver)
    solve_ac_pf(data, solver)
end

end # module OPFHelpers
