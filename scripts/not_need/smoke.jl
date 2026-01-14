include(joinpath(@__DIR__, "..", "src", "LPSv2.jl"))
using .LPSv2

paths = LPSv2.project_paths()

inst = LPSv2.make_instance(paths, "u1"; N=6, M=5)
st = LPSv2.init_state(inst; t0=0.0)

println(inst.utility, " N=", inst.N, " M=", inst.M, " mats=", length(inst.U))
println("state t=", st.t)
