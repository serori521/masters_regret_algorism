# src/init_state.jl
module InitState

using ..CoreTypes: LPSInstance, LPSState

function make_instance(read_utility_value, paths, utility::String; N::Int=6, M::Int=5)
    U = read_utility_value(paths, utility; N=N, M=M)
    return LPSInstance(utility, N, M, U)
end

function init_state(inst::LPSInstance; t0::Float64=0.0)
    return LPSState(t0, inst)
end

end # module
