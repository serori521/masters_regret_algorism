module SetRegretCore

export minimax_regret_tuple,
       find_optimal_trange,
       create_minimax_R_Matrix,
       initialize_linear_models!,
       run_lps,
       max_regret_vector,
       ranking_from_MR

const EPS_DEFAULT = 1e-12

const SnapshotEntry = NamedTuple{
    (:t, :MR, :rank, :winners),
    Tuple{Float64, Vector{Float64}, Vector{Int}, Vector{Int}}
}

include(joinpath(@__DIR__, "set_regret_core", "regret_core.jl"))
include(joinpath(@__DIR__, "set_regret_core", "events.jl"))
include(joinpath(@__DIR__, "set_regret_core", "run_lps.jl"))

end # module
