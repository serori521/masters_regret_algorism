module SetRegretCore
# SetRegretCore.jl に追加（SnapshotEntry の下あたり）
const RankTimelineEntry = NamedTuple{(:t, :rank), Tuple{Float64, Vector{Int}}}

# export にも追加（必要なら）
export RankTimelineEntry

export minimax_regret_tuple,
    find_optimal_trange,
    create_minimax_R_Matrix,
    initialize_linear_models!,
    run_lps,
    max_regret_vector,
    ranking_from_MR

const EPS_DEFAULT = 1e-15

const SnapshotEntry = NamedTuple{
    (:t, :MR, :rank, :winners),
    Tuple{Float64,Vector{Float64},Vector{Int},Vector{Int}}
}

include(joinpath(@__DIR__, "set_regret_core", "regret_core.jl"))
include(joinpath(@__DIR__, "set_regret_core", "events.jl"))
include(joinpath(@__DIR__, "set_regret_core", "run_lps.jl"))


end # module
