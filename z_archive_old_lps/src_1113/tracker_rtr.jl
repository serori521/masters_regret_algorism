isdefined(Main, :SetRegretCore) || include("cp_regret_replace.jl")

module RegretRTRTracker

using ..SetRegretCore

export find_change_points,
       find_change_points_debug,
       print_change_point_logs,
       snapshot_at

const EPS = SetRegretCore.EPS_DEFAULT

@inline function _dedup_points!(xs::Vector{Float64}; eps::Float64=1e-9)
    sort!(xs)
    keep = Float64[]
    for x in xs
        if isempty(keep) || x - keep[end] > eps
            push!(keep, x)
        end
    end
    empty!(xs)
    append!(xs, keep)
    return xs
end

function _run_core(utility::Matrix{Float64}, L::Vector{Float64}, R::Vector{Float64})
    tL, tR = SetRegretCore.find_optimal_trange(L, R)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    result = SetRegretCore.run_lps(matrix, L, R, tL, tR)
    return (tL=tL, tR=tR, result=result)
end

"""
区間 [tL, tR] で検出された変化点（必要なら端点を含む）を返す。
"""
function find_change_points(
    utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64};
    include_endpoints::Bool=true
)
    data = _run_core(utility, L, R)
    raw = include_endpoints ?
        vcat([data.tL], data.result.changes, [data.tR]) :
        copy(data.result.changes)
    return collect(_dedup_points!(raw))
end

"""
任意 t における MR スナップショットを取得。
"""
function snapshot_at(
    utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64},
    t::Float64
)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(matrix, L, R, t)
    MR = SetRegretCore.max_regret_vector(matrix, t)
    ranking = SetRegretCore.ranking_from_MR(copy(MR))
    winners = findall(x -> x <= minimum(MR) + EPS, MR)
    return (t=t, MR=MR, ranking=ranking, winners=winners)
end

"""
変化点リストと、各点でのスナップショットを返す。
"""
function find_change_points_debug(
    utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64};
    include_endpoints::Bool=true
)
    cps = find_change_points(utility, L, R; include_endpoints=include_endpoints)
    logs = [snapshot_at(utility, L, R, t) for t in cps]
    return cps, logs
end

"""
ログを簡易表示。
"""
function print_change_point_logs(logs)
    for (idx, log) in enumerate(logs)
        println("---- Change #", idx, "  t = ", log.t)
        println("winners (min MR): ", log.winners)
        println("ranking (low→high MR): ", log.ranking)
        println("MR: ", log.MR)
    end
end

end # module RegretRTRTracker
