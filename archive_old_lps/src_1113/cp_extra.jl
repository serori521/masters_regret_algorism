isdefined(Main, :SetRegretCore) || include("cp_regret_replace.jl")

module RegretRTR

using ..SetRegretCore

export RTRState,
       build_state!,
       build_state,
       step_RTR!,
       run_RTR!,
       get_current_MR_ranking,
       get_change_points,
       get_timeline

"""
高レベル状態：SetRegretCore の行列表現と重み区間を保持し，
run_lps の実行結果（変化点・タイムライン）をキャッシュする。
"""
mutable struct RTRState
    matrix::Array{SetRegretCore.minimax_regret_tuple,2}
    wL::Vector{Float64}
    wU::Vector{Float64}
    tL::Float64
    tR::Float64
    t::Float64
    Tchg::Vector{Float64}
    timeline::Vector{SetRegretCore.SnapshotEntry}
    solved::Bool
    function RTRState(matrix, wL, wU, tL, tR)
        return new(matrix, wL, wU, tL, tR, tR, Float64[],
                   SetRegretCore.SnapshotEntry[], false)
    end
end

"""
build_state!(utility, wL, wU)

- utility[p,i] = u_i(o_p)
- wL, wU は区間重み
"""
function build_state!(utility::Matrix{Float64},
                      wL::Vector{Float64},
                      wU::Vector{Float64})
    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    return RTRState(matrix, copy(wL), copy(wU), tL, tR)
end

# ノーバン版（Notebook 互換）
build_state(args...) = build_state!(args...)

# 内部ユーティリティ：run_lps を1回だけ実行
function run_RTR!(st::RTRState; force::Bool=false)
    if st.solved && !force
        return st.Tchg
    end
    result = SetRegretCore.run_lps(st.matrix, st.wL, st.wU, st.tL, st.tR)
    st.Tchg = copy(result.changes)
    st.timeline = result.timeline
    st.t = st.tL
    st.solved = true
    return st.Tchg
end

"""
step_RTR! はイベント単位の前進を模倣するが，
KDS実装では run_lps で一気に走査する。
"""
function step_RTR!(st::RTRState)
    run_RTR!(st)
    return st.t, :completed, Tuple{Int,Int}[]
end

"""
最新スナップショット（または timeline[idx]）の MR ベクトルと順位を返す。
"""
function get_current_MR_ranking(st::RTRState; idx::Union{Int,Nothing}=nothing)
    run_RTR!(st)
    ntl = st.timeline
    isempty(ntl) && return Float64[], Int[]
    i = isnothing(idx) ? length(ntl) : clamp(idx, 1, length(ntl))
    snap = ntl[i]
    return snap.MR, snap.rank
end

"""
変化点ログを返す。必要に応じて sort/unique は呼び出し側で行う。
"""
function get_change_points(st::RTRState)
    run_RTR!(st)
    return st.Tchg
end

"""
タイムライン全体（t, MR, rank, winners の列）を取得。
"""
function get_timeline(st::RTRState)
    run_RTR!(st)
    return st.timeline
end

end # module RegretRTR
