module RegretRTR
# 右→左（TR→TL）で Minimax Regret を追跡するモジュール
# - include("set_regret.jl") の差分更新＆境界O(1)更新を活用
# - 外側：全体の順位変化を収集
# - 内側：各 p の最大リグレット相手の交点（区間内）も解析計算で収集

export RTRState,
       build_state, advance!, run!,
       current_ranking, change_points, rankings_at,
       best_opponents, max_regret_vector, ranking_from_MR,
       save_rank_changes_csv

include("set_regret.jl")  # 同ディレクトリの最新版 set_regret.jl を取り込み

# =============== 状態 ===============

mutable struct RTRState
    matrix::Array{minimax_regret_tuple,2}
    L::Vector{Float64}
    R::Vector{Float64}
    tL::Float64
    tR::Float64
    t::Float64
    cps::Vector{Float64}                  # 外側・内側ともに「評価すべきt」を溜める
    rks::Dict{Float64,Vector{Int}}        # tごとの全体ランキング
end

# =============== ビルド/初期化 ===============

function build_state(utility::AbstractMatrix{<:Real},
                     L::AbstractVector{<:Real},
                     R::AbstractVector{<:Real})::RTRState
    U  = Matrix{Float64}(utility)
    Lf = collect(Float64, L)
    Rf = collect(Float64, R)
    tL, tR = find_optimal_trange(Lf, Rf)

    M = create_minimax_R_Matrix(U)
    initialize_linear_models!(M, Lf, Rf, tR)

    MR0 = max_regret_vector(M)
    rk0 = ranking_from_MR(MR0)

    return RTRState(M, Lf, Rf, tL, tR, tR, [tR], Dict(tR => rk0))
end

# =============== ユーティリティ（“内側”計算用のスナップショット評価） ===============

# 区間内での「現在ペアの一次式」を使って、t における R_{p,q}(t) を読み出す（非破壊）
@inline regret_at(cell::minimax_regret_tuple, t::Float64) = cell.slope * t + cell.intercept

# t における MR ベクトルを（状態を変えずに）評価
function max_regret_vector_at(matrix::Array{minimax_regret_tuple,2}, t::Float64)
    A = size(matrix,1)
    MR = fill(-Inf, A)
    @inbounds for p in 1:A
        m = -Inf
        for q in 1:A
            if p == q; continue; end
            v = regret_at(matrix[p,q], t)
            if v > m; m = v; end
        end
        MR[p] = m
    end
    return MR
end

# t における “p の最大リグレット相手” を非破壊で返す
@inline function best_opponent_at(matrix::Array{minimax_regret_tuple,2}, p::Int, t::Float64)
    A = size(matrix,1)
    best_q, best_v = 0, -Inf
    @inbounds for q in 1:A
        if q == p; continue; end
        v = regret_at(matrix[p,q], t)
        if v > best_v
            best_v = v; best_q = q
        end
    end
    return best_q
end

# =============== 内側：区間 (t_left, t_right] で起きる「各 p の相手交代」を収集 ===============

"""
collect_inner_crossings!(st, t_left, t_right; atol=1e-12)

- 現在の一次式（A,B）は区間内で不変と仮定（外側の境界でしか変わらない）
- 各 p について、t_right 時点の勝者 q* と他の q の直線交点を解析計算：
    t_cross = (B_q - B_q*) / (A_q* - A_q)
  で、t_left < t_cross ≤ t_right のものを候補に採用
- すべての候補 t を一つの集合にまとめて降順（右→左）に並べ、
  その時点での“全体ランキング”を再評価して記録
"""
function collect_inner_crossings!(st::RTRState, t_left::Float64, t_right::Float64; atol=1e-12)
    A = size(st.matrix,1)
    t_candidates = Float64[]

    # 各 p について候補交点を集める
    @inbounds for p in 1:A
        q_star = best_opponent_at(st.matrix, p, t_right)
        Astar  = st.matrix[p,q_star].slope
        Bstar  = st.matrix[p,q_star].intercept
        for q in 1:A
            if q == p || q == q_star; continue; end
            Aq = st.matrix[p,q].slope
            Bq = st.matrix[p,q].intercept
            denom = (Astar - Aq)
            if abs(denom) <= atol
                continue # 平行（交点なし）
            end
            t_cross = (Bq - Bstar) / denom
            if (t_left + atol) < t_cross && t_cross <= (t_right + atol)
                push!(t_candidates, t_cross)
            end
        end
    end

    if isempty(t_candidates); return; end

    # 重複・近接をまとめ、右→左で処理
    sort!(t_candidates; rev=true)
    unique_t = Float64[]
    lastp = Inf
    for tc in t_candidates
        if isempty(unique_t) || abs(tc - lastp) > 1e-10
            push!(unique_t, tc)
            lastp = tc
        end
    end

    # それぞれの t_cross で全体ランキングを“非破壊に”再評価して記録
    for tc in unique_t
        MRtc = max_regret_vector_at(st.matrix, tc)
        rk   = ranking_from_MR(MRtc)
        # 直前の記録と違うときだけ追加
        last_rk = st.rks[st.cps[end]]
        if rk != last_rk
            push!(st.cps, tc)
            st.rks[tc] = rk
        end
    end
end

# =============== 1 ステップ前進（外側差分＋内側交点の記録） ===============

"""
advance!(st) -> (t_next, hit_pairs)

流れ：
 1) 次の外側境界 t_next を決める（set_regret.jl）
 2) 区間 (t_next, t_prev] の中で起きる「内側の相手交代」を解析計算で収集（非破壊）
 3) 全ペアを差分更新（Δt）し、境界ヒットだけ O(1) 再構成
 4) t_next での全体ランキングを比較して変化があれば記録
"""
function advance!(st::RTRState)
    t_prev = st.t
    # 1) 先に次境界を見ておく（区間端を確定）
    t_next, hits = next_boundary_TR!(st.matrix, t_prev, st.tL)

    # 2) 区間内の“内側”交点を記録（状態を変えずに評価）
    collect_inner_crossings!(st, t_next, t_prev)

    # 3) 差分更新 → ヒットのみ1段昇格
    dt = t_next - t_prev
    @inbounds for i in 1:size(st.matrix,1), j in 1:size(st.matrix,2)
        if i == j; continue; end
        update_regret_by_dt!(st.matrix[i,j], dt)
    end
    @inbounds for (i,j) in hits
        promote_right_once!(st.matrix[i,j], t_next)
    end

    # 4) 境界点 t_next での外側ランキングを記録（必要なら）
    MR = max_regret_vector(st.matrix)
    rk = ranking_from_MR(MR)
    last_rk = st.rks[st.cps[end]]
    if rk != last_rk
        push!(st.cps, t_next)
        st.rks[t_next] = rk
    end

    st.t = t_next
    return t_next, hits
end

# =============== 全区間実行 ===============

function run!(st::RTRState; tol::Float64=1e-12)
    while st.t > st.tL + tol
        advance!(st)
    end
    return st
end

# =============== ビュー/ユーティリティ ===============

current_ranking(st::RTRState) = st.rks[st.cps[end]]
change_points(st::RTRState) = st.cps
rankings_at(st::RTRState, t::Float64) = get(st.rks, t, nothing)

function best_opponents(matrix::Array{minimax_regret_tuple,2})
    A = size(matrix,1)
    res = fill(0, A)
    @inbounds for p in 1:A
        best_q, best_v = 0, -Inf
        for q in 1:A
            if q == p; continue; end
            v = matrix[p,q].regret
            if v > best_v
                best_v = v; best_q = q
            end
        end
        res[p] = best_q
    end
    return res
end

const max_regret_vector = RegretRTR.max_regret_vector
const ranking_from_MR   = RegretRTR.ranking_from_MR

# =============== CSV 保存（任意） ===============

using DelimitedFiles

function save_rank_changes_csv(st::RTRState, path::AbstractString)
    points = sort(unique(st.cps))  # 念のため昇順化して保存してもOK
    if isempty(points)
        writedlm(path, ["t_change_point"])
        return path
    end
    A = length(st.rks[points[end]])
    header = ["t_change_point"; ["rank_$(i)" for i in 1:A]...]
    open(path, "w") do io
        println(io, join(header, ','))
        for t in points
            rk = st.rks[t]
            row = [t; rk...]
            println(io, join(row, ','))
        end
    end
    return path
end

end # module
