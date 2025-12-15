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

###############################
# データ構造
###############################
mutable struct minimax_regret_tuple
    difference_U::Vector{Float64}   # diff_i = u_i(q)-u_i(p)
    rank::Vector{Int}               # diff降順のインデックス並び

    sumL_all::Float64               # Σ_i w_i^L
    sum_diffL::Float64              # Σ_i diff_i * w_i^L
    cumW::Vector{Float64}           # Σ_{s<=j} (w^U - w^L)[rank[s]]
    cumDiffW::Vector{Float64}       # Σ_{s<=j} diff[rank[s]]*(w^U - w^L)[rank[s]]

    full_count::Int                 # F のサイズ
    partial_idx::Int                # k* = rank[full_count+1]（実際の基準番号）
    slope::Float64                  # A_{p,q}
    intercept::Float64              # B_{p,q}
    tstar::Float64                  # 次の係数切替点 t*_{p,q}
end

function _minimax_empty()
    minimax_regret_tuple(
        Float64[], Int[],
        0.0, 0.0, Float64[], Float64[],
        0, 0, 0.0, 0.0, -Inf
    )
end


###############################
# 1. t の範囲
###############################
function find_optimal_trange(L::Vector{Float64}, R::Vector{Float64})
    max_sum = -Inf
    min_sum = Inf
    n = length(L)
    @inbounds for j in 1:n
        sum_ij_R = sum(L[i] for i in 1:n if i != j) + R[j]
        sum_ij_L = sum(R[i] for i in 1:n if i != j) + L[j]
        max_sum = max(max_sum, sum_ij_R)
        min_sum = min(min_sum, sum_ij_L)
    end
    t_R = 1 / max_sum
    t_L = 1 / min_sum
    return min(t_L, t_R), max(t_L, t_R)
end


###############################
# 2. 全ペアの差分ベクトルを用意
###############################
function create_minimax_R_Matrix(utility::Matrix{Float64})
    A, _ = size(utility)
    matrix = [_minimax_empty() for _ in 1:A, _ in 1:A]

    @inbounds for i in 1:A-1
        for j in i+1:A
            d_ij = vec(utility[j, :] .- utility[i, :])
            r_ij = sortperm(d_ij; rev=true)
            cell_ij = matrix[i, j]
            cell_ij.difference_U = d_ij
            cell_ij.rank = r_ij

            d_ji = -d_ij
            r_ji = sortperm(d_ji; rev=true)
            cell_ji = matrix[j, i]
            cell_ji.difference_U = d_ji
            cell_ji.rank = r_ji
        end
    end
    return matrix
end


###############################
# 3. 事前キャッシュ
###############################
function precompute_pair_caches!(
    cell::minimax_regret_tuple,
    wL::Vector{Float64}, wU::Vector{Float64}
)
    rank = cell.rank
    diff = cell.difference_U
    N = length(rank)

    cell.sumL_all = sum(wL)

    width_base = wU .- wL
    w_rank = width_base[rank]
    cell.cumW = N == 0 ? Float64[] : cumsum(w_rank)

    d_rank = diff[rank] .* w_rank
    cell.cumDiffW = N == 0 ? Float64[] : cumsum(d_rank)

    cell.sum_diffL = sum(@inbounds(diff[k] * wL[k]) for k in eachindex(wL))
    return
end


###############################
# 4. 線形モデル (A,B,t*) の構築
###############################
function set_linear_model_for_pair!(
    cell::minimax_regret_tuple,
    wL::Vector{Float64}, wU::Vector{Float64},
    t::Float64; eps::Float64=EPS_DEFAULT
)
    rank = cell.rank
    diff = cell.difference_U
    C = length(rank)

    if C == 0
        cell.full_count = 0
        cell.partial_idx = 0
        cell.slope = 0.0
        cell.intercept = 0.0
        cell.tstar = -Inf
        return
    end

    if isempty(cell.cumW)
        precompute_pair_caches!(cell, wL, wU)
    end

    width_each = (wU .- wL) .* t
    sumL_all = cell.sumL_all
    z = 1.0 - t * sumL_all
    if z < 0.0 && z > -eps
        z = 0.0
    end

    sumW_full = 0.0
    full_count = 0
    partial_idx = 0

    @inbounds for idx in 1:C
        k = rank[idx]
        wuse = width_each[k]
        if z > wuse + eps
            z -= wuse
            sumW_full += (wU[k] - wL[k])
            full_count += 1
        else
            partial_idx = k
            break
        end
    end

    if partial_idx == 0
        if full_count == 0
            partial_idx = rank[1]
        else
            partial_idx = rank[full_count]
            full_count -= 1
        end
    end

    m = full_count
    kstar = partial_idx
    B = diff[kstar]

    sumDiffFull = m == 0 ? 0.0 : cell.cumDiffW[m]
    sumWidthFull = m == 0 ? 0.0 : cell.cumW[m]
    A_val = cell.sum_diffL + sumDiffFull - diff[kstar] * (sumL_all + sumWidthFull)

    cell.full_count = m
    cell.partial_idx = kstar
    cell.slope = A_val
    cell.intercept = B
    cell.tstar = boundary_t_right_cached(cell)
    return
end

@inline function boundary_t_right_cached(cell::minimax_regret_tuple)
    m = cell.full_count
    if isempty(cell.cumW) || m + 1 > length(cell.cumW)
        return -Inf
    end
    return 1.0 / (cell.sumL_all + cell.cumW[m+1])
end


###############################
# 5. 初期化
###############################
function initialize_linear_models!(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    tR::Float64; eps::Float64=EPS_DEFAULT
)
    A = size(matrix, 1)
    @inbounds for i in 1:A, j in 1:A
        if i == j
            continue
        end
        precompute_pair_caches!(matrix[i, j], wL, wU)
        set_linear_model_for_pair!(matrix[i, j], wL, wU, tR; eps=eps)
    end
    return
end


###############################
# 6. 評価と順位
###############################
@inline function evaluate_regret(cell::minimax_regret_tuple, t::Float64)
    return cell.slope * t + cell.intercept
end

function max_regret_vector(matrix::Array{minimax_regret_tuple,2}, t::Float64)
    A = size(matrix, 1)
    MR = fill(-Inf, A)
    @inbounds for p in 1:A
        best = -Inf
        for q in 1:A
            if p == q
                continue
            end
            val = evaluate_regret(matrix[p, q], t)
            if val > best
                best = val
            end
        end
        MR[p] = best
    end
    return MR
end

@inline function ranking_from_MR(MR::Vector{Float64})
    return sortperm(MR)
end


###############################
# 7. 内部ユーティリティ
###############################
function argmax_regret_index(
    matrix::Array{minimax_regret_tuple,2},
    p::Int, t::Float64; preferred::Int=0, eps::Float64=EPS_DEFAULT
)
    best_q = 0
    best_val = -Inf
    A = size(matrix, 1)
    @inbounds for q in 1:A
        if q == p
            continue
        end
        val = evaluate_regret(matrix[p, q], t)
        if val > best_val + eps ||
           (abs(val - best_val) <= eps && q == preferred)
            best_val = val
            best_q = q
        end
    end
    return best_q
end

function find_inner_crossing(
    matrix::Array{minimax_regret_tuple,2},
    p::Int, qstar::Int,
    t_min::Float64, t_max::Float64;
    eps::Float64=EPS_DEFAULT
)
    if qstar == 0
        return 0, t_min
    end
    A = size(matrix, 1)
    line_star = matrix[p, qstar]
    Astar = line_star.slope
    Bstar = line_star.intercept

    best_x = t_min
    best_q = 0

    @inbounds for q in 1:A
        if q == p || q == qstar
            continue
        end
        line_q = matrix[p, q]
        Adelta = Astar - line_q.slope
        if Adelta <= eps
            continue
        end
        x = (line_q.intercept - Bstar) / Adelta
        lower = maximum((t_min, line_star.tstar, line_q.tstar))
        if lower <= x && x <= t_max - eps && x > best_x + eps
            best_x = x
            best_q = q
        end
    end

    if best_q == 0
        return 0, t_min
    end
    return best_q, best_x
end

function collect_outer_changes(
    matrix::Array{minimax_regret_tuple,2},
    qstar::Vector{Int},
    x_p_max::Vector{Float64},
    t_min::Float64, t_max::Float64;
    eps::Float64=EPS_DEFAULT
)
    A = length(qstar)
    changes = Float64[]

    @inbounds for p1 in 1:A-1
        for p2 in p1+1:A
            q1 = qstar[p1]
            q2 = qstar[p2]
            if q1 == 0 || q2 == 0
                continue
            end

            line1 = matrix[p1, q1]
            line2 = matrix[p2, q2]
            Adelta = line1.slope - line2.slope
            if abs(Adelta) <= eps
                continue
            end
            x = (line2.intercept - line1.intercept) / Adelta
            lower = maximum((t_min, line1.tstar, line2.tstar, x_p_max[p1], x_p_max[p2]))
            if lower <= x && x <= t_max - eps
                push!(changes, x)
            end
        end
    end

    sort!(changes; rev=true)
    return changes
end

function next_coefficient_event(
    matrix::Array{minimax_regret_tuple,2},
    t_L::Float64, t_cur::Float64;
    eps::Float64=EPS_DEFAULT
)
    best = t_L
    pairs = Tuple{Int,Int}[]
    A = size(matrix, 1)

    @inbounds for i in 1:A, j in 1:A
        if i == j
            continue
        end
        tstar = matrix[i, j].tstar
        if !(t_L + eps < tstar < t_cur - eps)
            continue
        end
        if tstar > best + eps
            best = tstar
            empty!(pairs)
            push!(pairs, (i, j))
        elseif abs(tstar - best) <= eps
            push!(pairs, (i, j))
        end
    end

    return best, pairs
end

function next_inner_event(
    x_p_max::Vector{Float64},
    t_L::Float64, t_cur::Float64;
    eps::Float64=EPS_DEFAULT
)
    best = t_L
    idxs = Int[]
    @inbounds for (p, x) in enumerate(x_p_max)
        if !(t_L + eps < x < t_cur - eps)
            continue
        end
        if x > best + eps
            best = x
            empty!(idxs)
            push!(idxs, p)
        elseif abs(x - best) <= eps
            push!(idxs, p)
        end
    end
    return best, idxs
end

function snapshot_state(
    matrix::Array{minimax_regret_tuple,2},
    qstar::Vector{Int},
    t::Float64;
    eps::Float64=EPS_DEFAULT
)
    A = length(qstar)
    MR = Vector{Float64}(undef, A)
    @inbounds for p in 1:A
        q = qstar[p]
        MR[p] = q == 0 ? -Inf : evaluate_regret(matrix[p, q], t)
    end
    winners = findall(x -> x <= minimum(MR) + eps, MR)
    return (t=t, MR=MR, rank=ranking_from_MR(copy(MR)), winners=Vector{Int}(winners))
end

function push_snapshot!(changes::Vector{Float64}, timeline::Vector{SnapshotEntry},
                        matrix::Array{minimax_regret_tuple,2}, qstar::Vector{Int},
                        t::Float64; eps::Float64=EPS_DEFAULT, detect_change::Bool=false)
    snap = snapshot_state(matrix, qstar, t; eps=eps)
    if detect_change && !isempty(timeline)
        prev = timeline[end]
        if prev.rank != snap.rank
            duplicated = any(abs(tc - t) <= eps for tc in changes)
            duplicated || push!(changes, t)
        end
    end
    push!(timeline, snap)
end
function refresh_inner_state!(
    matrix::Array{minimax_regret_tuple,2},
    p::Int,
    t::Float64,
    t_L::Float64,
    qstar::Vector{Int},
    hat_q::Vector{Int},
    x_p_max::Vector{Float64};
    eps::Float64=EPS_DEFAULT,
    preferred::Int=0
)
    preferred = preferred == 0 ? qstar[p] : preferred
    best = argmax_regret_index(matrix, p, t; preferred=preferred, eps=eps)
    qstar[p] = best
    challenger, xmax = find_inner_crossing(matrix, p, best, t_L, t; eps=eps)
    hat_q[p] = challenger
    x_p_max[p] = challenger == 0 ? t_L : xmax
end


###############################
# 8. KDSベース左向き走査メインループ
###############################
function run_lps(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    t_L::Float64, t_U::Float64;
    eps::Float64=EPS_DEFAULT
)
    initialize_linear_models!(matrix, wL, wU, t_U; eps=eps)

    A = size(matrix, 1)
    qstar = zeros(Int, A)
    hat_q = zeros(Int, A)
    x_p_max = fill(t_L, A)

    @inbounds for p in 1:A
        qstar[p] = argmax_regret_index(matrix, p, t_U; eps=eps)
        h, xp = find_inner_crossing(matrix, p, qstar[p], t_L, t_U; eps=eps)
        hat_q[p] = h
        x_p_max[p] = h == 0 ? t_L : xp
    end

    Tchg = Float64[]
    timeline = SnapshotEntry[]
    push_snapshot!(Tchg, timeline, matrix, qstar, t_U; eps=eps, detect_change=false)

    t = t_U
    while t > t_L + eps
        E1, active_pairs = next_coefficient_event(matrix, t_L, t; eps=eps)
        E2, inner_indices = next_inner_event(x_p_max, t_L, t; eps=eps)
        t_next = max(max(E1, E2), t_L)

        if t_next >= t - eps
            break
        end

        S = collect_outer_changes(matrix, qstar, x_p_max, t_next, t; eps=eps)
        for x in S
            push!(Tchg, x)
            push_snapshot!(Tchg, timeline, matrix, qstar, x; eps=eps, detect_change=false)
        end

        affected = Int[]

        if !isempty(active_pairs) && abs(t_next - E1) <= eps
            for (p, q) in active_pairs
                set_linear_model_for_pair!(matrix[p, q], wL, wU, t_next; eps=eps)
                push!(affected, p)
            end
        end

        if !isempty(inner_indices) && abs(t_next - E2) <= eps
            for p in inner_indices
                if hat_q[p] != 0
                    qstar[p] = hat_q[p]
                end
                push!(affected, p)
            end
        end

        t = t_next

        if isempty(affected)
            # 到達可能なイベントがなかった場合は終了
            break
        end

        unique!(affected)
        for p in affected
            refresh_inner_state!(matrix, p, t, t_L, qstar, hat_q, x_p_max;
                                 eps=eps, preferred=qstar[p])
        end

        push_snapshot!(Tchg, timeline, matrix, qstar, t; eps=eps, detect_change=true)
    end

    return (changes=Tchg, timeline=timeline)
end

end # module
