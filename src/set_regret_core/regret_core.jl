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
    partial_idx::Int                # k* = rank[full_count+1]
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

    full_count = 0
    partial_idx = 0

    @inbounds for idx in 1:C
        k = rank[idx]
        wuse = width_each[k]
        if z > wuse + eps
            z -= wuse
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
        i == j && continue
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
            p == q && continue
            val = evaluate_regret(matrix[p, q], t)
            best = max(best, val)
        end
        MR[p] = best
    end
    return MR
end

@inline function ranking_from_MR(MR::Vector{Float64})
    return sortperm(MR)
end


###############################
# 7. 内部ユーティリティ（コア）
###############################
function argmax_regret_index(
    matrix::Array{minimax_regret_tuple,2},
    p::Int, t::Float64; preferred::Int=0, eps::Float64=EPS_DEFAULT
)
    best_q = 0
    best_val = -Inf
    A = size(matrix, 1)
    @inbounds for q in 1:A
        q == p && continue
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
    qstar == 0 && return 0, t_min

    A = size(matrix, 1)
    line_star = matrix[p, qstar]
    Astar = line_star.slope
    Bstar = line_star.intercept

    best_x = t_min
    best_q = 0

    @inbounds for q in 1:A
        (q == p || q == qstar) && continue
        line_q = matrix[p, q]
        Adelta = Astar - line_q.slope
        Adelta <= eps && continue

        x = (line_q.intercept - Bstar) / Adelta
        lower = maximum((t_min, line_star.tstar, line_q.tstar))
        if lower <= x && x <= t_max - eps && x > best_x + eps
            best_x = x
            best_q = q
        end
    end

    best_q == 0 && return 0, t_min
    return best_q, best_x
end
