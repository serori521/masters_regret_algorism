module MinimaxRegretCore

export minimax_regret_tuple, find_optimal_trange, create_minimax_R_Matrix
export initialize_linear_models!, advance_TR_once!, max_regret_vector, ranking_from_MR

# [あなたのコードソース 2, 4-6]
mutable struct minimax_regret_tuple
    # --- 固定情報 ---
    difference_U::Vector{Float64}
    rank::Vector{Int}
    # --- 現在のリグレット ---
    regret::Float64
    # --- 旧calc_IPWの互換（参照用のみ） ---
    interm_index::Int
    Avail_space::Float64
    # --- 一次式と状態 ---
    full_count::Int
    partial_idx::Int
    slope::Float64
    intercept::Float64
    # --- キャッシュ ---
    sumL_all::Float64
    sum_diffL::Float64
    cumW::Vector{Float64}
    cumDiffW::Vector{Float64}
end

_minimax_empty() = minimax_regret_tuple(Float64[], Int[], 0.0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, Float64[], Float64[])

# [あなたのコードソース 1, 3]
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

# [あなたのコードソース 7-9]
function create_minimax_R_Matrix(utility::Matrix{Float64})
    A, N = size(utility)
    matrix = [_minimax_empty() for _ in 1:A, _ in 1:A]
    @inbounds for i in 1:A-1, j in i+1:A
        d_ij = vec(utility[j, :] .- utility[i, :])
        r_ij = sortperm(d_ij; rev=true)
        matrix[i, j].difference_U = d_ij
        matrix[i, j].rank = r_ij

        d_ji = -d_ij
        r_ji = sortperm(d_ji; rev=true)
        matrix[j, i].difference_U = d_ji
        matrix[j, i].rank = r_ji
    end
    return matrix
end

# [あなたのコードソース 10, 11]
function precompute_pair_caches!(cell::minimax_regret_tuple, L::Vector{Float64}, R::Vector{Float64})
    rank = cell.rank
    diff = cell.difference_U
    N = length(rank)

    cell.sumL_all = sum(L)
    width_base = R .- L
    w_rank = width_base[rank]
    cell.cumW = N == 0 ? Float64[] : cumsum(w_rank)

    d_rank = diff[rank] .* w_rank
    cell.cumDiffW = N == 0 ? Float64[] : cumsum(d_rank)

    cell.sum_diffL = sum(@inbounds(diff[k] * L[k]) for k in eachindex(L))
    return
end

# [あなたのコードソース 12-17]
function set_linear_model_for_pair!(cell::minimax_regret_tuple, L::Vector{Float64}, R::Vector{Float64}, t::Float64; eps::Float64=1e-12)
    rank = cell.rank
    C = length(rank)
    if C == 0
        cell.regret = 0.0
        cell.full_count = 0
        cell.partial_idx = 0
        cell.slope = 0.0
        cell.intercept = 0.0
        return
    end
    if isempty(cell.cumW)
        precompute_pair_caches!(cell, L, R)
    end

    sumL_all = cell.sumL_all
    width = (R .- L) .* t
    z = 1.0 - t * sumL_all
    if z < 0.0 && z > -eps
        z = 0.0
    end

    sumW_full = 0.0
    full_count = 0
    partial_idx = 0
    @inbounds for idx in 1:C
        k = rank[idx]
        w = width[k]
        if z > w + eps
            z -= w
            sumW_full += (R[k] - L[k])
            full_count += 1
            continue
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
            sumW_full -= (R[partial_idx] - L[partial_idx])
        end
    end

    m = full_count
    kstar = partial_idx
    B = cell.difference_U[kstar]
    A = cell.sum_diffL + (m == 0 ? 0.0 : cell.cumDiffW[m]) - B * (sumL_all + (m == 0 ? 0.0 : cell.cumW[m]))

    cell.full_count = m
    cell.partial_idx = kstar
    cell.slope = A
    cell.intercept = B
    cell.interm_index = kstar
    cell.Avail_space = 0.0
    cell.regret = A * t + B
    return
end

# [あなたのコードソース 18, 19]
@inline function boundary_t_right_cached(cell::minimax_regret_tuple)
    m = cell.full_count
    (isempty(cell.cumW) || m + 1 > length(cell.cumW)) && return -Inf
    return 1.0 / (cell.sumL_all + cell.cumW[m+1])
end

# [あなたのコードソース 20]
@inline function rebuild_slope!(cell::minimax_regret_tuple)
    m = cell.full_count
    kstar = cell.partial_idx
    B = cell.difference_U[kstar]
    cell.slope = cell.sum_diffL + (m == 0 ? 0.0 : cell.cumDiffW[m]) - B * (cell.sumL_all + (m == 0 ? 0.0 : cell.cumW[m]))
end

# [あなたのコードソース 21]
@inline function promote_right_once!(cell::minimax_regret_tuple, t_now::Float64)
    cell.full_count += 1
    m = cell.full_count
    r = cell.rank
    cell.partial_idx = m < length(r) ? r[m+1] : r[m]
    rebuild_slope!(cell)
    cell.intercept = cell.regret - cell.slope * t_now
    cell.regret = cell.slope * t_now + cell.intercept # 連続性を保証
end

# [あなたのコードソース 22]
function initialize_linear_models!(matrix::Array{minimax_regret_tuple,2}, L::Vector{Float64}, R::Vector{Float64}, tR::Float64)
    A = size(matrix, 1)
    @inbounds for i in 1:A, j in 1:A
        i == j && continue
        precompute_pair_caches!(matrix[i, j], L, R)
        set_linear_model_for_pair!(matrix[i, j], L, R, tR)
    end
end

# [あなたのコードソース 23-26]
function next_boundary_TR!(matrix::Array{minimax_regret_tuple,2}, t_cur::Float64, t_L::Float64; eps::Float64=1e-12)
    A = size(matrix, 1)
    t_next = t_L
    @inbounds for i in 1:A, j in 1:A
        i == j && continue
        tstar = boundary_t_right_cached(matrix[i, j])
        if (t_L + eps) < tstar < (t_cur - eps)
            if tstar > t_next + eps
                t_next = tstar
            end
        end
    end

    hit_pairs = Tuple{Int,Int}[]
    if t_next > t_L + eps
        @inbounds for i in 1:A, j in 1:A
            i == j && continue
            tstar = boundary_t_right_cached(matrix[i, j])
            if abs(tstar - t_next) <= 1e-10
                push!(hit_pairs, (i, j))
            end
        end
    end
    return t_next, hit_pairs
end

# [あなたのコードソース 27]
function advance_TR_once!(matrix::Array{minimax_regret_tuple,2}, t_cur::Float64, t_L::Float64)
    t_next, hit_pairs = next_boundary_TR!(matrix, t_cur, t_L)
    dt = t_next - t_cur
    @inbounds for i in 1:size(matrix, 1), j in 1:size(matrix, 2)
        i == j && continue
        matrix[i, j].regret += matrix[i, j].slope * dt
    end
    @inbounds for (i, j) in hit_pairs
        promote_right_once!(matrix[i, j], t_next)
    end
    return t_next, hit_pairs
end

# [あなたのコードソース 28, 29]
function max_regret_vector(matrix::Array{minimax_regret_tuple,2})
    A = size(matrix, 1)
    MR = fill(-Inf, A)
    @inbounds for p in 1:A
        mx = -Inf
        @inbounds for q in 1:A
            p == q && continue
            mx = max(mx, matrix[p, q].regret)
        end
        MR[p] = mx
    end
    return MR
end

@inline ranking_from_MR(MR::Vector{Float64}) = sortperm(MR)

end # module MinimaxRegretCore