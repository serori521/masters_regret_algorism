###############################
# set_regret.jl  (右→左 専用最適化版)
#
# 変更点（ハイライト）
# - ✅ R_{p,q}(t) を“能動集合”が不変の区間で一次式 R=A*t+B として保持
# - ✅ Δt だけ差分更新：regret += slope * Δt （全ペア O(n^2) の足し算だけ）
# - ✅ 境界 t* は O(1)：t* = 1 / (ΣL + cumW[m+1])（キャッシュ使用）
# - ✅ 境界ヒット時の再構成も O(1)：
#      「部分 k* がフルになる → full_count=m→m+1, k*←rank[m+1]」を
#      promote_right_once! で 1 ステップだけ進め、slopeを再構成
# - ✅ 右→左（TR→TL）運用に限定（左→右は別式が必要のため非対応）
###############################

#############
# 1. t 範囲（元calc_IPW）
#############
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

#############
# 2. データ構造（calc_IPWのtupleを拡張）
#############
mutable struct minimax_regret_tuple
    # --- 固定情報 ---
    difference_U::Vector{Float64}   # diff (N)
    rank::Vector{Int}               # diff降順の添字 (貪欲順)

    # --- 現在のリグレット ---
    regret::Float64                 # R_{p,q}(t)

    # --- 旧calc_IPWの互換（参照用のみ） ---
    interm_index::Int               # 部分割当の基準インデックス（元）
    Avail_space::Float64            # 未使用（右→左では使わない）

    # --- 一次式と状態（能動集合が不変の間は一定） ---
    full_count::Int                 # F のサイズ (rankの先頭から full_count 個がフル)
    partial_idx::Int                # k* = rank[full_count+1]
    slope::Float64                  # A (傾き)
    intercept::Float64              # B (切片 = diff[k*])

    # --- キャッシュ（O(1)化の要） ---
    sumL_all::Float64              # Σ L_i
    sum_diffL::Float64             # Σ diff_i * L_i
    cumW::Vector{Float64}          # cumW[j] = Σ_{s<=j} (R-L)[rank[s]]
    cumDiffW::Vector{Float64}      # cumDiffW[j] = Σ_{s<=j} diff[rank[s]]*(R-L)[rank[s]]
end

_minimax_empty() = minimax_regret_tuple(Float64[], Int[],
    0.0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, Float64[], Float64[])

#############
# 3. 差分ベクトルの前計算（一般化）
#    上三角/下三角の diff を同時設定（rank は各方向で降順）
#############
function create_minimax_R_Matrix(utility::Matrix{Float64})
    A, N = size(utility)
    matrix = [_minimax_empty() for _ in 1:A, _ in 1:A]
    @inbounds for i in 1:A-1
        for j in i+1:A
            d_ij = vec(utility[j, :] .- utility[i, :])
            r_ij = sortperm(d_ij; rev=true)
            matrix[i, j].difference_U = d_ij
            matrix[i, j].rank = r_ij

            d_ji = -d_ij
            r_ji = sortperm(d_ji; rev=true)
            matrix[j, i].difference_U = d_ji
            matrix[j, i].rank = r_ji
        end
    end
    return matrix
end

#############
# 4. ペアごとのキャッシュをセット（1回だけ）
#    sumL_all, sum_diffL, cumW, cumDiffW
#############
function precompute_pair_caches!(cell::minimax_regret_tuple, L::Vector{Float64}, R::Vector{Float64})
    rank = cell.rank
    diff = cell.difference_U
    N = length(rank)

    cell.sumL_all = sum(L)
    width_base = R .- L  # t に依らない幅係数
    w_rank = width_base[rank]
    cell.cumW = N == 0 ? Float64[] : cumsum(w_rank)

    d_rank = diff[rank] .* w_rank
    cell.cumDiffW = N == 0 ? Float64[] : cumsum(d_rank)

    cell.sum_diffL = sum(@inbounds(diff[k] * L[k]) for k in eachindex(L))
    return
end

#############
# 5. 線形モデル(A,B)の初期設定（t=TR など区間端で1回）
#    Greedy を解析的に行い、F と k* を決定 → A,B を構成 → regret=A*t+B
#############
function set_linear_model_for_pair!(
    cell::minimax_regret_tuple,
    L::Vector{Float64}, R::Vector{Float64},
    t::Float64; eps::Float64=1e-12
)
    rank = cell.rank
    diff = cell.difference_U
    C = length(rank)
    if C == 0
        cell.regret = 0.0
        cell.full_count = 0
        cell.partial_idx = 0
        cell.slope = 0.0
        cell.intercept = 0.0
        return
    end

    # キャッシュ前提（未セットならセット）
    if isempty(cell.cumW)
        precompute_pair_caches!(cell, L, R)
    end

    sumL_all = cell.sumL_all
    width = (R .- L) .* t
    z = 1.0 - t * sumL_all
    if z < 0.0 && z > -eps
        z = 0.0
    end

    # F を rank 先頭から貪欲に構築
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

    # 退化：部分が無い場合は直前を“部分扱い”に倒す
    if partial_idx == 0
        if full_count == 0
            partial_idx = rank[1]
        else
            partial_idx = rank[full_count]
            full_count -= 1
            sumW_full -= (R[partial_idx] - L[partial_idx])
        end
    end

    # A,B 構成（cumW/cumDiffW/sum_diffL を使用し O(1) ）
    m = full_count
    kstar = partial_idx
    B = diff[kstar]
    A =
        cell.sum_diffL +
        (m == 0 ? 0.0 : cell.cumDiffW[m]) -
        diff[kstar] * (sumL_all + (m == 0 ? 0.0 : cell.cumW[m]))

    # セット
    cell.full_count = m
    cell.partial_idx = kstar
    cell.slope = A
    cell.intercept = B
    cell.interm_index = kstar
    cell.Avail_space = 0.0 # 右→左では使わない
    cell.regret = A * t + B
    return
end

#############
# 6. 右→左の区間境界 t*（O(1) キャッシュ版）
#    今が m 個フル＋k*=rank[m+1] のとき：t* = 1 / (ΣL + cumW[m+1])
#############
@inline function boundary_t_right_cached(cell::minimax_regret_tuple)
    m = cell.full_count
    if isempty(cell.cumW) || m + 1 > length(cell.cumW)
        return -Inf   # これ以上右→左の境界なし扱い
    end
    return 1.0 / (cell.sumL_all + cell.cumW[m+1])
end

#############
# 7. 差分更新（区間内は一次式）
#############
@inline function update_regret_by_dt!(cell::minimax_regret_tuple, dt::Float64)
    cell.regret += cell.slope * dt
end

#############
# 8. 傾きの再構成 A（O(1)）
#############
@inline function rebuild_slope!(cell::minimax_regret_tuple)
    m = cell.full_count
    kstar = cell.partial_idx
    cell.slope =
        cell.sum_diffL +
        (m == 0 ? 0.0 : cell.cumDiffW[m]) -
        cell.difference_U[kstar] * (cell.sumL_all + (m == 0 ? 0.0 : cell.cumW[m]))
end

#############
# 9. 境界ヒット時の「1個だけ進める」更新（O(1)）
#    k* がフルになる → m←m+1, k*←rank[m+1]、A を再構成
#    B は理論値 diff[new_k] だが、連続性のため現在値から補正も許容
#############
@inline function promote_right_once!(cell::minimax_regret_tuple, t_now::Float64)
    cell.full_count += 1
    m = cell.full_count
    r = cell.rank

    # 次の partial 候補
    if m < length(r)
        new_k = r[m+1]
        cell.partial_idx = new_k
    else
        cell.partial_idx = r[m]  # 末尾に到達
    end

    # 新しい slope を再構成
    rebuild_slope!(cell)

    # ★重要：切片は常に「連続性」で決める（つじつま合わせではなく連続性の必然）
    cell.intercept = cell.regret - cell.slope * t_now

    # いまの t_now における regret は連続のはずなので、念のため合わせる
    cell.regret = cell.slope * t_now + cell.intercept
end


#############
# 10. 初期化：t=TR で全ペアの線形モデルを確定（キャッシュ込み）
#############
function initialize_linear_models!(
    matrix::Array{minimax_regret_tuple,2},
    L::Vector{Float64}, R::Vector{Float64},
    tR::Float64
)
    A = size(matrix, 1)
    @inbounds for i in 1:A, j in 1:A
        if i == j
            continue
        end
        precompute_pair_caches!(matrix[i, j], L, R)
        set_linear_model_for_pair!(matrix[i, j], L, R, tR)
    end
    return
end

#############
# 11. 次の境界（右→左）：全ペアの t* を走査し、一番近い左側へジャンプ
#############
function next_boundary_TR!(
    matrix::Array{minimax_regret_tuple,2},
    t_cur::Float64, t_L::Float64; eps::Float64=1e-12
)
    A = size(matrix, 1)
    t_next = t_L
    hit_pairs = Tuple{Int,Int}[]

    # 候補選定（t_L < t* < t_cur の最大）
    @inbounds for i in 1:A, j in 1:A
        if i == j
            continue
        end
        tstar = boundary_t_right_cached(matrix[i, j])
        if (t_L + eps) < tstar < (t_cur - eps)
            if tstar > t_next + eps
                t_next = tstar
            end
        end
    end

    # 同時ヒット収集（許容誤差内）
    if t_next > t_L + eps
        @inbounds for i in 1:A, j in 1:A
            if i == j
                continue
            end
            tstar = boundary_t_right_cached(matrix[i, j])
            if abs(tstar - t_next) <= 1e-10
                push!(hit_pairs, (i, j))
            end
        end
    end

    return t_next, hit_pairs
end

#############
# 12. 1ステップ前進（TR→TL）：差分更新→ヒットだけ1段昇格
#############
function advance_TR_once!(
    matrix::Array{minimax_regret_tuple,2},
    t_cur::Float64, t_L::Float64
)
    t_next, hit_pairs = next_boundary_TR!(matrix, t_cur, t_L)

    # 全ペア差分更新（区間内は能動集合不変）
    dt = t_next - t_cur  # (<0)
    @inbounds for i in 1:size(matrix, 1), j in 1:size(matrix, 2)
        if i == j
            continue
        end
        update_regret_by_dt!(matrix[i, j], dt)
    end

    # 境界ヒットのみ O(1) 状態更新
    @inbounds for (i, j) in hit_pairs
        promote_right_once!(matrix[i, j], t_next)
    end

    return t_next, hit_pairs
end

#############
# 13. MR とランキング（外側の“上の線”用）
#############
function max_regret_vector(matrix::Array{minimax_regret_tuple,2})
    A = size(matrix, 1)
    MR = fill(-Inf, A)
    @inbounds for p in 1:A
        mx = -Inf
        @inbounds for q in 1:A
            if p == q
                continue
            end
            rv = matrix[p, q].regret
            if rv > mx
                mx = rv
            end
        end
        MR[p] = mx
    end
    return MR
end

@inline function ranking_from_MR(MR::Vector{Float64})
    return sortperm(MR)  # MR小さい順（1位が先頭）
end

#############
# 14. 使い方（右→左に限定）
#
#   tL,tR = find_optimal_trange(L,R)
#   M = create_minimax_R_Matrix(utility)
#   initialize_linear_models!(M, L, R, tR)
#   t = tR
#   while t > tL + 1e-12
#       t, hits = advance_TR_once!(M, t, tL)
#       MR = max_regret_vector(M)
#       rk = ranking_from_MR(MR)
#       # 必要なら記録…
#   end
#
# ※ 左→右に進める場合は境界式が異なるため、本ファイルでは非対応。
#############
