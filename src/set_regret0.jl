###############################
# set_regret.jl
#  - Minimax Regret の貪欲割当を「区分線形」化し、
#    1) 能動集合(Full+Partial)が不変の区間では R_{p,q}(t)=A*t + B を保持
#    2) Δt だけ slope を掛けて O(1) 更新（差分更新）
#    3) 能動集合が変わる境界 t* だけ再構成
#  - 元calc_IPWのうち必要な部品(ト範囲・差分ベクトル前計算)＋今回の追加で構成
###############################

##########
# 1. t 範囲の導出（元calc_IPW）
##########
function find_optimal_trange(L::Vector{Float64}, R::Vector{Float64})
    max_sum = -Inf
    min_sum =  Inf
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


##########
# 2. データ構造（calc_IPWの tuple を拡張）
##########
mutable struct minimax_regret_tuple
    # --- 固有差分（固定） ---
    difference_U::Vector{Float64}   # 各基準の効用差 diff (サイズ = N)
    rank::Vector{Int}               # diff の降順インデックス（貪欲の適用順）

    # --- 現在のリグレット値（更新される） ---
    regret::Float64                 # 現在の R_{p,q}(t)

    # --- 旧calc_IPWの記録（互換/参考） ---
    interm_index::Int               # 部分充填の基準インデックス（rank参照ベース）
    Avail_space::Float64            # 旧: 残余幅（参考値）

    # --- 追加：区分線形モデルの状態（能動集合が不変の間は一定） ---
    full_count::Int                 # 完全に幅を使い切った基準の数（rank の先頭から full_count）
    partial_idx::Int                # 部分充填されている“ただ1つの基準”の元インデックス
    slope::Float64                  # A（傾き）  : R(t) = A*t + B
    intercept::Float64              # B（切片）  : = diff[partial_idx]
    sumL_all::Float64               # Σ_i L_i
    sumW_full::Float64              # Σ_{f∈F} (R_f - L_f)
end

# 空セルを作る（行列初期化用）
_minimax_empty() = minimax_regret_tuple(Float64[], Int[], 0.0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0)


##########
# 3. 差分ベクトル前計算（元calc_IPWの create_minimax_R_Matrix を一般化）
#    - 上三角だけでなく下三角も同時に埋める（diffとrankを対称に持たせる）
##########
function create_minimax_R_Matrix(utility::Matrix{Float64})
    A, N = size(utility)  # 代替案数×基準数
    matrix = [_minimax_empty() for _ in 1:A, _ in 1:A]

    @inbounds for i in 1:A-1
        for j in i+1:A
            diff_ij = vec(utility[j, :] .- utility[i, :])
            rk_ij   = sortperm(diff_ij; rev=true)

            # (i,j)
            matrix[i,j].difference_U = diff_ij
            matrix[i,j].rank         = rk_ij

            # (j,i) は符号を反転
            diff_ji = -diff_ij
            rk_ji   = sortperm(diff_ji; rev=true)

            matrix[j,i].difference_U = diff_ji
            matrix[j,i].rank         = rk_ji
        end
    end

    # 対角は 0 のまま（diff/rank は空）。必要なら任意に初期化。
    return matrix
end


##########
# 4. 能動集合を決めて線形モデル(A,B)を設定（今回の追加の要）
#    - t における Greedy 割当を一回“解析的”に行い、F/k* と A,B を確定する
#    - cell.regret も A*t+B で同期
##########
function set_linear_model_for_pair!(
    cell::minimax_regret_tuple,
    L::Vector{Float64}, R::Vector{Float64},
    t::Float64; eps::Float64=1e-12
)
    rank = cell.rank
    diff = cell.difference_U
    C = length(rank)

    # 正規性：Y_L = L*t, width = (R-L)*t, 残余 z0 = 1 - t*ΣL
    sumL_all = sum(L)
    width = (R .- L) .* t
    z = 1.0 - t*sumL_all

    # 端の数値揺れ対策
    if z < 0.0 && z > -eps
        z = 0.0
    end

    # 完全充填集合 F を rank の先頭から構築
    sumW_full = 0.0
    full_count = 0
    partial_idx = 0

    @inbounds for idx in 1:C
        k = rank[idx]
        w = width[k]
        if z > w + eps
            # 完全に幅を使う
            z -= w
            sumW_full += (R[k]-L[k])
            full_count += 1
            continue
        else
            # 部分充填の位置（k*）
            partial_idx = k
            break
        end
    end

    # 退化：全部フルで z が残っていない（または rank を走破）のとき
    if partial_idx == 0
        # 直前を部分扱いに倒して安定化（N=1等の極端ケース除外）
        if full_count == 0
            # 全く充填不要：z<=0 の場合。diff最大の要素を部分とみなす
            partial_idx = rank[1]
            full_count = 0
            sumW_full = 0.0
        else
            partial_idx = rank[full_count]
            sumW_full -= (R[partial_idx]-L[partial_idx])
            full_count -= 1
        end
    end

    # R(t) = A*t + B の A,B を構成
    # B = diff[k*]
    # A = Σ diff_i*L_i + Σ_{f∈F} diff_f*(R-L)_f - diff[k*]*( ΣL + Σ_{f∈F}(R-L) )
    B = diff[partial_idx]
    sum_diffL = sum(@inbounds(diff[i]*L[i]) for i in 1:C)
    sum_diffW_full = 0.0
    @inbounds for j in 1:full_count
        fj = rank[j]
        sum_diffW_full += diff[fj] * (R[fj]-L[fj])
    end
    A = sum_diffL + sum_diffW_full - diff[partial_idx]*(sumL_all + sumW_full)

    # セル更新
    cell.full_count  = full_count
    cell.partial_idx = partial_idx
    cell.slope       = A
    cell.intercept   = B
    cell.sumL_all    = sumL_all
    cell.sumW_full   = sumW_full
    cell.interm_index = partial_idx  # 互換（部分位置）
    # 旧Avail_space は互換保持のみ（使わない）
    cell.Avail_space = 0.0

    # 現在値を線形式で同期
    cell.regret = A*t + B
    return
end


##########
# 5. “右端方向（TR→左）”の区間境界 t* を明示計算
#    t* = 1 / [ ΣL + Σ_{F∪{k*}} (R-L) ]
##########
@inline function boundary_t_right(cell::minimax_regret_tuple, L::Vector{Float64}, R::Vector{Float64})
    kstar = cell.partial_idx
    return 1.0 / (cell.sumL_all + cell.sumW_full + (R[kstar]-L[kstar]))
end


##########
# 6. 差分更新（能動集合不変の間は R += slope * Δt でOK）
##########
@inline function update_regret_by_dt!(cell::minimax_regret_tuple, dt::Float64)
    cell.regret += cell.slope * dt
end


##########
# 7. 行列全体の初期化／再構成ユーティリティ
##########
function initialize_linear_models!(
    matrix::Array{minimax_regret_tuple,2},
    L::Vector{Float64}, R::Vector{Float64},
    t::Float64
)
    A = size(matrix,1)
    @inbounds for i in 1:A, j in 1:A
        if i == j; continue; end
        set_linear_model_for_pair!(matrix[i,j], L, R, t)
    end
    return
end


##########
# 8. 次の境界（TR→TLの一歩）を決める
#    - 現在 t_cur から「最も近い左側の境界」へ進む
#    - もし境界が無ければ t_L に張り付く
#    - ぶつかるペアを返す（そこでだけ能動集合を再構成）
##########
function next_boundary_TR!(
    matrix::Array{minimax_regret_tuple,2},
    L::Vector{Float64}, R::Vector{Float64},
    t_cur::Float64, t_L::Float64; eps::Float64=1e-12
)
    A = size(matrix,1)
    t_next = t_L
    hit_pairs = Tuple{Int,Int}[]

    # 全ペアの境界 t* を走査し、t_L < t* < t_cur の中で最大のものを選ぶ
    @inbounds for i in 1:A, j in 1:A
        if i == j; continue; end
        tstar = boundary_t_right(matrix[i,j], L, R)
        if (t_L + eps) < tstar && tstar < (t_cur - eps)
            if tstar > t_next + eps
                t_next = tstar
            end
        end
    end

    # 境界に当たるペア（許容誤差内）を収集
    if t_next > t_L + eps
        @inbounds for i in 1:A, j in 1:A
            if i == j; continue; end
            tstar = boundary_t_right(matrix[i,j], L, R)
            if abs(tstar - t_next) <= 1e-10
                push!(hit_pairs, (i,j))
            end
        end
    end

    return t_next, hit_pairs
end


##########
# 9. Δt 分の一括差分更新＋境界での能動集合の再構成
##########
function advance_TR_once!(
    matrix::Array{minimax_regret_tuple,2},
    L::Vector{Float64}, R::Vector{Float64},
    t_cur::Float64, t_L::Float64
)
    # まず次の境界へ
    t_next, hit_pairs = next_boundary_TR!(matrix, L, R, t_cur, t_L)

    # Δt で全ペアを差分更新（能動集合不変範囲）
    dt = t_next - t_cur  # 負（左へ進む）
    @inbounds for i in 1:size(matrix,1), j in 1:size(matrix,2)
        if i == j; continue; end
        update_regret_by_dt!(matrix[i,j], dt)
    end

    # 境界到達したペアのみ、能動集合と線形モデルを再構成
    @inbounds for (i,j) in hit_pairs
        set_linear_model_for_pair!(matrix[i,j], L, R, t_next)
    end

    return t_next, hit_pairs
end


##########
# 10.（任意）現在 t における各代替案 p の最大リグレットとランキング
##########
function max_regret_vector(matrix::Array{minimax_regret_tuple,2})
    A = size(matrix,1)
    MR = fill(-Inf, A)
    @inbounds for p in 1:A
        maxv = -Inf
        @inbounds for q in 1:A
            if p == q; continue; end
            rv = matrix[p,q].regret
            if rv > maxv
                maxv = rv
            end
        end
        MR[p] = maxv
    end
    return MR
end

# MRが小さいほど良い → 昇順で並べる（返り値は代替案IDの並び）
@inline function ranking_from_MR(MR::Vector{Float64})
    return sortperm(MR)  # ascending
end


##########
# 11. 使い方メモ（モジュール側のメインループ想定）
#
#  (1) tL, tR = find_optimal_trange(L, R)
#  (2) M = create_minimax_R_Matrix(utility)
#  (3) initialize_linear_models!(M, L, R, tR)    # 右端で線形モデル確定
#  (4) t = tR
#      while t > tL + eps
#         t, hits = advance_TR_once!(M, L, R, t, tL)
#         MR = max_regret_vector(M)
#         rk = ranking_from_MR(MR)
#         # 必要なら記録...
#      end
#
#  ※ 左→右へ（tL→tR）進めたい場合は、boundary 式を対称に組み直すか、
#     明示的に “左端側の境界” を導出して next を選ぶ実装にしてください。
##########
