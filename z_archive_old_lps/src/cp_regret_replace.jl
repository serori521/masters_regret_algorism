module SetRegretCore

# 公開API
export minimax_regret_tuple,
       find_optimal_trange,
       create_minimax_R_Matrix,
       initialize_linear_models!,
       advance_TR_once!,
       max_regret_vector,
       ranking_from_MR,
       check_outer_change


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
# 2. データ構造
###############################
mutable struct minimax_regret_tuple
    # --- 固定情報 ---
    difference_U::Vector{Float64}   # diff_i = u_i(q)-u_i(p)
    rank::Vector{Int}               # diff降順のインデックス並び（貪欲順）

    # --- 現在の R_{p,q}(t) 値 ---
    regret::Float64                 # 現在時刻の R_{p,q}(t)

    # --- 旧calc_IPWとの名残（右→左では実質使わないが保持） ---
    interm_index::Int               # 部分割当インデックス（現partial_idxと同義にしてよい）
    Avail_space::Float64            # 右→左では未使用

    # --- 能動集合の線形モデル (F, k*) に依存するパラメータ ---
    full_count::Int                 # F のサイズ：rank[1:full_count] はフルで使われている
    partial_idx::Int                # k* = rank[full_count+1] （部分充填の基準）
    slope::Float64                  # A_{p,q} （傾き）
    intercept::Float64              # B_{p,q} （切片 = diff[k*] ではあるが後で連続性調整あり）

    # --- キャッシュ（O(1)でA,B,t*を再構成するための情報） ---
    sumL_all::Float64               # Σ_i w_i^L
    sum_diffL::Float64              # Σ_i diff_i * w_i^L
    cumW::Vector{Float64}           # cumW[j]   = Σ_{s<=j} (w^U - w^L)[rank[s]]
    cumDiffW::Vector{Float64}       # cumDiffW[j] = Σ_{s<=j} diff[rank[s]]*(w^U - w^L)[rank[s]]

    # --- 次の能動集合切替時刻（右→左での次のイベント候補） ---
    tstar::Float64                  # t*_{p,q} = 1 / ( Σ w^L + Σ_{f∈F∪{k*}} (w^U-w^L) )
    valid::Bool                     # tstar がまだ探索範囲内で有効か
end

function _minimax_empty()
    minimax_regret_tuple(
        Float64[], Int[],
        0.0,
        0, 0.0,
        0, 0, 0.0, 0.0,
        0.0, 0.0, Float64[], Float64[],
        -Inf, true
    )
end


###############################
# 3. 全ペアの差分ベクトル(difference_U, rank)を用意
###############################
function create_minimax_R_Matrix(utility::Matrix{Float64})
    A, N = size(utility)
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
# 4. ペアごとの事前キャッシュ
#    sumL_all, sum_diffL, cumW, cumDiffW をまとめて計算
###############################
function precompute_pair_caches!(
    cell::minimax_regret_tuple,
    wL::Vector{Float64}, wU::Vector{Float64}
)
    rank = cell.rank
    diff = cell.difference_U
    N = length(rank)

    # Σ_i w_i^L
    cell.sumL_all = sum(wL)

    # rank順に (w^U - w^L) の累積
    width_base = wU .- wL
    w_rank = width_base[rank]
    cell.cumW = N == 0 ? Float64[] : cumsum(w_rank)

    # diff_i * (w^U - w^L)_i の累積
    d_rank = diff[rank] .* w_rank
    cell.cumDiffW = N == 0 ? Float64[] : cumsum(d_rank)

    # Σ_i diff_i * w_i^L
    cell.sum_diffL = sum(@inbounds(diff[k] * wL[k]) for k in eachindex(wL))

    return
end


###############################
# 5. 1ペアの線形モデル(A,B)の初期化（t現在値で決定）
#
# Greedy:
#   - rankの先頭から wWidth[k] = (w^U[k]-w^L[k]) * t を順に詰め，
#     残余 z = 1 - t * Σ w_i^L から引いていく
#   - 完全に詰めきれたものが F（full_count 個）
#   - 最後に途中まで入ってるものが k* = partial_idx
#
# その (F, k*) から slope(A), intercept(B) を O(1) で組み立て，
# R_{p,q}(t) = A t + B を現在時刻に合わせて cell.regret にセット。
# あわせて次境界時刻 t*_{p,q} もキャッシュ。
###############################
function set_linear_model_for_pair!(
    cell::minimax_regret_tuple,
    wL::Vector{Float64}, wU::Vector{Float64},
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
        cell.tstar = -Inf
        cell.valid = false
        return
    end

    # キャッシュが未計算なら初回に作る
    if isempty(cell.cumW)
        precompute_pair_caches!(cell, wL, wU)
    end

    sumL_all = cell.sumL_all
    width_each = (wU .- wL) .* t   # (w^U - w^L)*t
    z = 1.0 - t * sumL_all
    if z < 0.0 && z > -eps
        z = 0.0
    end

    # Greedyで F と k* を決定
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

    # 退化：全部フルで埋まってpartialが無いケース
    if partial_idx == 0
        if full_count == 0
            partial_idx = rank[1]
        else
            partial_idx = rank[full_count]
            full_count -= 1
        end
    end

    # A,B を計算
    m = full_count
    kstar = partial_idx
    B = diff[kstar]

    # slope:
    # A = Σ diff_i w^L_i
    #   + Σ_{f∈F} diff_f (w^U_f - w^L_f)
    #   - diff_{k*} ( Σ w^L_i + Σ_{f∈F} (w^U_f - w^L_f) )
    A_val =
        cell.sum_diffL +
        (m == 0 ? 0.0 : cell.cumDiffW[m]) -
        diff[kstar] * ( cell.sumL_all + (m == 0 ? 0.0 : cell.cumW[m]) )

    # セット
    cell.full_count = m
    cell.partial_idx = kstar
    cell.slope = A_val
    cell.intercept = B
    cell.interm_index = kstar
    cell.Avail_space = 0.0  # 右→左では未使用
    cell.regret = A_val * t + B

    # 次の境界時刻 t* をキャッシュ
    cell.tstar = boundary_t_right_cached(cell)
    cell.valid = isfinite(cell.tstar)

    return
end


###############################
# 6. 次の「能動集合切替」境界 t*_{p,q}
#    t* = 1 / ( Σ w^L + Σ_{f∈F∪{k*}} (w^U - w^L) )
###############################
@inline function boundary_t_right_cached(cell::minimax_regret_tuple)
    m = cell.full_count
    if isempty(cell.cumW) || m + 1 > length(cell.cumW)
        return -Inf   # もうこれ以上変わらない
    end
    return 1.0 / (cell.sumL_all + cell.cumW[m+1])
end


###############################
# 7. 差分更新（傾き一定区間なら R += slope * Δt で済む）
###############################
@inline function update_regret_by_dt!(cell::minimax_regret_tuple, dt::Float64)
    cell.regret += cell.slope * dt
end


###############################
# 8. 現在の (full_count, partial_idx) に基づいて slope を再構築
###############################
@inline function rebuild_slope!(cell::minimax_regret_tuple)
    m = cell.full_count
    kstar = cell.partial_idx
    cell.slope =
        cell.sum_diffL +
        (m == 0 ? 0.0 : cell.cumDiffW[m]) -
        cell.difference_U[kstar] * (cell.sumL_all + (m == 0 ? 0.0 : cell.cumW[m]))
end


###############################
# 9. promote_right_once!
#    能動集合を 1ステップ進める（k* がフル側に吸収されたとみなす）
#    → full_count += 1
#    → 新しい partial_idx を rank[m+1] に更新
#    → slope/intercept/regret/tstar を一貫的に更新
###############################
@inline function promote_right_once!(
    cell::minimax_regret_tuple,
    t_now::Float64
)
    cell.full_count += 1
    m = cell.full_count
    r = cell.rank

    # 新しい k* 候補
    if m < length(r)
        new_k = r[m+1]
        cell.partial_idx = new_k
    else
        # 末尾まで行った場合は最後の要素をpartialとして固定
        cell.partial_idx = r[m]
    end

    # slope を再計算
    rebuild_slope!(cell)

    # intercept を（連続性から）再推定
    # R(t_now) はこれまで update_regret_by_dt! 済みなので、その値に
    #   intercept := R(t_now) - slope * t_now
    cell.intercept = cell.regret - cell.slope * t_now

    # 念のため連続性を明示
    cell.regret = cell.slope * t_now + cell.intercept

    # 次の境界 t* を更新しキャッシュ
    cell.tstar = boundary_t_right_cached(cell)
    cell.valid = isfinite(cell.tstar)

    # interm_index も partial_idx に合わせておく
    cell.interm_index = cell.partial_idx
end


###############################
# 10. 初期化：t = t_R 時点で全 (p,q) の線形モデルを確定
###############################
function initialize_linear_models!(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    tR::Float64
)
    A = size(matrix, 1)
    @inbounds for i in 1:A, j in 1:A
        if i == j
            continue
        end
        precompute_pair_caches!(matrix[i, j], wL, wU)
        set_linear_model_for_pair!(matrix[i, j], wL, wU, tR)
    end
    return
end


###############################
# 11. 現在時刻での MR_p(t) ベクトルと Minimax解 c, その支配相手 q*
###############################
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
    # 小さいほど良い（= Minimax Regret が小さい）
    return sortperm(MR)
end

# Minimax解 c と その時点の支配相手 q* を返す
function current_minimax_pair(matrix::Array{minimax_regret_tuple,2})
    MR = max_regret_vector(matrix)
    # c = argmin_p MR[p]
    c = argmin(MR)
    # q* = argmax_q R_{c,q}(t)
    best_q = 0
    best_val = -Inf
    @inbounds for q in 1:size(matrix,1)
        if q == c; continue; end
        rv = matrix[c,q].regret
        if rv > best_val
            best_val = rv
            best_q = q
        end
    end
    return c, best_q
end


###############################
# 12. 内側交点探索 FindInnerCrossing (TeX Algorithm 5.2)
#     - 固定 c
#     - 今の1位 q* （= cのMRを与える相手）
#     - それと他の q との交点を調べる
#     - 区間 (t_left, t_right] の中で最大のものを返す
###############################
function find_inner_crossing(
    matrix::Array{minimax_regret_tuple,2},
    c::Int, qstar::Int,
    t_left::Float64, t_right::Float64;
    eps::Float64=1e-12
)
    # 交点候補の最大値を返す。なければ t_left を返す
    t_candidate = t_left
    Astar = matrix[c,qstar].slope
    Bstar = matrix[c,qstar].intercept

    @inbounds for q in 1:size(matrix,1)
        if q == c || q == qstar
            continue
        end
        Aq = matrix[c,q].slope
        Bq = matrix[c,q].intercept

        AΔ = Astar - Aq
        BΔ = Bq - Bstar
        if abs(AΔ) < eps
            continue
        end
        x = BΔ / AΔ  # R_{c,q*}(x) = R_{c,q}(x)

        if (t_left + eps) < x && x <= (t_right + eps)
            if x > t_candidate + eps
                t_candidate = x
            end
        end
    end

    return t_candidate
end


###############################
# 13. 外側順位の変化検出 (TeX Algorithm 5.3)
#     - 時刻 t_old -> t_new に進んだときに
#       MR_p(t) の順位が変わったかどうかを見る
#     - 変わったペア (p1,p2) について
#       MR_{p1}(t) = MR_{p2}(t) の交点を線形で解き、
#       それが (t_new, t_old] にあれば記録
###############################
function check_outer_change(
    matrix::Array{minimax_regret_tuple,2},
    t_old::Float64, t_new::Float64;
    eps::Float64=1e-12
)
    # 返り値：交点候補時刻の配列（空でもよい）
    # 注意：ここでは MR_p(t) を「pの中のmax R_{p,q}(t)」で近似的に
    # 線形化するには、「その区間では p の中で支配している q が一定」と仮定する必要がある。
    # 今はその仮定で交点を出す。厳密にやるなら p ごとに支配qが変わるイベントも見る必要あり。
    A = size(matrix,1)

    # t_old 時点の MR とその支配q
    MR_old = fill(-Inf, A)
    argq_old = fill(0, A)
    @inbounds for p in 1:A
        bestv = -Inf
        bestq = 0
        for q in 1:A
            if p == q; continue; end
            v = matrix[p,q].slope * t_old + matrix[p,q].intercept
            if v > bestv
                bestv = v
                bestq = q
            end
        end
        MR_old[p] = bestv
        argq_old[p] = bestq
    end

    # t_new 時点の MR とその支配q
    MR_new = fill(-Inf, A)
    argq_new = fill(0, A)
    @inbounds for p in 1:A
        bestv = -Inf
        bestq = 0
        for q in 1:A
            if p == q; continue; end
            v = matrix[p,q].slope * t_new + matrix[p,q].intercept
            if v > bestv
                bestv = v
                bestq = q
            end
        end
        MR_new[p] = bestv
        argq_new[p] = bestq
    end

    # ランク（小さいほど良い）
    rk_old = sortperm(MR_old)
    rk_new = sortperm(MR_new)

    # 順位が変わった候補だけ交点を推定
    Tchg = Float64[]
    @inbounds for p1 in 1:A-1
        for p2 in p1+1:A
            # 順位の相対関係が変わったか？
            old_order = (findfirst(==(p1), rk_old) < findfirst(==(p2), rk_old))
            new_order = (findfirst(==(p1), rk_new) < findfirst(==(p2), rk_new))
            if old_order == new_order
                continue
            end

            # p1のMRは (p1, q1_old) の直線，p2のMRは (p2, q2_old) の直線として交点を推定
            q1 = argq_old[p1]
            q2 = argq_old[p2]

            A1 = matrix[p1,q1].slope
            B1 = matrix[p1,q1].intercept
            A2 = matrix[p2,q2].slope
            B2 = matrix[p2,q2].intercept

            AΔ = A1 - A2
            BΔ = B2 - B1
            if abs(AΔ) < eps
                continue
            end
            xcand = BΔ / AΔ  # MR_{p1}(xcand) = MR_{p2}(xcand)

            # 右→左なので t_old > t_new
            if (t_new - eps) < xcand && xcand <= (t_old + eps)
                push!(Tchg, xcand)
            end
        end
    end

    return Tchg
end


###############################
# 14. 次イベント計算：next_event_TR!
#
# 右→左走査で，次にどこまでジャンプするかを決める。
# 候補は：
#   (i) 能動集合境界: tstar_{p,q} の最大（t_cur より左・t_L より右）
#   (ii) 内側交点: 現Minimax解 c とその支配相手 q* を固定して，
#        FindInnerCrossing で得られる t_in
#
# 戻り値:
#   t_next::Float64
#   hit_pairs::Vector{Tuple{Int,Int}}  ... t_next にヒットした (p,q) の集合
#   event_type::Symbol                ... :active_set または :inner
###############################
function next_event_TR!(
    matrix::Array{minimax_regret_tuple,2},
    t_cur::Float64, t_L::Float64;
    eps::Float64=1e-12
)
    A = size(matrix,1)

    # ---- (i) 能動集合側イベント候補 ----
    t_next_active = t_L
    @inbounds for i in 1:A, j in 1:A
        if i == j; continue; end
        cell = matrix[i,j]
        tstar = cell.tstar
        # t_L < tstar < t_cur の中で最大をとる
        if (t_L + eps) < tstar < (t_cur - eps)
            if tstar > t_next_active + eps
                t_next_active = tstar
            end
        end
    end

    # ---- (ii) 内側交点イベント候補 ----
    # Minimax解 c とその1位 q*
    c, qstar = current_minimax_pair(matrix)
    t_in = find_inner_crossing(matrix, c, qstar, t_L, t_cur; eps=eps)

    # ---- (iii) t_next の決定 ----
    # とりあえず "より右に近い（=大きい）ほう" を優先
    t_candidate = t_L
    event_type = :none

    if t_in > t_candidate + eps
        t_candidate = t_in
        event_type = :inner
    end
    if t_next_active > t_candidate + eps
        t_candidate = t_next_active
        event_type = :active_set
    end

    # 同時ヒットする能動集合ペアを列挙
    hit_pairs = Tuple{Int,Int}[]
    if event_type == :active_set && t_candidate > t_L + eps
        @inbounds for i in 1:A, j in 1:A
            if i == j; continue; end
            if abs(matrix[i,j].tstar - t_candidate) <= 1e-10
                push!(hit_pairs, (i,j))
            end
        end
    elseif event_type == :inner
        # innerの場合，"ヒットペア"は (c,qstar) だけを優先的に再構成対象にする
        push!(hit_pairs, (c,qstar))
    end

    return t_candidate, hit_pairs, event_type
end


###############################
# 15. 1ステップ前進（TR→TL）
#
# 手順：
#  (1) next_event_TR! で t_next とイベント種を決める
#  (2) 全ペアの regret を Δt で差分更新
#  (3) ヒットしたペアだけ promote_right_once! または再線形化
#      - 能動集合イベント(:active_set) → promote_right_once!
#      - 内側イベント(:inner) → set_linear_model_for_pair! し直しても良いが、
#        ほんとは現1位(q*)の能動集合の見直し相当
#  (4) 外側順位変化を check_outer_change で検出し，交点を返す
###############################
function advance_TR_once!(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    t_cur::Float64, t_L::Float64
)
    # (1) 次イベント
    t_next, hit_pairs, event_type = next_event_TR!(matrix, t_cur, t_L)

    # (2) 差分更新
    dt = t_next - t_cur  # (<0)
    @inbounds for i in 1:size(matrix,1), j in 1:size(matrix,2)
        if i == j; continue; end
        update_regret_by_dt!(matrix[i,j], dt)
    end

    # (3) イベントの種類ごとにモデル更新
    if event_type == :active_set
        # 能動集合境界ヒット：複数ペアあり得る
        @inbounds for (i,j) in hit_pairs
            promote_right_once!(matrix[i,j], t_next)
        end
    elseif event_type == :inner
        # 内側交点で "cの1位q*" が切り替わる局面
        # → その (c,q*) だけ線形モデルを組み直す
        @inbounds for (i,j) in hit_pairs
            # i==c, j==qstar のはず
            set_linear_model_for_pair!(matrix[i,j], wL, wU, t_next)
        end
    else
        # event_type == :none → もう動けない
        # 何もしない
    end

    # (4) 外側順位変化の記録
    Tchg = check_outer_change(matrix, t_cur, t_next)

    return t_next, hit_pairs, event_type, Tchg
end


###############################
# 16. 想定ループの使い方
#
#   tL, tR = find_optimal_trange(wL, wU)
#   M = create_minimax_R_Matrix(utility)
#   initialize_linear_models!(M, wL, wU, tR)
#
#   t = tR
#   all_changes = Float64[]
#   while t > tL + 1e-12
#       t, hits, evtype, dT = advance_TR_once!(M, wL, wU, t, tL)
#       append!(all_changes, dT)
#       # (optional) ログ取り:
#       #   MR = max_regret_vector(M)
#       #   rk = ranking_from_MR(MR)
#   end
#
# これで:
#   - all_changes に Minimax Regret の順位が切り替わる t がたまる
#   - 各ステップは能動集合の変更(:active_set)か
#     内側1位交代(:inner)のいずれかにより決定される
###############################

end # module SetRegretCore
