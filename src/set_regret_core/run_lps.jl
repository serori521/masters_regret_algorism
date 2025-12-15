###############################
# ログ用スナップショット
###############################
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

function dump_lps_lines!(
    path, matrix, qstar, hat_q, t
)
    A = length(qstar)
    open(path, "a") do io
        for p in 1:A
            q = qstar[p]
            if q != 0
                cell = matrix[p, q]
                println(io, "$t,$p,$q,$(cell.slope),$(cell.intercept),qstar")
            end
            h = hat_q[p]
            if h != 0
                cell = matrix[p, h]
                println(io, "$t,$p,$h,$(cell.slope),$(cell.intercept),hat")
            end
        end
    end
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

function refresh_all_pairs!(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    t::Float64; eps::Float64=EPS_DEFAULT
)
    A = size(matrix, 1)
    @inbounds for p in 1:A, q in 1:A
        p == q && continue
        set_linear_model_for_pair!(matrix[p, q], wL, wU, t; eps=eps)
    end
end

@inline function t_left(t; δ=1e-14)
    return t - max(δ, 1e-14 * max(1.0, abs(t)))
end

# E1(係数切替)が起きたペアだけ、ジャンプ後の t でモデルを更新する
function apply_E1_updates!(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    pairs::Vector{Tuple{Int,Int}},
    t::Float64;
    eps::Float64=EPS_DEFAULT
)
    isempty(pairs) && return
    tt = t_left(t)  # 左向き走査なので「少し左」で合わせる
    @inbounds for (i, j) in pairs
        i == j && continue
        set_linear_model_for_pair!(matrix[i, j], wL, wU, tt; eps=eps)
        # 対称側も安全のため更新（コスト小）
        set_linear_model_for_pair!(matrix[j, i], wL, wU, tt; eps=eps)
    end
end

# 初期化時のみ使用する状態同期関数
function sync_state!(
    matrix, wL, wU, t, t_L,
    qstar, hat_q, x_p_max,
    order, pos;
    eps=EPS_DEFAULT,
    refresh_pairs::Bool=true
)
    A = length(qstar)

    if refresh_pairs
        refresh_all_pairs!(matrix, wL, wU, t_left(t); eps=eps)
    end

    @inbounds for p in 1:A
        refresh_inner_state!(matrix, p, t_left(t), t_L, qstar, hat_q, x_p_max;
            eps=eps, preferred=qstar[p])
    end

    # order/pos を確定
    order_new = snapshot_state(matrix, qstar, t_left(t); eps=eps).rank
    @inbounds for (i, p) in enumerate(order_new)
        pos[p] = i
    end
    return order_new
end

# -----------------------------------------------------------
# キャッシュ付き交点計算ヘルパー
# -----------------------------------------------------------
@inline function compute_intersection_val(
    matrix::Array{minimax_regret_tuple,2},
    qstar::Vector{Int},
    p1::Int, p2::Int;
    eps::Float64=1e-15 # デフォルト値を安全な値に変更
)
    q1 = qstar[p1]
    q2 = qstar[p2]
    (q1 == 0 || q2 == 0) && return -Inf

    l1 = matrix[p1, q1]
    l2 = matrix[p2, q2]

    Adelta = l1.slope - l2.slope
    if abs(Adelta) <= eps
        return -Inf # 平行
    end

    return (l2.intercept - l1.intercept) / Adelta
end

###############################
# KDSベース左向き走査メインループ (完全版)
###############################
function run_lps(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    t_L::Float64, t_U::Float64;
    eps::Float64=EPS_DEFAULT
)
    # 1. 初期化
    initialize_linear_models!(matrix, wL, wU, t_U; eps=eps)
    A = size(matrix, 1)

    qstar = zeros(Int, A)
    hat_q = zeros(Int, A)
    x_p_max = fill(t_L, A)

    # 交点キャッシュ: (p1, p2) -> 交点時刻 x
    # キーは常に p1 < p2 で保存
    cached_crossings = Dict{Tuple{Int,Int},Float64}()

    # ダーティフラグ: trueならキャッシュの再計算が必要
    dirty_outer = trues(A)

    # 順位管理
    order = collect(1:A)
    pos = zeros(Int, A)

    # t=t_U で初期状態に完全同期
    order = sync_state!(matrix, wL, wU, t_U, t_L, qstar, hat_q, x_p_max, order, pos;
        eps=eps, refresh_pairs=true)
    
    timeline = RankTimelineEntry[]    
    push!(timeline, (t=t_U, rank=copy(order)))
    Tchg = Float64[]
          # ★軽量timeline

    t = t_U
    
    # -------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------
    while t > t_L + eps

        # --- (1) 次のジャンプ時刻（E1/E2）を計算 ---
        E1, pairsE1 = next_coefficient_event(matrix, t_L, t; eps=eps)
        E2, idxsE2 = next_inner_event(x_p_max, t_L, t; eps=eps)
        
        t_next = max(max(E1, E2), t_L)

        if t_next >= t
            break
        end

        # --- (2) E3（外側順位変化）の収集 (キャッシュ活用版) ---
        
        # 2-a. Dirtyなペアのキャッシュを更新
        dirty_ps = findall(dirty_outer)
        if !isempty(dirty_ps)
            @inbounds for p1 in dirty_ps
                for p2 in 1:A
                    p1 == p2 && continue
                    if dirty_outer[p2] && p2 < p1
                        continue
                    end
                    # キーの正規化 (p_min, p_max)
                    k = p1 < p2 ? (p1, p2) : (p2, p1)
                    
                    val = compute_intersection_val(matrix, qstar, k[1], k[2]; eps=eps)
                    # println(t,":",k,",",val)
                    cached_crossings[k] = val
                end
            end
            dirty_outer .= false
        end

        # 2-b. キャッシュ内の全交点から、有効範囲 (t_next, t] にあるものを収集
        events = NamedTuple{(:x, :p1, :p2),Tuple{Float64,Int,Int}}[]

        for ((p1, p2), x) in cached_crossings
            x == -Inf && continue

            # 区間チェック: ★ここを修正！ t_next + eps をやめて厳密な t_next < x にする
            # t_next ギリギリの交点も拾う（重複は後段で弾かれる）
            if t_next < x && x <= t + eps
                # 有効条件チェック (lower bound)
                l1 = matrix[p1, qstar[p1]]
                l2 = matrix[p2, qstar[p2]]

                # ここも安全のため eps のゲタを少し甘く見るか、あるいは厳密にする
                # 「交点 x はバリアより右（未来）でなければならない」
                lower = max(t_L, l1.tstar, l2.tstar, x_p_max[p1], x_p_max[p2])
                if lower <= x + eps
                    push!(events, (x=x, p1=p1, p2=p2))
                end
            end
        end

        # 時刻降順にソート
        sort!(events; by=e -> e.x, rev=true)

        # --- (3) イベント処理（Swap） ---
        k = 1
        num_events = length(events)
        while k <= num_events
            x = events[k].x

            # 同じ時刻(x)のイベントをまとめる
            j = k
            while j <= num_events && abs(events[j].x - x) <= eps
                j += 1
            end

            did_swap = false
            progress = true
            swapped = Set{Tuple{Int,Int}}()

            # バブルソート的なスワップ処理
            while progress
                progress = false
                for idx in k:(j-1)
                    p1 = events[idx].p1
                    p2 = events[idx].p2
                    key = p1 < p2 ? (p1, p2) : (p2, p1)
                    key in swapped && continue

                    i1 = pos[p1]
                    i2 = pos[p2]

                    # 隣接していなければスワップできない
                    abs(i1 - i2) == 1 || continue

                    i = min(i1, i2)
                    order[i], order[i+1] = order[i+1], order[i]
                    pos[order[i]] = i
                    pos[order[i+1]] = i + 1

                    push!(swapped, key)
                    did_swap = true
                    progress = true
                end
            end

            if did_swap
                push!(Tchg, x)
                push!(timeline, (t=x, rank=copy(order)))
            end

            k = j
        end

        # --- (4) ジャンプと状態更新 ---
        t = t_next

        # (a) E1: 係数切替
        fireE1 = abs(E1 - t) <= eps
        if fireE1
            apply_E1_updates!(matrix, wL, wU, pairsE1, t; eps=eps)
            @inbounds for (i, j) in pairsE1
                
                if qstar[i] == j
                    dirty_outer[i] = true
                end
            end
        end

        # (b) E2: 内側1位交代
        fireE2 = (abs(E2 - t) <= eps) && !isempty(idxsE2)
        t_eval = t_left(t)

        if fireE2
            @inbounds for p in idxsE2
                refresh_inner_state!(matrix, p, t_eval, t_L, qstar, hat_q, x_p_max; eps=eps)
                dirty_outer[p] = true
            end
        end

        if fireE1
            @inbounds for (i, j) in pairsE1
                refresh_inner_state!(matrix, i, t_eval, t_L, qstar, hat_q, x_p_max; eps=eps)
            end
        end

        
    end

    # ループ終了後（最後に左端の順位を入れる）
    if isempty(timeline) || abs(timeline[end].t - t_L) > eps
        push!(timeline, (t=t_L, rank=copy(order)))
    end

    return (tL=t_L, tU=t_U, changes=Tchg, timeline=timeline)
end