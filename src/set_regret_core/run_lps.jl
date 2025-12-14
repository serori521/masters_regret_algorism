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
"""
    debug_outer_rejection(matrix, qstar, x_p_max, t_min, t_max; x0=nothing, eps=1e-12,
                          near_tol=1e-6, print_all_in_interval=false, max_print=80)

collect_outer_changes の「候補が push されない理由」を特定するデバッグ。

- x0 を指定すると、交点 x が x0 近傍（|x-x0|<=near_tol）のペアだけ詳しく出す
- print_all_in_interval=true にすると、区間 [t_min, t_max] に入る交点を全部出す（多いので注意）
戻り値: 該当候補の NamedTuple 配列（後で手元でソート/分析できる）
"""
function debug_outer_rejection(matrix, qstar, x_p_max, t_min, t_max;
    x0=nothing, eps=1e-12, near_tol=1e-6,
    print_all_in_interval=false, max_print=80)

    A = length(qstar)
    out = NamedTuple[]
    nprinted = 0

    # minimax_regret_tuple から必要フィールドを取る（名前が違っても落ちにくいように）
    get_slope(l) = getproperty(l, :slope)
    get_intercept(l) = getproperty(l, :intercept)
    get_tstar(l) = getproperty(l, :tstar)

    for p1 in 1:A-1, p2 in p1+1:A
        q1 = qstar[p1]
        q2 = qstar[p2]
        (q1 == 0 || q2 == 0) && continue

        l1 = matrix[p1, q1]
        l2 = matrix[p2, q2]

        A1 = get_slope(l1)
        B1 = get_intercept(l1)
        t1 = get_tstar(l1)
        A2 = get_slope(l2)
        B2 = get_intercept(l2)
        t2 = get_tstar(l2)

        denom = (A1 - A2)
        abs(denom) <= eps && continue  # 平行（交点なし）

        x = (B2 - B1) / denom

        in_interval = (x >= t_min - eps) && (x <= t_max + eps)
        near_x0 = (x0 === nothing) ? false : (abs(x - x0) <= near_tol)

        if !(near_x0 || (print_all_in_interval && in_interval))
            continue
        end

        # collect_outer_changes と同じ lower
        parts = (t_min=t_min, t1=t1, t2=t2, xp1=x_p_max[p1], xp2=x_p_max[p2])
        lower = max(parts.t_min, parts.t1, parts.t2, parts.xp1, parts.xp2)

        pass_lower = (lower <= x + eps)
        pass_upper = (x <= t_max - eps)

        verdict =
            if pass_lower && pass_upper
                :PUSHED
            elseif !pass_lower
                :REJECT_LOWER
            else
                :REJECT_UPPER
            end

        # どの要素が lower を支配してるか（複数同率あり）
        dom = String[]
        abs(parts.t_min - lower) <= 10eps && push!(dom, "t_min")
        abs(parts.t1 - lower) <= 10eps && push!(dom, "tstar(p1)")
        abs(parts.t2 - lower) <= 10eps && push!(dom, "tstar(p2)")
        abs(parts.xp1 - lower) <= 10eps && push!(dom, "x_p_max(p1)")
        abs(parts.xp2 - lower) <= 10eps && push!(dom, "x_p_max(p2)")

        rec = (
            p1=p1, p2=p2, q1=q1, q2=q2,
            x=x, lower=lower, t_min=t_min, t_max=t_max,
            dom=join(dom, "|"),
            parts=parts,
            verdict=verdict
        )
        push!(out, rec)

        if nprinted < max_print
            println("---- candidate ----")
            println("p1=$p1 q1=$q1   p2=$p2 q2=$q2")
            println("x = $x")
            println("lower = $lower   (dom: ", join(dom, ", "), ")")
            println("parts = ", parts)
            println("check: lower<=x? ", pass_lower, "   x<=t_max-eps? ", pass_upper, "   => ", verdict)
            nprinted += 1
        end
    end

    println("\n[debug_outer_rejection] total hits = ", length(out), " (printed ", min(length(out), max_print), ")")
    return out
end

@inline function t_left(t; δ=1e-12)
    return t - max(δ, 1e-12 * max(1.0, abs(t)))
end

# stateを「いまのt」に必ず整合させる（デバッグ中は常にfull refresh推奨）
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

    # order/pos を確定（E1/E2のジャンプ後に必ずやる）
    order_new = snapshot_state(matrix, qstar, t; eps=eps).rank
    @inbounds for (i, p) in enumerate(order_new)
        pos[p] = i
    end
    return order_new
end




function run_lps(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    t_L::Float64, t_U::Float64;
    eps::Float64=EPS_DEFAULT
)
    initialize_linear_models!(matrix, wL, wU, t_U; eps=eps)  # 初期モデル:contentReference[oaicite:6]{index=6}

    A = size(matrix, 1)
    qstar = zeros(Int, A)
    hat_q = zeros(Int, A)
    x_p_max = fill(t_L, A)

    # order/pos は外側E3のswap用
    order = collect(1:A)
    pos = zeros(Int, A)

    # まず t=t_U に完全同期（Inv-A/B/C）
    order = sync_state!(matrix, wL, wU, t_U, t_L, qstar, hat_q, x_p_max, order, pos;
        eps=eps, refresh_pairs=true)

    Tchg = Float64[]
    timeline = SnapshotEntry[]
    push_snapshot!(Tchg, timeline, matrix, qstar, t_U; eps=eps, detect_change=false)

    t = t_U
    while t > t_L + eps
        # --- (0) ループ先頭で不変条件（Inv-A/B/C）を保証（デバッグ中はtrue推奨）
        order = sync_state!(matrix, wL, wU, t, t_L, qstar, hat_q, x_p_max, order, pos;
            eps=eps, refresh_pairs=true)

        # --- (1) 次のジャンプ時刻（E1/E2）
        E1, _ = next_coefficient_event(matrix, t_L, t; eps=eps)     # tstarベース:contentReference[oaicite:7]{index=7}
        E2, _ = next_inner_event(x_p_max, t_L, t; eps=eps)          # x_p_maxベース:contentReference[oaicite:8]{index=8}
        t_next = max(max(E1, E2), t_L)

        if t_next >= t - eps
            break
        end

        # --- (2) E3（外側順位変化）を区間 (t_next, t] で列挙
        events = collect_outer_changes(matrix, qstar, x_p_max, t_next, t; eps=eps)
        # collect_outer_changes は tstar と x_p_max を lower に使うので、ここで整合が必須:contentReference[oaicite:9]{index=9}

        k = 1
        while k <= length(events)
            x = events[k].x

            # 同じxをまとめる
            j = k
            while j <= length(events) && abs(events[j].x - x) <= eps
                j += 1
            end

            # x時点でモデルを合わせる（デバッグ段階は全更新でOK）
            refresh_all_pairs!(matrix, wL, wU, t_left(x); eps=eps)

            did_swap = false
            progress = true
            swapped = Set{Tuple{Int,Int}}()

            while progress
                progress = false
                for idx in k:(j-1)
                    p1 = events[idx].p1
                    p2 = events[idx].p2
                    key = p1 < p2 ? (p1, p2) : (p2, p1)
                    key in swapped && continue

                    i1 = pos[p1]
                    i2 = pos[p2]
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
                snap = snapshot_state(matrix, qstar, x; eps=eps)
                push!(timeline, (t=x, MR=snap.MR, rank=copy(order), winners=snap.winners))
            end

            k = j
        end

        # --- (3) ジャンプ：t ← t_next
        t = t_next

        # --- (4) ジャンプ後に必ず完全同期（Inv-A/B/C）
        order = sync_state!(matrix, wL, wU, t, t_L, qstar, hat_q, x_p_max, order, pos;
            eps=eps, refresh_pairs=true)

        push_snapshot!(Tchg, timeline, matrix, qstar, t; eps=eps, detect_change=false)
    end

    return (changes=Tchg, timeline=timeline)
end

###############################
# 8. KDSベース左向き走査メインループ
###############################
