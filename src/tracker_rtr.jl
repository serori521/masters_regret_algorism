module RegretRTRTracker

using ..SetRegretCore

const EPS = 1e-12

"行列表現の (p,q) セルから、区間内の任意 t における R_{p,q}(t)=A t + B を評価"
@inline function eval_line(cell::SetRegretCore.minimax_regret_tuple, t::Float64)
    return cell.slope * t + cell.intercept
end

"区間内の任意 t における MR ベクトルを '評価のみ' で生成（状態は更新しない）"
function eval_MR_vector(matrix::Array{SetRegretCore.minimax_regret_tuple,2}, t::Float64)
    A = size(matrix, 1)
    MR = Vector{Float64}(undef, A)
    @inbounds for p in 1:A
        m = -Inf
        @inbounds for q in 1:A
            q == p && continue
            v = eval_line(matrix[p, q], t)
            if v > m
                m = v
            end
        end
        MR[p] = m
    end
    return MR
end

"指定 p の、区間内時刻 t における支配相手 argmax_q R_{p,q}(t) を返す"
function dominant_opponent_at(matrix::Array{SetRegretCore.minimax_regret_tuple,2},
    p::Int, t::Float64)
    A = size(matrix, 1)
    bestq = 0
    bestv = -Inf
    @inbounds for q in 1:A
        q == p && continue
        v = eval_line(matrix[p, q], t)
        if v > bestv
            bestv = v
            bestq = q
        end
    end
    return bestq, bestv
end

"内側交点のうち、(t_bp, t_prev) に存在する最右のものを返す（無ければ nothing）"
function rightmost_inner_crossing(matrix::Array{SetRegretCore.minimax_regret_tuple,2},
    c::Int, qc::Int,
    t_bp::Float64, t_prev::Float64)
    A = size(matrix, 1)
    # 区間左端で qc を上回る候補のみを見る
    rc_bp = eval_line(matrix[c, qc], t_bp)
    tstar_max = -Inf
    found = false
    @inbounds for q in 1:A
        q == c && continue
        q == qc && continue
        rq_bp = eval_line(matrix[c, q], t_bp)
        if rq_bp > rc_bp + 1e-14  # 厳密に上回るものだけ
            A1, B1 = matrix[c, qc].slope, matrix[c, qc].intercept
            A2, B2 = matrix[c, q].slope, matrix[c, q].intercept
            denom = (A1 - A2)
            abs(denom) < 1e-16 && continue # 平行なら交わらない
            tstar = (B2 - B1) / denom
            if (t_bp + 1e-12) < tstar < (t_prev - 1e-12)
                if tstar > tstar_max
                    tstar_max = tstar
                    found = true
                end
            end
        end
    end
    return found ? tstar_max : nothing
end
# 候補 t* を検証してフィルタ
# function filter_true_rank_changes(utility, L, R, tstars; eps=1e-7)
#     keep = Float64[]
#     for t in tstars
#         tL = t - eps
#         tR = t + eps
#         sL = snapshot_at(utility, L, R, tL)
#         sR = snapshot_at(utility, L, R, tR)
#         if sL.ranking != sR.ranking
#             push!(keep, t)
#         end
#     end
#     # 近接重複の間引き（既存の処理に合わせる）
#     sort!(keep)
#     dedup = Float64[]
#     last = -Inf
#     for x in keep
#         if x - last > 1e-9
#             push!(dedup, x)
#             last = x
#         end
#     end
#     return dedup
# end
"外側順位の変化を [tL, tR] (tL < tR) の区間で調べ、交点 t* を列挙する（区間内のみ二分探索）"
function outer_crossings_in_interval(matrix::Array{SetRegretCore.minimax_regret_tuple,2},
    tL::Float64, tR::Float64)
    MR_L = eval_MR_vector(matrix, tL)
    MR_R = eval_MR_vector(matrix, tR)
    # 順位ベクトルを作る（小さい順、同値は安定に）
    idxL = collect(eachindex(MR_L))
    idxR = collect(eachindex(MR_R))
    rkL = sort(idxL; by=i -> MR_L[i])  # 昇順
    rkR = sort(idxR; by=i -> MR_R[i])  # 昇順

    # 逆転した候補ペア（単純化: 上位近傍の入れ替わりを中心に検出）
    # 全ペアは O(A^2) になるが、Aが大きくなければこのままでもOK
    A = length(MR_L)
    tstars = Float64[]
    @inbounds for p1 in 1:A-1, p2 in p1+1:A
        # 符号が変われば、その間に等しい点がある
        fL = MR_L[p1] - MR_L[p2]
        fR = MR_R[p1] - MR_R[p2]
        if fL == 0 || fR == 0
            # 端点で一致（レア）。端点側を記録する（tL優先）
            if abs(fL) <= 1e-14
                push!(tstars, tL)
            elseif abs(fR) <= 1e-14
                push!(tstars, tR)
            end
            continue
        end
        if (fL > 0 && fR < 0) || (fL < 0 && fR > 0)
            # 2分探索で t* を復元（区間内は piecewise-linear なので収束は確実）
            a, b = tL, tR
            fa, fb = fL, fR
            for _ in 1:80
                m = 0.5 * (a + b)
                MR_m = eval_MR_vector(matrix, m)
                fm = MR_m[p1] - MR_m[p2]
                if abs(fm) < 1e-12
                    a = b = m
                    break
                end
                if (fa > 0 && fm > 0) || (fa < 0 && fm < 0)
                    a, fa = m, fm
                else
                    b, fb = m, fm
                end
                if abs(b - a) < 1e-12
                    break
                end
            end
            push!(tstars, 0.5 * (a + b))
        end
    end
    # 重複除去とソート
    sort!(tstars)
    uniq = Float64[]
    lastv = -Inf
    for t in tstars
        if t - lastv > 1e-10
            push!(uniq, t)
            lastv = t
        end
    end
    return uniq
end

"TeX Algorithm: 右→左で順位変化点を収集"
function find_change_points(utility::Matrix{Float64}, L::Vector{Float64}, R::Vector{Float64})
    tL, tR = SetRegretCore.find_optimal_trange(L, R)
    M = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(M, L, R, tR)

    changes = Float64[]

    t = tR
    while t > tL + EPS
        t_prev = t
        # 1) 次の“大域ブレークポイント”（傾き一定区間の左端）
        t_bp, _ = SetRegretCore.next_boundary_TR!(M, t_prev, tL)

        # 2) 現在のMinimax解 c と 3) その支配相手 qc（時刻 t_prev で評価）
        MR_prev = SetRegretCore.max_regret_vector(M)
        rk_prev = SetRegretCore.ranking_from_MR(MR_prev)
        c = rk_prev[1]
        qc, _ = dominant_opponent_at(M, c, t_prev)

        # 4) 内側候補を用いた最右交点（t_bp, t_prev）を探索
        t_inner = rightmost_inner_crossing(M, c, qc, t_bp, t_prev)

        # 5) 次イベント点の決定
        t_next = isnothing(t_inner) ? t_bp : t_inner

        # 6) 外側順位の変化チェックの直前に追加
        snapL = snapshot_at(utility, L, R, t_next)
        snapR = snapshot_at(utility, L, R, t_prev)

        if snapL.ranking == snapR.ranking
            # 端点の順位が同じ → この区間では並びは変わっていないのでスキップ
            # t を進める処理へ
        else
            append!(changes, outer_crossings_in_interval(M, t_next, t_prev))
        end

        # 7) 時刻とモデルを進める
        if abs(t_next - t_bp) <= 1e-12
            # ブレークポイントまで一気に進め、傾き集合を更新
            t, _ = SetRegretCore.advance_TR_once!(M, t_prev, tL)
        else
            # 区間内の途中停止：差分更新のみ（境界は越えない）
            dt = t_next - t_prev # (<0)
            @inbounds for i in 1:size(M, 1), j in 1:size(M, 2)
                i == j && continue
                SetRegretCore.update_regret_by_dt!(M[i, j], dt)
            end
            t = t_next
        end
    end
    dedup = Float64[]
    # 正規化（範囲外や重複を弾く）
    push!(dedup, tL)  # 始端も必ず含める
    changes = filter(x -> tL - 1e-9 <= x <= tR + 1e-9, changes)
    sort!(changes)
    # 近接重複の間引き

    last = -Inf
    for x in changes
        if x - last > 1e-9
            push!(dedup, x)
            last = x
        end
    end
    push!(dedup, tR)  # 終端は必ず含める
    return dedup
end
# =======================================デバッグ用
# === 追加: 任意tのスナップショット ===
function snapshot_at(utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64},
    t::Float64)
    Mtmp = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(Mtmp, L, R, t)
    MR = SetRegretCore.max_regret_vector(Mtmp)
    # 昇順のインデックス（=順位）
    rk = sort(collect(eachindex(MR)); by=i -> MR[i])
    # Minimax解（同値許容）
    m = minimum(MR)
    winners = findall(x -> x ≤ m + 1e-10, MR)
    return (t=t, MR=MR, ranking=rk, winners=winners)
end
# === 追加: デバッグ出力付き ===
"""
find_change_points_debug(utility, L, R)

戻り値:
- cps::Vector{Float64}        : change point の時刻
- logs::Vector{NamedTuple}    : 各 t における (t, MR, ranking, winners)
"""
function find_change_points_debug(utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64})
    cps = find_change_points(utility, L, R)
    logs = [snapshot_at(utility, L, R, t) for t in cps]
    return cps, logs
end
# === 追加: 簡易プリンタ ===
function print_change_point_logs(logs)
    for (k, s) in enumerate(logs)
        println("---- Change #", k, "  t = ", s.t)
        println("winners (min MR): ", s.winners)
        println("ranking (low→high MR): ", s.ranking)
        println("MR: ", s.MR)
    end
end

end # module
