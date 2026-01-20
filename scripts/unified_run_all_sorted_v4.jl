# scripts/unified_run_all_sorted.jl
#
# Unified runner for:
#   (1) minimax_regret (LPS)
#   (2) maximin (scan)
#   (3) maximax (scan)
#
# Run:
#   julia --project=. scripts/unified_run_all_sorted.jl
#
# v4 updates:
#   1) WORST precision/recall/F1 (worst-match across intervals)
#   2) denom_pairs (= C(Alt,2)) で正規化しない（= 一致ペア数 0..denom_pairs を保持）
#      - 例: Alt=5 なら denom_pairs=10。1000×100×10=1,000,000 のように
#        「一致ペア数の総量」をそのまま扱える。
#   3) I/J（区間数）で割る平均化は維持（区間数が多いほど値が増える等の歪みを防ぐ）
#   4) Optional: write per-PCM (repeat) summaries for minimax_regret
#      into data/metrics_julia/pcm_logs/ and merge them.

include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using Base.Threads
using Printf

# -------------------------
# Config
# -------------------------
const NS = 4:8
const M  = 5
const REPEAT_NUM = 1000
const UTILITY_MATRIX_NUM = 100

const UTILITIES = ["u1", "u2"]
const TRUE_WEIGHT_TYPES = ["A", "B", "C", "D", "E"]

const ACTIVE_METHOD_DIRS = unique([
    "AMRD", "AMRwc", "AMRW", "AMRWW", "DMIN",
    "E-AMRD", "E-AMRW", "E-AMRWW",
    "E-MMRD", "E-MMRW", "E-MMRWW",
    "E-DMIN", "E-WMIN", "E-WWMIN", "EV",
    "G-AMRD", "G-AMRW", "G-AMRWW",
    "G-MMRD", "G-MMRW", "G-MMRWW",
    "G-DMIN", "G-WMIN", "G-WWMIN", "GM",
    "MMRD",  "MMRwc", "MMRW", "MMRWW",
    "WMIN", "WWMIN",
    "eAMRd", "eAMRdc", "eAMRw", "eAMRwc",
    "eMMRd", "eMMRdc", "eMMRw", "eMMRwc",
    "gAMRd", "gAMRdc", "gAMRw", "gAMRwc",
    "gMMRd", "gMMRdc", "gMMRw", "gMMRwc"
])
const METHOD_DIRS = ["/" * m for m in ACTIVE_METHOD_DIRS]

# eps settings
const EPS_REGRET = SetRegretCore.EPS_DEFAULT
const EPS_SCAN   = 1e-6

# Optional per-PCM logs for minimax_regret (big files)
const WRITE_PCM_LOG = true

# Parallelization level
const PAR_LEVEL = :task

# -------------------------
# Shared helpers
# -------------------------
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m
@inline max_pairs(Alt::Int) = Alt * (Alt - 1) ÷ 2

# 2つの順位（代替案の並び）について、順序が一致するペア数（0..C(Alt,2)）を数える
function count_concordant_pairs(rank1::Vector{Int}, rank2::Vector{Int})
    n = length(rank1)
    pos2 = zeros(Int, n)
    @inbounds for (i, a) in enumerate(rank2)
        pos2[a] = i
    end
    cnt = 0
    @inbounds for i in 1:n-1
        ai = rank1[i]
        for j in i+1:n
            aj = rank1[j]
            cnt += (pos2[ai] < pos2[aj]) ? 1 : 0
        end
    end
    return cnt
end

@inline top1_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1])
@inline top2_comp_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1]) && (r1[2] == r2[2])
@inline function top2_include_ok(r1::Vector{Int}, r2::Vector{Int})
    a1,a2 = r1[1], r1[2]
    b1,b2 = r2[1], r2[2]
    return (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)
end

# 降順に並んだ区間端点 ts 上で、値 t が属する区間インデックス i を二分探索で返す
function find_interval_index(ts::Vector{Float64}, t::Float64)
    I = max(length(ts) - 1, 1)
    t_hi, t_lo = ts[1], ts[end]
    if t >= t_hi
        return 1
    elseif t <= t_lo
        return I
    end
    lo, hi = 1, I
    while lo <= hi
        mid = (lo + hi) >>> 1
        if ts[mid] >= t >= ts[mid+1]
            return mid
        elseif t > ts[mid]
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return clamp(lo, 1, I)
end

# 真の区間の中点に対応する予測区間を辿り、対角線上の一致ペア数平均（0..denom_pairs）を返す
function diagonal_mean_on_line(true_ts::Vector{Float64}, m_ts::Vector{Float64},
                               true_ranks::Vector{Vector{Int}}, m_ranks::Vector{Vector{Int}})
    # 対角線（真の区間の中点に対応する予測区間）上での一致ペア数の平均（0..denom_pairs）
    J = max(length(true_ts) - 1, 1)
    tU_true, tL_true = true_ts[1], true_ts[end]
    tU_pred, tL_pred = m_ts[1], m_ts[end]
    diag_sum = 0.0
    cnt = 0
    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])
        α = (t_true_mid - tU_true) / (tL_true - tU_true)
        t_pred_mid = tU_pred + α * (tL_pred - tU_pred)
        t_pred_mid = clamp(t_pred_mid, tL_pred, tU_pred)
        i = find_interval_index(m_ts, t_pred_mid)
        diag_sum += count_concordant_pairs(m_ranks[i], true_ranks[j])
        cnt += 1
    end
    return diag_sum / cnt
end

"""
case_metrics returns:
  precision_best, recall_best, f1_best,        # 0..denom_pairs（denom_pairs で割らない）
  precision_worst, recall_worst, f1_worst,     # 0..denom_pairs
  diag_mean, full_mean,                        # 0..denom_pairs
  top1_rate, top2_comp_rate, top2_include_rate,# 0..1（セル割合）
  top1_cnt, top2_comp_cnt, top2_include_cnt,   # セル件数
  total_cells                                 # I*J

注意:
  - I/J で割る（区間数で平均する）のは必須。
  - denom_pairs（=Alt*(Alt-1)/2）では割らない。
    例: Alt=5 なら denom_pairs=10 なので、
    「1000×100×10=1,000,000 組のうち何組一致したか」を素直に扱える。
"""
# 1ケース（真タイムライン vs 予測タイムライン）の評価指標をまとめて計算して返す（denom_pairs で割らない）
function case_metrics(true_ts::Vector{Float64}, true_ranks::Vector{Vector{Int}},
                      m_ts::Vector{Float64}, m_ranks::Vector{Vector{Int}})
    J = max(length(true_ranks) - 1, 1)
    I = max(length(m_ranks) - 1, 1)

    # Precision / Recall（一致ペア数の最大・最小マッチ）
    prec_best_sum = 0.0
    prec_worst_sum = 0.0
    for i in 1:I
        ri = m_ranks[i]
        best = -1
        worst = typemax(Int)
        for j in 1:J
            c = count_concordant_pairs(ri, true_ranks[j])
            if c > best; best = c; end
            if c < worst; worst = c; end
        end
        prec_best_sum += best
        prec_worst_sum += worst
    end
    precision_best = prec_best_sum / I
    precision_worst = prec_worst_sum / I

    rec_best_sum = 0.0
    rec_worst_sum = 0.0
    for j in 1:J
        rj = true_ranks[j]
        best = -1
        worst = typemax(Int)
        for i in 1:I
            c = count_concordant_pairs(m_ranks[i], rj)
            if c > best; best = c; end
            if c < worst; worst = c; end
        end
        rec_best_sum += best
        rec_worst_sum += worst
    end
    recall_best = rec_best_sum / J
    recall_worst = rec_worst_sum / J

    f1_best = (precision_best + recall_best) > 0 ? (2 * precision_best * recall_best / (precision_best + recall_best)) : 0.0
    f1_worst = (precision_worst + recall_worst) > 0 ? (2 * precision_worst * recall_worst / (precision_worst + recall_worst)) : 0.0

    # Diagonal / Full mean（一致ペア数の平均）
    diag_mean = diagonal_mean_on_line(true_ts, m_ts, true_ranks, m_ranks)

    total_cells = I * J
    top1_cnt = 0
    top2c_cnt = 0
    top2i_cnt = 0
    cell_sum = 0.0

    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            rj = true_ranks[j]
            top1_cnt  += top1_ok(ri, rj) ? 1 : 0
            top2c_cnt += top2_comp_ok(ri, rj) ? 1 : 0
            top2i_cnt += top2_include_ok(ri, rj) ? 1 : 0
            cell_sum  += count_concordant_pairs(ri, rj)
        end
    end

    full_mean = cell_sum / total_cells
    top1_rate = top1_cnt / total_cells
    top2c_rate = top2c_cnt / total_cells
    top2i_rate = top2i_cnt / total_cells

    return precision_best, recall_best, f1_best,
           precision_worst, recall_worst, f1_worst,
           diag_mean, full_mean,
           top1_rate, top2c_rate, top2i_rate,
           top1_cnt, top2c_cnt, top2i_cnt,
           total_cells
end
# -------------------------
# Logic: Regret (LPS)
# -------------------------
# LPSの結果 res.timeline から (ts, ranks) を取り出し、ts降順にそろえて返す
function points_from_res(res)
    ts = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    p = sortperm(ts; rev=true)
    return ts[p], ranks[p]
end


# 予測重み法1つを minimax_regret (LPS) で評価し、ケース合計を返す
# minimax_regret(LPS) で 1手法を全ケース評価し、合計値（sum_...）を返す。必要ならPCM単位ログも吐く
function summarize_one_regret(paths, utility::String, N::Int, tw::String, method::String;
                              eps::Float64=EPS_REGRET, pcm_log_dir::Union{Nothing,String}=nothing)
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)

    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        return nothing
    end

    repeat = min(REPEAT_NUM, length(methodW))
    repeat == 0 && return nothing

    # Accumulators（すべて denom_pairs で割らない 0..denom_pairs のまま）
    #  1 Pbest, 2 Rbest, 3 Fbest,
    #  4 Pworst,5 Rworst,6 Fworst,
    #  7 Diag, 8 Full,
    #  9 Top1(rate),10 Top2c(rate),11 Top2i(rate)
    acc = zeros(Float64, 11)

    # セル件数の合計（Top 系をセル数で重み付けしたいとき用）
    acc_cells = zeros(Int, 4)  # Top1_cnt, Top2c_cnt, Top2i_cnt, total_cells

    # interval/breakpoint counts
    acc_int = zeros(Int, 4) # TrueI, PredI, TrueB, PredB

    # Per-PCM (repeat) sums across utility matrices（optional）
    pcm_acc = pcm_log_dir === nothing ? nothing : zeros(Float64, 8, repeat)
    pcm_cells = pcm_log_dir === nothing ? nothing : zeros(Int, 4, repeat)

    cases = 0

    for utl_num in 1:UTILITY_MATRIX_NUM
        U = Matrix(utility_mats[utl_num])

        # true timeline (once per utility matrix)
        matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
        res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
        true_ts, true_ranks = points_from_res(res_true)
        true_I = max(length(true_ts) - 1, 1)
        true_B = max(length(true_ts) - 2, 0)

        for r in 1:repeat
            wL, wU = methodW[r].L, methodW[r].R
            tL, tU = SetRegretCore.find_optimal_trange(wL, wU)

            matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
            res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
            m_ts, m_ranks = points_from_res(res_m)

            p, rr, f1,
            pw, rw, f1w,
            d, full,
            t1, t2c, t2i,
            t1cnt, t2ccnt, t2icnt,
            total_cells = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

            acc[1] += p
            acc[2] += rr
            acc[3] += f1
            acc[4] += pw
            acc[5] += rw
            acc[6] += f1w
            acc[7] += d
            acc[8] += full
            acc[9] += t1
            acc[10] += t2c
            acc[11] += t2i

            acc_cells[1] += t1cnt
            acc_cells[2] += t2ccnt
            acc_cells[3] += t2icnt
            acc_cells[4] += total_cells

            acc_int[1] += true_I
            acc_int[2] += max(length(m_ts) - 1, 1)
            acc_int[3] += true_B
            acc_int[4] += max(length(m_ts) - 2, 0)

            if pcm_acc !== nothing
                pcm_acc[1, r] += p
                pcm_acc[2, r] += rr
                pcm_acc[3, r] += f1
                pcm_acc[4, r] += pw
                pcm_acc[5, r] += rw
                pcm_acc[6, r] += f1w
                pcm_acc[7, r] += d
                pcm_acc[8, r] += full

                pcm_cells[1, r] += t1cnt
                pcm_cells[2, r] += t2ccnt
                pcm_cells[3, r] += t2icnt
                pcm_cells[4, r] += total_cells
            end

            cases += 1
        end
    end

    cases == 0 && return nothing

    # Write per-PCM summaries for minimax_regret (mean across utility matrices)
    if pcm_acc !== nothing
        mkpath(pcm_log_dir)
        mname = method_clean(method)
        outpath = joinpath(pcm_log_dir,
            "pcm_minimax_regret__N=$(N)__tw=$(tw)__utility=$(utility)__method=$(mname).csv")

        denom_pairs = max_pairs(M)

        open(outpath, "w") do io
            println(io, join([
                "repeat_idx",
                "precision_best","recall_best","f1_best",
                "precision_worst","recall_worst","f1_worst",
                "diag_mean","full_mean",
                "top1_cnt","top2_comp_cnt","top2_include_cnt","total_cells",
                "denom_pairs"
            ], ','))

            for r in 1:repeat
                # mean across utility matrices (counts 0..denom_pairs)
                p = pcm_acc[1, r] / UTILITY_MATRIX_NUM
                rr = pcm_acc[2, r] / UTILITY_MATRIX_NUM
                f1 = pcm_acc[3, r] / UTILITY_MATRIX_NUM
                pw = pcm_acc[4, r] / UTILITY_MATRIX_NUM
                rw = pcm_acc[5, r] / UTILITY_MATRIX_NUM
                f1w = pcm_acc[6, r] / UTILITY_MATRIX_NUM
                d = pcm_acc[7, r] / UTILITY_MATRIX_NUM
                full = pcm_acc[8, r] / UTILITY_MATRIX_NUM

                println(io, join([
                    string(r),
                    @sprintf("%.10f", p),
                    @sprintf("%.10f", rr),
                    @sprintf("%.10f", f1),
                    @sprintf("%.10f", pw),
                    @sprintf("%.10f", rw),
                    @sprintf("%.10f", f1w),
                    @sprintf("%.10f", d),
                    @sprintf("%.10f", full),
                    string(pcm_cells[1, r]),
                    string(pcm_cells[2, r]),
                    string(pcm_cells[3, r]),
                    string(pcm_cells[4, r]),
                    string(denom_pairs)
                ], ','))
            end
        end
    end

    return (
        rule = "minimax_regret",
        utility = utility,
        N = N,
        tw = tw,
        method = method_clean(method),

        # 0..denom_pairs（denom_pairsで割らない）
        sum_precision = acc[1],
        sum_recall = acc[2],
        sum_f1 = acc[3],

        sum_precision_worst = acc[4],
        sum_recall_worst = acc[5],
        sum_f1_worst = acc[6],

        sum_diag_mean = acc[7],
        sum_full_mean = acc[8],

        # 0..1（ケース平均）
        sum_top1 = acc[9],
        sum_top2_comp = acc[10],
        sum_top2_include = acc[11],

        # セル数で重み付けしたいとき用の生カウント
        sum_top1_cnt = acc_cells[1],
        sum_top2_comp_cnt = acc_cells[2],
        sum_top2_include_cnt = acc_cells[3],
        sum_total_cells = acc_cells[4],

        sum_true_intervals = acc_int[1],
        sum_pred_intervals = acc_int[2],
        sum_true_breaks = acc_int[3],
        sum_pred_breaks = acc_int[4],
        cases = cases
    )
end

# -------------------------
# Logic: Maximin/Maximax (Scan)
# -------------------------
# 順位ベクトルの中で代替案 a と b を入れ替える（in-place）
function swap_in_rank!(rank::Vector{Int}, a::Int, b::Int)
    @inbounds for i in eachindex(rank)
        if rank[i] == a
            rank[i] = b
        elseif rank[i] == b
            rank[i] = a
        end
    end
    return rank
end

# maximin/maximax で使う、各代替案の基準並び替え順（降順/昇順）perm を作る
function build_perm(U::Matrix{Float64}, rule::Symbol)
    Alt, N = size(U)
    perm = Vector{Vector{Int}}(undef, Alt)
    rev = (rule == :maximax)
    for a in 1:Alt
        perm[a] = sortperm(@view U[a, :]; rev=rev)
    end
    return perm
end

# maximin/maximax の内部で、t固定時の最悪（または最良）配分 z と総効用 totalU を計算する
function maximin_totalU!(totalU::Vector{Float64}, z::Matrix{Float64}, star::Vector{Int},
                         U::Matrix{Float64}, yL::Vector{Float64}, yR::Vector{Float64}, perm::Vector{Vector{Int}})
    Alt, N = size(U)
    for k in 1:Alt
        cap = sum(yL)
        it = 1
        while it <= N - 1
            j = perm[k][it]
            diff = yR[j] - yL[j]
            if cap + diff <= 1.0 + 1e-12
                z[k, j] = yR[j]
                cap += diff
                it += 1
            else
                break
            end
        end
        j = perm[k][it]
        z[k, j] = 1.0 - cap + yL[j]
        star[k] = j
        it += 1
        while it <= N
            j = perm[k][it]
            z[k, j] = yL[j]
            it += 1
        end
        totalU[k] = sum(U[k, :] .* z[k, :])
    end
end

# maximin/maximax の順位変化をスキャンして (ts, ranks) を生成する
function scan_timeline_maximinmaximax(U::Matrix{Float64}, wL::Vector{Float64}, wR::Vector{Float64}, rule::Symbol;
                                      epsi::Float64=EPS_SCAN, max_events::Int=200)
    Alt, N = size(U)
    tL, tU = SetRegretCore.find_optimal_trange(wL, wR)
    perm = build_perm(U, rule)

    yL = zeros(N)
    yR = zeros(N)
    yL2 = zeros(N)
    yR2 = zeros(N)

    totalU = zeros(Alt)
    totalU2 = zeros(Alt)

    z = zeros(Alt, N)
    z2 = zeros(Alt, N)

    star = zeros(Int, Alt)
    star2 = zeros(Int, Alt)

    ts = Float64[tU]
    ranks = Vector{Vector{Int}}()

    # Initial t=tU
    yL .= wL .* tU
    yR .= wR .* tU
    fill!(z, 0.0)
    maximin_totalU!(totalU, z, star, U, yL, yR, perm)
    push!(ranks, sortperm(totalU; rev=true))

    t_snap = tU
    iter = 0

    while t_snap > tL + 1e-15
        iter += 1
        iter > max_events && break

        yL .= wL .* t_snap
        yR .= wR .* t_snap
        fill!(z, 0.0)
        maximin_totalU!(totalU, z, star, U, yL, yR, perm)

        Sl = Inf
        for a in 1:Alt
            Sl = min(Sl, yR[star[a]] - z[a, star[a]])
        end

        r = 1.0 / (1.0 + Sl)
        if r == 1.0
            r -= epsi
        end
        if r * t_snap < tL
            r = tL / t_snap
        end

        yL2 .= yL .* r
        yR2 .= yR .* r
        fill!(z2, 0.0)
        maximin_totalU!(totalU2, z2, star2, U, yL2, yR2, perm)

        crossings = Vector{Tuple{Float64, Int, Int}}()
        for i in 1:Alt-1, j in i+1:Alt
            if (totalU[i] - totalU[j]) * (totalU2[i] - totalU2[j]) <= 0.0
                denom = U[i, star[i]] - U[j, star[j]]
                if abs(denom) > 1e-14
                    push!(crossings, (1.0 / (1.0 + -(totalU[i] - totalU[j]) / denom), i, j))
                end
            end
        end
        sort!(crossings, by = x -> x[1])

        current_rank = copy(ranks[end])
        if !isempty(crossings)
            for (r2, i, j) in reverse(crossings)
                push!(ts, t_snap * r2)
                swap_in_rank!(current_rank, i, j)
                push!(ranks, copy(current_rank))
            end
        end

        t_snap = t_snap * r
        if t_snap <= tL + 1e-15
            break
        end
    end

    if isempty(ts) || ts[end] > tL + 1e-15
        push!(ts, tL)
        push!(ranks, copy(ranks[end]))
    end

    return ts, ranks
end

# 1手法を maximin/maximax のスキャンで評価し、ケース合計を返す
# maximin/maximax(スキャン) で 1手法を全ケース評価し、合計値（sum_...）を返す
function summarize_one_scan(paths, rule::Symbol, utility::String, N::Int, tw::String, method::String;
                            epsi::Float64=EPS_SCAN)
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)

    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        return nothing
    end

    repeat = min(REPEAT_NUM, length(methodW))
    repeat == 0 && return nothing

    # Accumulators（すべて denom_pairs で割らない 0..denom_pairs のまま）
    #  1 Pbest, 2 Rbest, 3 Fbest,
    #  4 Pworst,5 Rworst,6 Fworst,
    #  7 Diag, 8 Full,
    #  9 Top1(rate),10 Top2c(rate),11 Top2i(rate)
    acc = zeros(Float64, 11)
    acc_cells = zeros(Int, 4)  # Top1_cnt, Top2c_cnt, Top2i_cnt, total_cells

    acc_int = zeros(Int, 4) # TrueI, PredI, TrueB, PredB
    cases = 0

    for utl_num in 1:UTILITY_MATRIX_NUM
        U = Matrix(utility_mats[utl_num])

        true_ts, true_ranks = scan_timeline_maximinmaximax(U, trueW.L, trueW.R, rule; epsi=epsi)
        true_I = max(length(true_ts) - 1, 1)
        true_B = max(length(true_ts) - 2, 0)

        for r in 1:repeat
            m_ts, m_ranks = scan_timeline_maximinmaximax(U, methodW[r].L, methodW[r].R, rule; epsi=epsi)

            p, rr, f1,
            pw, rw, f1w,
            d, full,
            t1, t2c, t2i,
            t1cnt, t2ccnt, t2icnt,
            total_cells = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

            acc[1] += p
            acc[2] += rr
            acc[3] += f1
            acc[4] += pw
            acc[5] += rw
            acc[6] += f1w
            acc[7] += d
            acc[8] += full
            acc[9] += t1
            acc[10] += t2c
            acc[11] += t2i

            acc_cells[1] += t1cnt
            acc_cells[2] += t2ccnt
            acc_cells[3] += t2icnt
            acc_cells[4] += total_cells

            acc_int[1] += true_I
            acc_int[2] += max(length(m_ts) - 1, 1)
            acc_int[3] += true_B
            acc_int[4] += max(length(m_ts) - 2, 0)
            cases += 1
        end
    end

    cases == 0 && return nothing

    return (
        rule = String(rule),
        utility = utility,
        N = N,
        tw = tw,
        method = method_clean(method),

        sum_precision = acc[1],
        sum_recall = acc[2],
        sum_f1 = acc[3],

        sum_precision_worst = acc[4],
        sum_recall_worst = acc[5],
        sum_f1_worst = acc[6],

        sum_diag_mean = acc[7],
        sum_full_mean = acc[8],

        sum_top1 = acc[9],
        sum_top2_comp = acc[10],
        sum_top2_include = acc[11],

        sum_top1_cnt = acc_cells[1],
        sum_top2_comp_cnt = acc_cells[2],
        sum_top2_include_cnt = acc_cells[3],
        sum_total_cells = acc_cells[4],

        sum_true_intervals = acc_int[1],
        sum_pred_intervals = acc_int[2],
        sum_true_breaks = acc_int[3],
        sum_pred_breaks = acc_int[4],
        cases = cases
    )
end

# -------------------------
# PCM log merge helper
# -------------------------
# 並列実行で分割出力されたPCMログCSVを、ヘッダ1つにまとめて結合する
function merge_pcm_logs(pcm_log_dir::String, outpath::String)
    files = sort(filter(f -> endswith(f, ".csv"), readdir(pcm_log_dir; join=true)))
    isempty(files) && return false

    open(outpath, "w") do out
        first = true
        for f in files
            open(f, "r") do io
                header = readline(io)
                if first
                    println(out, header)
                    first = false
                end
                for line in eachline(io)
                    println(out, line)
                end
            end
        end
    end
    return true
end

# -------------------------
# Master entry
# -------------------------
# 全ルール×条件×手法を並列実行して結果を集め、整列して返す（必要ならPCMログも作成）
function collect_all_results(pcm_log_dir::Union{Nothing,String})
    paths = Paths.project_paths()

    tasks_regret = [(utility, N, tw, m) for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]
    rules = (:maximin, :maximax)
    tasks_scan = [(rule, utility, N, tw, m) for rule in rules for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]

    total_tasks = length(tasks_regret) + length(tasks_scan)
    all_results = Vector{Any}(undef, total_tasks)

    @info "Starting unified processing (v4)" total_tasks threads=nthreads()

    @threads for idx in 1:total_tasks
        if idx <= length(tasks_regret)
            (utility, N, tw, method) = tasks_regret[idx]
            res = summarize_one_regret(paths, utility, N, tw, method; pcm_log_dir=pcm_log_dir)
            all_results[idx] = res
        else
            local_idx = idx - length(tasks_regret)
            (rule, utility, N, tw, method) = tasks_scan[local_idx]
            res = summarize_one_scan(paths, rule, utility, N, tw, method)
            all_results[idx] = res
        end

        if idx % 100 == 0
            @info "Progress" idx total_tasks
        end
    end

    valid_rows = [r for r in all_results if r !== nothing]

    # Sort: rule -> N -> tw -> utility -> method
    rule_order = Dict("minimax_regret"=>1, "maximin"=>2, "maximax"=>3)
    sort!(valid_rows, by = r -> (get(rule_order, r.rule, 99), r.N, r.tw, r.utility, r.method))

    return valid_rows
end

# 実行エントリ：結果CSVを書き出し、必要ならPCMログもマージする
function main()
    paths = Paths.project_paths()
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)

    pcm_log_dir = nothing
    if WRITE_PCM_LOG
        pcm_log_dir = joinpath(outdir, "pcm_logs")
        mkpath(pcm_log_dir)
    end

    rows = collect_all_results(pcm_log_dir)

    outpath = joinpath(outdir, "grid_summary_ALL_RULES_v4.csv")

    open(outpath, "w") do io
        println(io, join([
            "rule","N","tw","utility","method",
            "sum_precision","sum_recall","sum_f1",
            "sum_precision_worst","sum_recall_worst","sum_f1_worst",
            "sum_diag_mean","sum_full_mean",
            "sum_top1","sum_top2_comp","sum_top2_include",
            "sum_top1_cnt","sum_top2_comp_cnt","sum_top2_include_cnt","sum_total_cells",
            "sum_true_intervals","sum_pred_intervals","sum_true_breaks","sum_pred_breaks",
            "cases"
        ], ','))

        for r in rows
            println(io, join([
                r.rule,
                string(r.N),
                r.tw,
                r.utility,
                r.method,

                @sprintf("%.10f", r.sum_precision),
                @sprintf("%.10f", r.sum_recall),
                @sprintf("%.10f", r.sum_f1),

                @sprintf("%.10f", r.sum_precision_worst),
                @sprintf("%.10f", r.sum_recall_worst),
                @sprintf("%.10f", r.sum_f1_worst),


                @sprintf("%.10f", r.sum_diag_mean),
                @sprintf("%.10f", r.sum_full_mean),

                @sprintf("%.10f", r.sum_top1),
                @sprintf("%.10f", r.sum_top2_comp),
                @sprintf("%.10f", r.sum_top2_include),

                string(r.sum_top1_cnt),
                string(r.sum_top2_comp_cnt),
                string(r.sum_top2_include_cnt),
                string(r.sum_total_cells),

                string(r.sum_true_intervals),
                string(r.sum_pred_intervals),
                string(r.sum_true_breaks),
                string(r.sum_pred_breaks),

                string(r.cases)
            ], ','))
        end
    end

    @info "Saved summary" outpath rows=length(rows)

    if WRITE_PCM_LOG && pcm_log_dir !== nothing
        merged = joinpath(outdir, "pcm_summary_minimax_regret_v4.csv")
        ok = merge_pcm_logs(pcm_log_dir, merged)
        @info "Merged PCM logs" merged ok
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
