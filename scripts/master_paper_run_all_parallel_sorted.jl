# scripts/master_paper_run_all_parallel_sorted.jl
#
# One-shot runner for:
#   (1) minimax_regret (LPS)
#   (2) maximin / maximax (scan)
#
# It writes only aggregated "sum_*" metrics (no raw grids), plus:
#   - sum_true_intervals, sum_pred_intervals
#   - sum_true_breaks,    sum_pred_breaks
#
# Output files (overwritten each run):
#   data/metrics_julia/grid_summary_minimax_regret_v3.csv
#   data/metrics_julia/grid_summary_maximinmaximax_v5.csv
#
# Parallelization:
#   Default: parallelize over (utility, N, tw, method[, rule]) tasks (like your v2 regret script),
#   and DO NOT do nested Threads.@threads inside the per-task loop.
#   If you want, you can switch PAR_LEVEL_* to :utility to parallelize within each task.
#
# Sorted CSV output:
#   This script collects all rows in memory, sorts by keys, then writes once.
#
# Run:
#   julia --project=. scripts/master_paper_run_all_parallel_sorted.jl

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

# Deduplicated method list (original scripts had duplicates for DMIN/WMIN).
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

# Parallelization level:
#   :task    -> Threads.@threads over tasks, inner loops serial (default; avoids nested threading)
#   :utility -> outer loop serial, Threads.@threads over utility matrices inside each task
const PAR_LEVEL_REGRET = :task
const PAR_LEVEL_SCAN   = :task

# -------------------------
# Shared helpers (evaluation)
# -------------------------
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m

@inline function max_pairs(Alt::Int)
    return Alt * (Alt - 1) ÷ 2
end

# rank1 の順序関係が rank2 と一致するペア数
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

# top1/top2判定（セル単位）
@inline top1_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1])
@inline top2_comp_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1]) && (r1[2] == r2[2])
@inline function top2_include_ok(r1::Vector{Int}, r2::Vector{Int})
    a1,a2 = r1[1], r1[2]
    b1,b2 = r2[1], r2[2]
    return (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)
end

# ts は降順。区間 i は [ts[i], ts[i+1]]。
# t がどの区間に属するか i を返す（1..I）。外側は端に丸める。
function find_interval_index(ts::Vector{Float64}, t::Float64)
    I = max(length(ts) - 1, 1)
    t_hi = ts[1]
    t_lo = ts[end]
    if t >= t_hi
        return 1
    elseif t <= t_lo
        return I
    end
    lo = 1
    hi = I
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

# diagonal mean: (true tU→tL) と (pred tU→tL) を結ぶ直線に沿うセルを拾う
function diagonal_mean_on_line(true_ts::Vector{Float64}, m_ts::Vector{Float64},
                               true_ranks::Vector{Vector{Int}}, m_ranks::Vector{Vector{Int}})
    J = max(length(true_ts) - 1, 1)  # true columns (interval count)
    I = max(length(m_ts) - 1, 1)     # pred rows   (interval count)

    tU_true = true_ts[1]
    tL_true = true_ts[end]
    tU_pred = m_ts[1]
    tL_pred = m_ts[end]

    denom_pairs = max_pairs(length(true_ranks[1]))

    diag_sum = 0.0
    cnt = 0
    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])
        α = (t_true_mid - tU_true) / (tL_true - tU_true)
        t_pred_mid = tU_pred + α * (tL_pred - tU_pred)

        if t_pred_mid > tU_pred
            t_pred_mid = tU_pred
        elseif t_pred_mid < tL_pred
            t_pred_mid = tL_pred
        end

        i = find_interval_index(m_ts, t_pred_mid)
        c = count_concordant_pairs(m_ranks[i], true_ranks[j])
        diag_sum += c
        cnt += 1
    end
    return (diag_sum / cnt) / denom_pairs
end

# 1ケース（1つのU, 1つの推定重み）で指標を返す（raw格子保存なし）
function case_metrics(true_ts::Vector{Float64}, true_ranks::Vector{Vector{Int}},
                      m_ts::Vector{Float64}, m_ranks::Vector{Vector{Int}})
    true_cnt = length(true_ranks)
    m_cnt = length(m_ranks)

    # NOTE: last row/col are duplicated due to endpoint inclusion
    J = max(true_cnt - 1, 1)  # true columns
    I = max(m_cnt - 1, 1)     # pred rows

    Alt = length(true_ranks[1])
    denom_pairs = max_pairs(Alt)

    # precision
    prec_sum = 0.0
    for i in 1:I
        best = -1
        ri = m_ranks[i]
        for j in 1:J
            c = count_concordant_pairs(ri, true_ranks[j])
            if c > best
                best = c
            end
        end
        prec_sum += best
    end
    precision = (prec_sum / I) / denom_pairs

    # recall
    rec_sum = 0.0
    for j in 1:J
        best = -1
        rj = true_ranks[j]
        for i in 1:I
            c = count_concordant_pairs(m_ranks[i], rj)
            if c > best
                best = c
            end
        end
        rec_sum += best
    end
    recall = (rec_sum / J) / denom_pairs

    # F1 per case
    f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0.0

    # diagonal mean on line
    diag_mean = diagonal_mean_on_line(true_ts, m_ts, true_ranks, m_ranks)

    # top-k ratios
    total_cells = I * J
    top1_cnt = 0
    top2c_cnt = 0
    top2i_cnt = 0
    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            rj = true_ranks[j]
            top1_cnt  += top1_ok(ri, rj) ? 1 : 0
            top2c_cnt += top2_comp_ok(ri, rj) ? 1 : 0
            top2i_cnt += top2_include_ok(ri, rj) ? 1 : 0
        end
    end
    top1_ratio  = top1_cnt  / total_cells
    top2c_ratio = top2c_cnt / total_cells
    top2i_ratio = top2i_cnt / total_cells

    # full mean
    cell_sum = 0.0
    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            cell_sum += count_concordant_pairs(ri, true_ranks[j])
        end
    end
    full_mean = (cell_sum / (I * J)) / denom_pairs

    return precision, recall, f1, diag_mean, full_mean, top1_ratio, top2c_ratio, top2i_ratio
end

# -------------------------
# (1) minimax_regret summary (LPS)
# -------------------------
function points_from_res(res)
    ts    = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    p = sortperm(ts; rev=true)
    return ts[p], ranks[p]
end

# summarize one condition (utility, N, tw, method) for minimax_regret
function summarize_one_regret(paths, utility::String, N::Int, tw::String, method::String; eps::Float64=EPS_REGRET)
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)

    trueW = LoadInstance.read_true_weights(paths, tw; N=N)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)

    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        @warn "NO_WEIGHT" "minimax_regret" utility N tw method
        return nothing
    end
    repeat = min(REPEAT_NUM, length(methodW))
    repeat == 0 && return nothing

    # accumulators
    sumFull = 0.0; sumP = 0.0; sumR = 0.0; sumF = 0.0; sumD = 0.0
    sumT1 = 0.0; sumT2c = 0.0; sumT2i = 0.0
    cases = 0

    # NEW: interval / breakpoint counters
    sum_true_intervals = 0
    sum_pred_intervals = 0
    sum_true_breaks    = 0
    sum_pred_breaks    = 0

    # choose parallel level
    if PAR_LEVEL_REGRET == :utility
        T = nthreads()
        sumFull_t = zeros(Float64, T)
        sumP_t    = zeros(Float64, T)
        sumR_t    = zeros(Float64, T)
        sumF_t    = zeros(Float64, T)
        sumD_t    = zeros(Float64, T)
        sumT1_t   = zeros(Float64, T)
        sumT2c_t  = zeros(Float64, T)
        sumT2i_t  = zeros(Float64, T)
        cases_t   = zeros(Int, T)

        sum_true_intervals_t = zeros(Int, T)
        sum_pred_intervals_t = zeros(Int, T)
        sum_true_breaks_t    = zeros(Int, T)
        sum_pred_breaks_t    = zeros(Int, T)

        @threads for utl_num in 1:UTILITY_MATRIX_NUM
            tid = threadid()
            U = Matrix(utility_mats[utl_num])

            matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
            res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
            true_ts, true_ranks = points_from_res(res_true)

            true_I = max(length(true_ts) - 1, 1)
            true_B = max(length(true_ts) - 2, 0)

            for r in 1:repeat
                wL = methodW[r].L
                wU = methodW[r].R
                tL, tU = SetRegretCore.find_optimal_trange(wL, wU)

                matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
                res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
                m_ts, m_ranks = points_from_res(res_m)

                p, rr, f1, d, full, t1, t2c, t2i = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

                pred_I = max(length(m_ts) - 1, 1)
                pred_B = max(length(m_ts) - 2, 0)
                sum_true_intervals_t[tid] += true_I
                sum_pred_intervals_t[tid] += pred_I
                sum_true_breaks_t[tid]    += true_B
                sum_pred_breaks_t[tid]    += pred_B

                sumFull_t[tid] += full
                sumP_t[tid]    += p
                sumR_t[tid]    += rr
                sumF_t[tid]    += f1
                sumD_t[tid]    += d
                sumT1_t[tid]   += t1
                sumT2c_t[tid]  += t2c
                sumT2i_t[tid]  += t2i
                cases_t[tid]   += 1
            end
        end

        cases = sum(cases_t)
        cases == 0 && return nothing
        sumFull = sum(sumFull_t); sumP = sum(sumP_t); sumR = sum(sumR_t); sumF = sum(sumF_t); sumD = sum(sumD_t)
        sumT1 = sum(sumT1_t); sumT2c = sum(sumT2c_t); sumT2i = sum(sumT2i_t)
        sum_true_intervals = sum(sum_true_intervals_t)
        sum_pred_intervals = sum(sum_pred_intervals_t)
        sum_true_breaks    = sum(sum_true_breaks_t)
        sum_pred_breaks    = sum(sum_pred_breaks_t)

    else
        # PAR_LEVEL_REGRET == :task (default): inner loops serial
        for utl_num in 1:UTILITY_MATRIX_NUM
            U = Matrix(utility_mats[utl_num])

            matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
            res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
            true_ts, true_ranks = points_from_res(res_true)

            true_I = max(length(true_ts) - 1, 1)
            true_B = max(length(true_ts) - 2, 0)

            for r in 1:repeat
                wL = methodW[r].L
                wU = methodW[r].R
                tL, tU = SetRegretCore.find_optimal_trange(wL, wU)

                matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
                res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
                m_ts, m_ranks = points_from_res(res_m)

                p, rr, f1, d, full, t1, t2c, t2i = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

                pred_I = max(length(m_ts) - 1, 1)
                pred_B = max(length(m_ts) - 2, 0)
                sum_true_intervals += true_I
                sum_pred_intervals += pred_I
                sum_true_breaks    += true_B
                sum_pred_breaks    += pred_B

                sumFull += full
                sumP    += p
                sumR    += rr
                sumF    += f1
                sumD    += d
                sumT1   += t1
                sumT2c  += t2c
                sumT2i  += t2i
                cases   += 1
            end
        end
        cases == 0 && return nothing
    end

    return (
        rule              = "minimax_regret",
        utility           = utility,
        N                 = N,
        tw                = tw,
        method            = method_clean(method),
        sum_precision     = sumP,
        sum_recall        = sumR,
        sum_f1            = sumF,
        sum_diag_mean     = sumD,
        sum_full_mean     = sumFull,
        sum_top1          = sumT1,
        sum_top2_comp     = sumT2c,
        sum_top2_include  = sumT2i,
        sum_true_intervals= sum_true_intervals,
        sum_pred_intervals= sum_pred_intervals,
        sum_true_breaks   = sum_true_breaks,
        sum_pred_breaks   = sum_pred_breaks,
        cases             = cases
    )
end

function run_minimax_regret_grid_summaries_v3_sorted(; outname::String="grid_summary_minimax_regret_v3.csv")
    paths = Paths.project_paths()
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)
    outpath = joinpath(outdir, outname)

    tasks = [(utility, N, tw, m) for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]
    results = Vector{Any}(undef, length(tasks))

    if PAR_LEVEL_REGRET == :task
        @threads for idx in eachindex(tasks)
            (utility, N, tw, method) = tasks[idx]
            res = summarize_one_regret(paths, utility, N, tw, method)
            results[idx] = res
            res === nothing || @info "regret task done" utility N tw method tid=threadid() cases=res.cases
        end
    else
        for idx in eachindex(tasks)
            (utility, N, tw, method) = tasks[idx]
            results[idx] = summarize_one_regret(paths, utility, N, tw, method)
        end
    end

    rows = [r for r in results if r !== nothing]
    sort!(rows, by = r -> (r.utility, r.N, r.tw, r.method))

    open(outpath, "w") do io
        println(io, join([
            "rule","N","tw","utility","method",
            "sum_precision","sum_recall","sum_f1","sum_diag_mean","sum_full_mean",
            "sum_top1","sum_top2_comp","sum_top2_include",
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
                @sprintf("%.10f", r.sum_diag_mean),
                @sprintf("%.10f", r.sum_full_mean),
                @sprintf("%.10f", r.sum_top1),
                @sprintf("%.10f", r.sum_top2_comp),
                @sprintf("%.10f", r.sum_top2_include),
                string(r.sum_true_intervals),
                string(r.sum_pred_intervals),
                string(r.sum_true_breaks),
                string(r.sum_pred_breaks),
                string(r.cases)
            ], ','))
        end
    end

    @info "regret saved (sorted)" outpath nrows=length(rows)
    return outpath
end

# -------------------------
# (2) maximin / maximax summary (scan)
# -------------------------
@inline function swap_in_rank!(rank::Vector{Int}, a::Int, b::Int)
    @inbounds for i in eachindex(rank)
        if rank[i] == a
            rank[i] = b
        elseif rank[i] == b
            rank[i] = a
        end
    end
    return rank
end

# perm: alternative-specific permutation of criteria (maximin: ascending, maximax: descending)
function build_perm(U::Matrix{Float64}, rule::Symbol)
    Alt, N = size(U)
    perm = Vector{Vector{Int}}(undef, Alt)
    rev = (rule == :maximax)
    @inbounds for a in 1:Alt
        perm[a] = sortperm(@view U[a, :]; rev=rev)
    end
    return perm
end

# C++ maximin() equivalent: compute totalU and star (for each alternative)
function maximin_totalU!(totalU::Vector{Float64}, z::Matrix{Float64}, star::Vector{Int},
                         U::Matrix{Float64}, yL::Vector{Float64}, yR::Vector{Float64},
                         perm::Vector{Vector{Int}})
    Alt, N = size(U)
    @inbounds for k in 1:Alt
        cap = 0.0
        for i in 1:N
            cap += yL[i]
        end

        it = 1
        while it <= N-1
            j = perm[k][it]
            if cap + (yR[j] - yL[j]) <= 1.0 + 1e-12
                z[k, j] = yR[j]
                cap += (yR[j] - yL[j])
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

        s = 0.0
        for i in 1:N
            s += U[k, i] * z[k, i]
        end
        totalU[k] = s
    end
    return nothing
end

# Scan timeline for maximin/maximax:
# returns (ts, ranks) where ts includes endpoints and all rank-change points.
function scan_timeline_maximinmaximax(U::Matrix{Float64}, wL::Vector{Float64}, wR::Vector{Float64}, rule::Symbol;
                                      epsi::Float64=EPS_SCAN, max_events::Int=200)
    Alt, N = size(U)
    tL, tU = SetRegretCore.find_optimal_trange(wL, wR)

    perm = build_perm(U, rule)

    yL  = zeros(Float64, N)
    yR  = zeros(Float64, N)
    yL2 = zeros(Float64, N)
    yR2 = zeros(Float64, N)

    totalU  = zeros(Float64, Alt)
    totalU2 = zeros(Float64, Alt)
    z   = zeros(Float64, Alt, N)
    z2  = zeros(Float64, Alt, N)
    star  = zeros(Int, Alt)
    star2 = zeros(Int, Alt)

    ts = Float64[tU]
    ranks = Vector{Vector{Int}}()

    # initial at tU
    @inbounds for i in 1:N
        yL[i] = wL[i] * tU
        yR[i] = wR[i] * tU
    end
    fill!(z, 0.0)
    maximin_totalU!(totalU, z, star, U, yL, yR, perm)
    rank = sortperm(totalU; rev=true)
    push!(ranks, copy(rank))

    t_snap = tU
    iter = 0
    while t_snap > tL + 1e-15
        iter += 1
        iter > max_events && break

        @inbounds for i in 1:N
            yL[i] = wL[i] * t_snap
            yR[i] = wR[i] * t_snap
        end
        fill!(z, 0.0)
        maximin_totalU!(totalU, z, star, U, yL, yR, perm)

        # next fold
        Sl = Inf
        for a in 1:Alt
            s = yR[star[a]] - z[a, star[a]]
            Sl = min(Sl, s)
        end
        r = 1.0 / (1.0 + Sl)
        if r == 1.0
            r -= epsi
        end
        if r * t_snap < tL
            r = tL / t_snap
        end
        t_fold = t_snap * r

        # values at t_fold
        @inbounds for i in 1:N
            yL2[i] = yL[i] * r
            yR2[i] = yR[i] * r
        end
        fill!(z2, 0.0)
        maximin_totalU!(totalU2, z2, star2, U, yL2, yR2, perm)

        # crossings within [t_fold, t_snap]
        crossings = Vector{Tuple{Float64,Int,Int}}()  # (r2, i, j)
        @inbounds for i in 1:Alt-1
            for j in i+1:Alt
                if (totalU[i] - totalU[j]) * (totalU2[i] - totalU2[j]) <= 0.0
                    denom = U[i, star[i]] - U[j, star[j]]
                    if abs(denom) < 1e-14
                        continue
                    end
                    S = -(totalU[i] - totalU[j]) / denom
                    r2 = 1.0 / (1.0 + S)
                    push!(crossings, (r2, i, j))
                end
            end
        end
        sort!(crossings, by=x->x[1])  # r2 ascending

        current_rank = copy(ranks[end])
        if !isempty(crossings)
            # t decreases, so apply larger r2 first (larger t_cross first)
            for (r2, i, j) in reverse(crossings)
                t_cross = t_snap * r2
                push!(ts, t_cross)
                swap_in_rank!(current_rank, i, j)
                push!(ranks, copy(current_rank))
            end
        end

        # advance to fold (fold itself is not stored)
        t_snap = t_fold
        if t_snap <= tL + 1e-15
            break
        end
    end

    # endpoint tL
    if isempty(ts) || ts[end] > tL + 1e-15
        push!(ts, tL)
        push!(ranks, copy(ranks[end]))
    end

    return ts, ranks
end

# summarize one condition (rule, utility, N, tw, method) for scan rules
function summarize_one_scan(paths, rule::Symbol, utility::String, N::Int, tw::String, method::String; epsi::Float64=EPS_SCAN)
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)

    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        @warn "NO_WEIGHT" String(rule) utility N tw method
        return nothing
    end
    repeat = min(REPEAT_NUM, length(methodW))
    repeat == 0 && return nothing

    sumFull = 0.0; sumP = 0.0; sumR = 0.0; sumF = 0.0; sumD = 0.0
    sumT1 = 0.0; sumT2c = 0.0; sumT2i = 0.0
    cases = 0

    sum_true_intervals = 0
    sum_pred_intervals = 0
    sum_true_breaks    = 0
    sum_pred_breaks    = 0

    if PAR_LEVEL_SCAN == :utility
        T = nthreads()
        sumFull_t = zeros(Float64, T)
        sumP_t    = zeros(Float64, T)
        sumR_t    = zeros(Float64, T)
        sumF_t    = zeros(Float64, T)
        sumD_t    = zeros(Float64, T)
        sumT1_t   = zeros(Float64, T)
        sumT2c_t  = zeros(Float64, T)
        sumT2i_t  = zeros(Float64, T)
        cases_t   = zeros(Int, T)

        sum_true_intervals_t = zeros(Int, T)
        sum_pred_intervals_t = zeros(Int, T)
        sum_true_breaks_t    = zeros(Int, T)
        sum_pred_breaks_t    = zeros(Int, T)

        @threads for utl_num in 1:UTILITY_MATRIX_NUM
            tid = threadid()
            U = Matrix(utility_mats[utl_num])

            true_ts, true_ranks = scan_timeline_maximinmaximax(U, trueW.L, trueW.R, rule; epsi=epsi)
            true_I = max(length(true_ts) - 1, 1)
            true_B = max(length(true_ts) - 2, 0)

            for r in 1:repeat
                wL = methodW[r].L
                wU = methodW[r].R

                m_ts, m_ranks = scan_timeline_maximinmaximax(U, wL, wU, rule; epsi=epsi)
                p, rr, f1, d, fm, t1, t2c, t2i = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

                pred_I = max(length(m_ts) - 1, 1)
                pred_B = max(length(m_ts) - 2, 0)
                sum_true_intervals_t[tid] += true_I
                sum_pred_intervals_t[tid] += pred_I
                sum_true_breaks_t[tid]    += true_B
                sum_pred_breaks_t[tid]    += pred_B

                sumFull_t[tid] += fm
                sumP_t[tid]    += p
                sumR_t[tid]    += rr
                sumF_t[tid]    += f1
                sumD_t[tid]    += d
                sumT1_t[tid]   += t1
                sumT2c_t[tid]  += t2c
                sumT2i_t[tid]  += t2i
                cases_t[tid]   += 1
            end
        end

        cases = sum(cases_t)
        cases == 0 && return nothing
        sumFull = sum(sumFull_t); sumP = sum(sumP_t); sumR = sum(sumR_t); sumF = sum(sumF_t); sumD = sum(sumD_t)
        sumT1 = sum(sumT1_t); sumT2c = sum(sumT2c_t); sumT2i = sum(sumT2i_t)
        sum_true_intervals = sum(sum_true_intervals_t)
        sum_pred_intervals = sum(sum_pred_intervals_t)
        sum_true_breaks    = sum(sum_true_breaks_t)
        sum_pred_breaks    = sum(sum_pred_breaks_t)

    else
        for utl_num in 1:UTILITY_MATRIX_NUM
            U = Matrix(utility_mats[utl_num])

            true_ts, true_ranks = scan_timeline_maximinmaximax(U, trueW.L, trueW.R, rule; epsi=epsi)
            true_I = max(length(true_ts) - 1, 1)
            true_B = max(length(true_ts) - 2, 0)

            for r in 1:repeat
                wL = methodW[r].L
                wU = methodW[r].R

                m_ts, m_ranks = scan_timeline_maximinmaximax(U, wL, wU, rule; epsi=epsi)
                p, rr, f1, d, fm, t1, t2c, t2i = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

                pred_I = max(length(m_ts) - 1, 1)
                pred_B = max(length(m_ts) - 2, 0)
                sum_true_intervals += true_I
                sum_pred_intervals += pred_I
                sum_true_breaks    += true_B
                sum_pred_breaks    += pred_B

                sumFull += fm
                sumP    += p
                sumR    += rr
                sumF    += f1
                sumD    += d
                sumT1   += t1
                sumT2c  += t2c
                sumT2i  += t2i
                cases   += 1
            end
        end
        cases == 0 && return nothing
    end

    return (
        rule              = String(rule),
        utility           = utility,
        N                 = N,
        tw                = tw,
        method            = method_clean(method),
        sum_precision     = sumP,
        sum_recall        = sumR,
        sum_f1            = sumF,
        sum_diag_mean     = sumD,
        sum_full_mean     = sumFull,
        sum_top1          = sumT1,
        sum_top2_comp     = sumT2c,
        sum_top2_include  = sumT2i,
        sum_true_intervals= sum_true_intervals,
        sum_pred_intervals= sum_pred_intervals,
        sum_true_breaks   = sum_true_breaks,
        sum_pred_breaks   = sum_pred_breaks,
        cases             = cases
    )
end

function run_maximinmaximax_grid_summaries_v5_sorted(; outname::String="grid_summary_maximinmaximax_v5.csv")
    paths = Paths.project_paths()
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)
    outpath = joinpath(outdir, outname)

    rules = (:maximin, :maximax)
    tasks = [(rule, utility, N, tw, m) for rule in rules for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]
    results = Vector{Any}(undef, length(tasks))

    if PAR_LEVEL_SCAN == :task
        @threads for idx in eachindex(tasks)
            (rule, utility, N, tw, method) = tasks[idx]
            res = summarize_one_scan(paths, rule, utility, N, tw, method)
            results[idx] = res
            res === nothing || @info "scan task done" rule utility N tw method tid=threadid() cases=res.cases
        end
    else
        for idx in eachindex(tasks)
            (rule, utility, N, tw, method) = tasks[idx]
            results[idx] = summarize_one_scan(paths, rule, utility, N, tw, method)
        end
    end

    rows = [r for r in results if r !== nothing]
    rule_order = Dict("maximin"=>1, "maximax"=>2)
    sort!(rows, by = r -> (rule_order[r.rule], r.utility, r.N, r.tw, r.method))

    open(outpath, "w") do io
        println(io, join([
            "rule","N","tw","utility","method",
            "sum_precision","sum_recall","sum_f1","sum_diag_mean","sum_full_mean",
            "sum_top1","sum_top2_comp","sum_top2_include",
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
                @sprintf("%.10f", r.sum_diag_mean),
                @sprintf("%.10f", r.sum_full_mean),
                @sprintf("%.10f", r.sum_top1),
                @sprintf("%.10f", r.sum_top2_comp),
                @sprintf("%.10f", r.sum_top2_include),
                string(r.sum_true_intervals),
                string(r.sum_pred_intervals),
                string(r.sum_true_breaks),
                string(r.sum_pred_breaks),
                string(r.cases)
            ], ','))
        end
    end

    @info "maximin/maximax saved (sorted)" outpath nrows=length(rows)
    return outpath
end

# -------------------------
# Master entry
# -------------------------
function main()
    @info "master runner start" nthreads=nthreads() par_regret=PAR_LEVEL_REGRET par_scan=PAR_LEVEL_SCAN

    p1 = run_minimax_regret_grid_summaries_v3_sorted()
    p2 = run_maximinmaximax_grid_summaries_v5_sorted()

    @info "master runner finished" regret=p1 maximinmaximax=p2
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
