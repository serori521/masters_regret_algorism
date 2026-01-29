# scripts/scale_bias_logger_minimax_regret.jl

# ------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using Printf

# -------------------------
# Config (EDIT HERE)
# -------------------------

# fixed for your experiment design
const M = 5
const REPEAT_NUM = 1000
const UTILITY_MATRIX_NUM = 100

# target conditions (you can narrow them for debugging)
const TARGET_NS        = [6]                   # e.g., [6] or [4,5,6,7,8]
const TARGET_UTILITIES = ["u1", "u2"]          # ["u1"] etc.
const TARGET_TW        = ["A"]                 # ["A","B","C","D","E"]

# methods to analyze (few is OK; set freely)
const TARGET_METHODS   = ["eAMRw"] # e.g., ["eAMRw"] only

# sampling: reduce for quick run (IMPORTANT for log size)
const TARGET_UTILITY_IDXS = 1:UTILITY_MATRIX_NUM   # e.g., 1:10
const TARGET_REPEAT_IDXS  = 1:REPEAT_NUM           # e.g., 1:50

# eps for LPS
const EPS_REGRET = SetRegretCore.EPS_DEFAULT

# output dir
const OUT_SUBDIR = "scale_bias_logs"

# -------------------------
# Helpers (ported from your runners)
# -------------------------

@inline max_pairs(Alt::Int) = Alt * (Alt - 1) ÷ 2
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m

# concordant pair count between two rankings (0..C(M,2))
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

# find interval index i such that ts[i] >= t >= ts[i+1] (ts is descending)
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

# LPS result -> (ts, ranks) sorted by t descending
function points_from_res(res)
    ts = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    p = sortperm(ts; rev=true)
    return ts[p], ranks[p]
end

@inline function alpha01(t::Float64, tU::Float64, tL::Float64)
    # note: typically tL < tU, so denom is negative -> alpha in [0,1]
    den = (tL - tU)
    den == 0.0 && return 0.0
    return (t - tU) / den
end

@inline function interval_mid_vec(wL::Vector{Float64}, wU::Vector{Float64})
    return 0.5 .* (wL .+ wU)
end

@inline function vec_to_str(v::AbstractVector{<:Real}; digits::Int=10, sep::String=";")
    return join((@sprintf("%.*f", digits, float(x)) for x in v), sep)
end

@inline function rank_to_str(r::Vector{Int})
    return join(r, ';')
end

"""
best-match for a given target rank:
  - maximize concordant pair count
  - tie-break: choose interval whose midpoint is closest to prefer_t
Returns (best_idx, best_score)
"""
function best_match_idx(target_rank::Vector{Int},
                        cand_ts::Vector{Float64},
                        cand_ranks::Vector{Vector{Int}},
                        prefer_t::Float64)
    I = max(length(cand_ranks) - 1, 1)
    best = -1
    best_i = 1
    best_dist = Inf
    for i in 1:I
        c = count_concordant_pairs(cand_ranks[i], target_rank)
        if c > best
            best = c
            best_i = i
            tmid = 0.5 * (cand_ts[i] + cand_ts[i+1])
            best_dist = abs(tmid - prefer_t)
        elseif c == best
            tmid = 0.5 * (cand_ts[i] + cand_ts[i+1])
            dist = abs(tmid - prefer_t)
            if dist < best_dist
                best_i = i
                best_dist = dist
            end
        end
    end
    return best_i, best
end

# -------------------------
# Core logging per setting
# -------------------------

function ensure_outdir(paths)
    outdir = joinpath(paths.data, "metrics_julia", OUT_SUBDIR)
    mkpath(outdir)
    return outdir
end

function open_with_header(path::String, header::Vector{String})
    io = open(path, "w")
    println(io, join(header, ','))
    return io
end

function log_true_to_pred!(io, meta, trueW_mid, tU_true, tL_true, true_ts, true_ranks,
                          wL, wU, tU_pred, tL_pred, m_ts, m_ranks,
                          utl_num::Int, r::Int)
    denom_pairs = max_pairs(M)
    J = max(length(true_ranks) - 1, 1)

    # mid of estimated interval weights
    est_mid = interval_mid_vec(wL, wU)

    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])
        α_true = alpha01(t_true_mid, tU_true, tL_true)

        # diagonal mapping: choose prefer_t on predicted side
        prefer_t = tU_pred + α_true * (tL_pred - tU_pred)
        prefer_t = clamp(prefer_t, tL_pred, tU_pred)

        true_rank = true_ranks[j]

        best_i, best_score = best_match_idx(true_rank, m_ts, m_ranks, prefer_t)

        t_pred_mid = 0.5 * (m_ts[best_i] + m_ts[best_i+1])
        α_pred = alpha01(t_pred_mid, tU_pred, tL_pred)

        # scaled mid vectors
        y_true = trueW_mid .* t_true_mid
        sum_true = sum(y_true)
        y_true_norm = y_true ./ sum_true

        y_pred = est_mid .* t_pred_mid
        sum_pred = sum(y_pred)
        y_pred_norm = y_pred ./ sum_pred

        # diffs (normalized) & scale ratio (raw sums)
        l1 = sum(abs.(y_pred_norm .- y_true_norm))
        l2 = sqrt(sum((y_pred_norm .- y_true_norm).^2))
        scale_ratio = sum_true == 0.0 ? NaN : (sum_pred / sum_true)

        # write row
        println(io, join([
            meta.rule, string(meta.N), meta.tw, meta.utility, meta.method,
            string(utl_num), string(r),
            string(j),
            @sprintf("%.10f", t_true_mid),
            @sprintf("%.10f", α_true),
            rank_to_str(true_rank),
            string(best_i),
            @sprintf("%.10f", t_pred_mid),
            @sprintf("%.10f", α_pred),
            rank_to_str(m_ranks[best_i]),
            string(denom_pairs),
            string(best_score),
            @sprintf("%.10f", best_score / denom_pairs),
            @sprintf("%.10f", sum_true),
            @sprintf("%.10f", sum_pred),
            @sprintf("%.10f", scale_ratio),
            @sprintf("%.10f", l1),
            @sprintf("%.10f", l2),
            vec_to_str(wL),
            vec_to_str(wU),
            vec_to_str(y_pred),
            vec_to_str(y_pred_norm),
            vec_to_str(y_true),
            vec_to_str(y_true_norm)
        ], ','))
    end
end

function log_pred_to_true!(io, meta, trueW_mid, tU_true, tL_true, true_ts, true_ranks,
                          wL, wU, tU_pred, tL_pred, m_ts, m_ranks,
                          utl_num::Int, r::Int)
    denom_pairs = max_pairs(M)
    I = max(length(m_ranks) - 1, 1)

    est_mid = interval_mid_vec(wL, wU)

    for i in 1:I
        t_pred_mid = 0.5 * (m_ts[i] + m_ts[i+1])
        α_pred = alpha01(t_pred_mid, tU_pred, tL_pred)

        # map back to true side for tie-break (diagonal reverse mapping)
        prefer_t_true = tU_true + α_pred * (tL_true - tU_true)
        prefer_t_true = clamp(prefer_t_true, tL_true, tU_true)

        pred_rank = m_ranks[i]

        best_j, best_score = best_match_idx(pred_rank, true_ts, true_ranks, prefer_t_true)

        t_true_mid = 0.5 * (true_ts[best_j] + true_ts[best_j+1])
        α_true = alpha01(t_true_mid, tU_true, tL_true)

        # scaled vectors
        y_true = trueW_mid .* t_true_mid
        sum_true = sum(y_true)
        y_true_norm = y_true ./ sum_true

        y_pred = est_mid .* t_pred_mid
        sum_pred = sum(y_pred)
        y_pred_norm = y_pred ./ sum_pred

        l1 = sum(abs.(y_pred_norm .- y_true_norm))
        l2 = sqrt(sum((y_pred_norm .- y_true_norm).^2))
        scale_ratio = sum_true == 0.0 ? NaN : (sum_pred / sum_true)
        edge_dist = min(α_pred, 1.0 - α_pred)

        println(io, join([
            meta.rule, string(meta.N), meta.tw, meta.utility, meta.method,
            string(utl_num), string(r),
            string(i),
            @sprintf("%.10f", t_pred_mid),
            @sprintf("%.10f", α_pred),
            @sprintf("%.10f", edge_dist),
            rank_to_str(pred_rank),
            string(best_j),
            @sprintf("%.10f", t_true_mid),
            @sprintf("%.10f", α_true),
            rank_to_str(true_ranks[best_j]),
            string(denom_pairs),
            string(best_score),
            @sprintf("%.10f", best_score / denom_pairs),
            @sprintf("%.10f", sum_true),
            @sprintf("%.10f", sum_pred),
            @sprintf("%.10f", scale_ratio),
            @sprintf("%.10f", l1),
            @sprintf("%.10f", l2),
            vec_to_str(wL),
            vec_to_str(wU)
        ], ','))
    end
end

function log_case_summary!(io, meta, tU_true, tL_true, true_ts, true_ranks,
                           wL, wU, tU_pred, tL_pred, m_ts, m_ranks,
                           utl_num::Int, r::Int)
    denom_pairs = max_pairs(M)
    J = max(length(true_ranks) - 1, 1)

    # case-level: average best score across true intervals, and mean alpha_pred among those best matches
    score_sum = 0.0
    alpha_pred_sum = 0.0
    perfect_cnt = 0

    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])
        α_true = alpha01(t_true_mid, tU_true, tL_true)
        prefer_t = tU_pred + α_true * (tL_pred - tU_pred)
        prefer_t = clamp(prefer_t, tL_pred, tU_pred)
        best_i, best_score = best_match_idx(true_ranks[j], m_ts, m_ranks, prefer_t)

        t_pred_mid = 0.5 * (m_ts[best_i] + m_ts[best_i+1])
        α_pred = alpha01(t_pred_mid, tU_pred, tL_pred)

        score_sum += best_score
        alpha_pred_sum += α_pred
        perfect_cnt += (best_score == denom_pairs) ? 1 : 0
    end

    mean_best_score = score_sum / J
    mean_best_score01 = mean_best_score / denom_pairs
    mean_alpha_pred = alpha_pred_sum / J
    perfect_rate = perfect_cnt / J

    println(io, join([
        meta.rule, string(meta.N), meta.tw, meta.utility, meta.method,
        string(utl_num), string(r),
        string(J),
        @sprintf("%.10f", mean_best_score),
        @sprintf("%.10f", mean_best_score01),
        @sprintf("%.10f", mean_alpha_pred),
        @sprintf("%.10f", perfect_rate),
        @sprintf("%.10f", tU_pred),
        @sprintf("%.10f", tL_pred)
    ], ','))
end

function run_one_setting(paths, utility::String, N::Int, tw::String, method::String)
    # load data
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)
    trueW_mid = interval_mid_vec(trueW.L, trueW.R)

    # load method weights
    filename = joinpath(tw, method)
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch err
        @warn "failed to read method weights" utility N tw method err
        return nothing
    end
    repeat_avail = min(REPEAT_NUM, length(methodW))
    if repeat_avail == 0
        @warn "no repeats available" utility N tw method
        return nothing
    end

    outdir = ensure_outdir(paths)
    tag = "rule=minimax_regret__N=$(N)__tw=$(tw)__utility=$(utility)__method=$(method)"

    f_true_to_pred = joinpath(outdir, "true_to_pred__$(tag).csv")
    f_pred_to_true = joinpath(outdir, "pred_to_true__$(tag).csv")
    f_case_summary = joinpath(outdir, "case_summary__$(tag).csv")

    # headers
    h1 = [
        "rule","N","tw","utility","method",
        "utl_num","repeat_idx",
        "true_interval_idx",
        "t_true_mid","alpha_true",
        "true_rank",
        "best_pred_interval_idx",
        "t_pred_mid","alpha_pred",
        "pred_rank",
        "denom_pairs",
        "best_score_pairs","best_score_01",
        "sum_true_mid_scaled","sum_pred_mid_scaled","sum_scale_ratio",
        "l1_normdiff","l2_normdiff",
        "wL_pred","wU_pred","y_pred_mid_scaled","y_pred_norm",
        "y_true_mid_scaled","y_true_norm"
    ]
    h2 = [
        "rule","N","tw","utility","method",
        "utl_num","repeat_idx",
        "pred_interval_idx",
        "t_pred_mid","alpha_pred","edge_dist",
        "pred_rank",
        "best_true_interval_idx",
        "t_true_mid","alpha_true",
        "true_rank",
        "denom_pairs",
        "best_score_pairs","best_score_01",
        "sum_true_mid_scaled","sum_pred_mid_scaled","sum_scale_ratio",
        "l1_normdiff","l2_normdiff",
        "wL_pred","wU_pred"
    ]
    h3 = [
        "rule","N","tw","utility","method",
        "utl_num","repeat_idx",
        "J_true_intervals",
        "mean_best_score_pairs","mean_best_score_01",
        "mean_alpha_pred",
        "perfect_match_rate",
        "tU_pred","tL_pred"
    ]

    io1 = open_with_header(f_true_to_pred, h1)
    io2 = open_with_header(f_pred_to_true, h2)
    io3 = open_with_header(f_case_summary, h3)

    meta = (rule="minimax_regret", utility=utility, N=N, tw=tw, method=method)

    # Precompute true timeline per utl_num (once), as in your winloss runner
    true_cache = Dict{Int, Tuple{Vector{Float64}, Vector{Vector{Int}}}}()

    utl_idxs = collect(TARGET_UTILITY_IDXS)
    rep_idxs = collect(TARGET_REPEAT_IDXS)

    for utl_num in utl_idxs
        U = Matrix(utility_mats[utl_num])
        # true timeline for this utility matrix
        if !haskey(true_cache, utl_num)
            matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
            res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=EPS_REGRET)
            true_ts, true_ranks = points_from_res(res_true)
            true_cache[utl_num] = (true_ts, true_ranks)
        end
        true_ts, true_ranks = true_cache[utl_num]

        # loop repeats
        for r in rep_idxs
            r > repeat_avail && continue

            wL = methodW[r].L
            wU = methodW[r].R
            tL_pred, tU_pred = SetRegretCore.find_optimal_trange(wL, wU)

            matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
            res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL_pred, tU_pred; eps=EPS_REGRET)
            m_ts, m_ranks = points_from_res(res_m)

            # logs
            log_true_to_pred!(io1, meta, trueW_mid, tU_true, tL_true, true_ts, true_ranks,
                              wL, wU, tU_pred, tL_pred, m_ts, m_ranks, utl_num, r)

            log_pred_to_true!(io2, meta, trueW_mid, tU_true, tL_true, true_ts, true_ranks,
                              wL, wU, tU_pred, tL_pred, m_ts, m_ranks, utl_num, r)

            log_case_summary!(io3, meta, tU_true, tL_true, true_ts, true_ranks,
                              wL, wU, tU_pred, tL_pred, m_ts, m_ranks, utl_num, r)
        end
        @info "done utility matrix" utility N tw method utl_num
    end

    close(io1); close(io2); close(io3)

    @info "saved logs" f_true_to_pred f_pred_to_true f_case_summary
    return (f_true_to_pred, f_pred_to_true, f_case_summary)
end

function main()
    paths = Paths.project_paths()

    # sanity print
    @info "threads" Threads.nthreads()
    @info "targets" TARGET_NS TARGET_UTILITIES TARGET_TW TARGET_METHODS
    @info "sampling" TARGET_UTILITY_IDXS TARGET_REPEAT_IDXS

    for utility in TARGET_UTILITIES, N in TARGET_NS, tw in TARGET_TW, method in TARGET_METHODS
        run_one_setting(paths, utility, N, tw, method)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
