# scripts/unified_run_all_sorted_weighted.jl
#
# Weighted version: interval lengths weighted by Δlog(s)
#
# Run:
#   julia --project=. scripts/unified_run_all_sorted_weighted.jl
#
# Outputs weighted versions of:
#   - wPrecision, wRecall, wF1 (best match)
#   - wPrecision_worst, wRecall_worst, wF1_worst
#   - wTop1_rate
#   - wFull_mean (全セルを区間長さの積で重み付け)

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

const EPS_REGRET = SetRegretCore.EPS_DEFAULT
const EPS_SCAN   = 1e-6

const WRITE_PCM_LOG = true

# -------------------------
# Helpers
# -------------------------
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m
@inline max_pairs(Alt::Int) = Alt * (Alt - 1) ÷ 2

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

# -------------------------
# 区間の長さ（log比）を計算
# -------------------------
"""
区間端点 ts (降順) から各区間の長さ log(ts[i]/ts[i+1]) を返す
区間数が0の場合や不正な値の場合はエラー情報を出力して終了
"""
function interval_log_lengths(ts::Vector{Float64})
    n_intervals = length(ts) - 1
    if n_intervals <= 0
        # 区間が存在しない場合は空配列
        return Float64[]
    end
    lengths = zeros(Float64, n_intervals)
    for i in 1:n_intervals
        ratio = ts[i] / ts[i+1]
        # 降順のはずなので ts[i] > ts[i+1] > 0 であるべき
        if ratio <= 0 || !isfinite(ratio) || ratio < 1.0
            # 異常値を検出したらエラー情報を返す（呼び出し側で処理）
            return Float64[]  # エラーマーカー
        end
        lengths[i] = log(ratio)
    end
    return lengths
end

# エラー詳細情報を出力して終了
function report_ts_error(context::String, ts::Vector{Float64})
    println("\n" * "="^80)
    println("ERROR: Invalid ts values detected")
    println("="^80)
    println("Context: ", context)
    println("\nts array (length=$(length(ts))):")
    for (idx, t) in enumerate(ts)
        println("  ts[$idx] = $t")
    end
    println("\nRatios ts[i]/ts[i+1]:")
    for i in 1:(length(ts)-1)
        ratio = ts[i] / ts[i+1]
        println("  ts[$i]/ts[$(i+1)] = $ratio (ts[$i]=$(ts[i]), ts[$(i+1)]=$(ts[i+1]))")
        if ratio <= 0 || !isfinite(ratio) || ratio < 1.0
            println("    ^^^ ERROR: Invalid ratio!")
        end
    end
    println("="^80)
    error("Stopping execution due to invalid ts values")
end

"""
加重平均版の case_metrics

返り値:
  wPrecision_best, wRecall_best, wF1_best,
  wPrecision_worst, wRecall_worst, wF1_worst,
  wTop1_rate,
  wFull_mean,
  cases (常に1)
"""
function weighted_case_metrics(true_ts::Vector{Float64}, true_ranks::Vector{Vector{Int}},
                                m_ts::Vector{Float64}, m_ranks::Vector{Vector{Int}})
    # 各タイムラインの区間長さ（log比）
    true_lengths = interval_log_lengths(true_ts)
    m_lengths = interval_log_lengths(m_ts)

    # 区間数が0の場合は評価不能（すべて0を返す）
    if isempty(true_lengths) || isempty(m_lengths)
        return 0.0, 0.0, 0.0,  # best
               0.0, 0.0, 0.0,  # worst
               0.0,             # top1
               0.0,             # full_mean
               0               # skip this case
    end

    # 区間数（注意：ranks は区間+1個の要素を持つ）
    J = length(true_lengths)
    I = length(m_lengths)

    total_true_log = sum(true_lengths)
    total_m_log = sum(m_lengths)

    # Precision（予測区間を重み付け）
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
        prec_best_sum += m_lengths[i] * best
        prec_worst_sum += m_lengths[i] * worst
    end
    wPrecision_best = prec_best_sum / total_m_log
    wPrecision_worst = prec_worst_sum / total_m_log

    # Recall（真区間を重み付け）
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
        rec_best_sum += true_lengths[j] * best
        rec_worst_sum += true_lengths[j] * worst
    end
    wRecall_best = rec_best_sum / total_true_log
    wRecall_worst = rec_worst_sum / total_true_log

    # F1（調和平均）
    wF1_best = (wPrecision_best + wRecall_best) > 0 ? 
               (2 * wPrecision_best * wRecall_best / (wPrecision_best + wRecall_best)) : 0.0
    wF1_worst = (wPrecision_worst + wRecall_worst) > 0 ? 
                (2 * wPrecision_worst * wRecall_worst / (wPrecision_worst + wRecall_worst)) : 0.0

    # Top1 一致率（区間の積で重み付け）
    top1_weighted_sum = 0.0
    total_weight = 0.0
    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            rj = true_ranks[j]
            weight = m_lengths[i] * true_lengths[j]
            top1_weighted_sum += weight * (top1_ok(ri, rj) ? 1.0 : 0.0)
            total_weight += weight
        end
    end
    wTop1_rate = top1_weighted_sum / total_weight

    # Full mean（全セルを区間長さの積で重み付け）
    # 注：full_meanは全セル(i,j)を使うので、bestとworstの区別はない
    full_weighted_sum = 0.0
    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            rj = true_ranks[j]
            weight = m_lengths[i] * true_lengths[j]
            full_weighted_sum += weight * count_concordant_pairs(ri, rj)
        end
    end
    wFull_mean = full_weighted_sum / total_weight

    return wPrecision_best, wRecall_best, wF1_best,
           wPrecision_worst, wRecall_worst, wF1_worst,
           wTop1_rate,
           wFull_mean,
           1  # cases
end

# -------------------------
# Regret (LPS)
# -------------------------
function points_from_res(res)
    ts = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    p = sortperm(ts; rev=true)
    return ts[p], ranks[p]
end

function summarize_one_regret_weighted(paths, utility::String, N::Int, tw::String, method::String;
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

    # Accumulators
    acc = zeros(Float64, 8)  # wP, wR, wF1, wPw, wRw, wF1w, wTop1, wFull

    # Per-PCM optional
    pcm_acc = pcm_log_dir === nothing ? nothing : zeros(Float64, 8, repeat)

    cases = 0

    for utl_num in 1:UTILITY_MATRIX_NUM
        U = Matrix(utility_mats[utl_num])

        matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
        res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
        true_ts, true_ranks = points_from_res(res_true)

        for r in 1:repeat
            wL, wU = methodW[r].L, methodW[r].R
            tL, tU = SetRegretCore.find_optimal_trange(wL, wU)

            matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
            res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
            m_ts, m_ranks = points_from_res(res_m)

            # エラーチェック：tsが不正な場合は詳細を出力して終了
            true_lengths_check = interval_log_lengths(true_ts)
            if isempty(true_lengths_check) && length(true_ts) > 1
                context = "minimax_regret | N=$N | tw=$tw | utility=$utility | method=$(method_clean(method)) | utl_num=$utl_num | repeat=$r | TRUE timeline"
                report_ts_error(context, true_ts)
            end
            m_lengths_check = interval_log_lengths(m_ts)
            if isempty(m_lengths_check) && length(m_ts) > 1
                context = "minimax_regret | N=$N | tw=$tw | utility=$utility | method=$(method_clean(method)) | utl_num=$utl_num | repeat=$r | PREDICTED timeline"
                report_ts_error(context, m_ts)
            end

            wP, wR, wF1, wPw, wRw, wF1w, wTop1, wFull, valid = 
                weighted_case_metrics(true_ts, true_ranks, m_ts, m_ranks)

            if valid > 0
                acc[1] += wP
                acc[2] += wR
                acc[3] += wF1
                acc[4] += wPw
                acc[5] += wRw
                acc[6] += wF1w
                acc[7] += wTop1
                acc[8] += wFull

                if pcm_acc !== nothing
                    pcm_acc[1, r] += wP
                    pcm_acc[2, r] += wR
                    pcm_acc[3, r] += wF1
                    pcm_acc[4, r] += wPw
                    pcm_acc[5, r] += wRw
                    pcm_acc[6, r] += wF1w
                    pcm_acc[7, r] += wTop1
                    pcm_acc[8, r] += wFull
                end

                cases += 1
            end
        end
    end

    cases == 0 && return nothing

    # Write per-PCM logs
    if pcm_acc !== nothing
        mkpath(pcm_log_dir)
        mname = method_clean(method)
        outpath = joinpath(pcm_log_dir,
            "pcm_weighted_minimax_regret__N=$(N)__tw=$(tw)__utility=$(utility)__method=$(mname).csv")

        denom_pairs = max_pairs(M)

        open(outpath, "w") do io
            println(io, join([
                "repeat_idx",
                "wPrecision_best","wRecall_best","wF1_best",
                "wPrecision_worst","wRecall_worst","wF1_worst",
                "wTop1_rate","wFull_mean",
                "denom_pairs"
            ], ','))

            for r in 1:repeat
                wP = pcm_acc[1, r] / UTILITY_MATRIX_NUM
                wR = pcm_acc[2, r] / UTILITY_MATRIX_NUM
                wF1 = pcm_acc[3, r] / UTILITY_MATRIX_NUM
                wPw = pcm_acc[4, r] / UTILITY_MATRIX_NUM
                wRw = pcm_acc[5, r] / UTILITY_MATRIX_NUM
                wF1w = pcm_acc[6, r] / UTILITY_MATRIX_NUM
                wTop1 = pcm_acc[7, r] / UTILITY_MATRIX_NUM
                wFull = pcm_acc[8, r] / UTILITY_MATRIX_NUM

                println(io, join([
                    string(r),
                    @sprintf("%.10f", wP),
                    @sprintf("%.10f", wR),
                    @sprintf("%.10f", wF1),
                    @sprintf("%.10f", wPw),
                    @sprintf("%.10f", wRw),
                    @sprintf("%.10f", wF1w),
                    @sprintf("%.10f", wTop1),
                    @sprintf("%.10f", wFull),
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

        sum_wPrecision = acc[1],
        sum_wRecall = acc[2],
        sum_wF1 = acc[3],

        sum_wPrecision_worst = acc[4],
        sum_wRecall_worst = acc[5],
        sum_wF1_worst = acc[6],

        sum_wTop1 = acc[7],
        sum_wFull_mean = acc[8],

        cases = cases
    )
end

# -------------------------
# Maximin/Maximax (Scan)
# -------------------------
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

function build_perm(U::Matrix{Float64}, rule::Symbol)
    Alt, N = size(U)
    perm = Vector{Vector{Int}}(undef, Alt)
    rev = (rule == :maximax)
    for a in 1:Alt
        perm[a] = sortperm(@view U[a, :]; rev=rev)
    end
    return perm
end

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

function summarize_one_scan_weighted(paths, rule::Symbol, utility::String, N::Int, tw::String, method::String;
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

    acc = zeros(Float64, 8)
    cases = 0

    for utl_num in 1:UTILITY_MATRIX_NUM
        U = Matrix(utility_mats[utl_num])

        true_ts, true_ranks = scan_timeline_maximinmaximax(U, trueW.L, trueW.R, rule; epsi=epsi)

        for r in 1:repeat
            m_ts, m_ranks = scan_timeline_maximinmaximax(U, methodW[r].L, methodW[r].R, rule; epsi=epsi)

            # エラーチェック：tsが不正な場合は詳細を出力して終了
            true_lengths_check = interval_log_lengths(true_ts)
            if isempty(true_lengths_check) && length(true_ts) > 1
                context = "$rule | N=$N | tw=$tw | utility=$utility | method=$(method_clean(method)) | utl_num=$utl_num | repeat=$r | TRUE timeline"
                report_ts_error(context, true_ts)
            end
            m_lengths_check = interval_log_lengths(m_ts)
            if isempty(m_lengths_check) && length(m_ts) > 1
                context = "$rule | N=$N | tw=$tw | utility=$utility | method=$(method_clean(method)) | utl_num=$utl_num | repeat=$r | PREDICTED timeline"
                report_ts_error(context, m_ts)
            end

            wP, wR, wF1, wPw, wRw, wF1w, wTop1, wFull, valid = 
                weighted_case_metrics(true_ts, true_ranks, m_ts, m_ranks)

            if valid > 0
                acc[1] += wP
                acc[2] += wR
                acc[3] += wF1
                acc[4] += wPw
                acc[5] += wRw
                acc[6] += wF1w
                acc[7] += wTop1
                acc[8] += wFull
                cases += 1
            end
        end
    end

    cases == 0 && return nothing

    return (
        rule = String(rule),
        utility = utility,
        N = N,
        tw = tw,
        method = method_clean(method),

        sum_wPrecision = acc[1],
        sum_wRecall = acc[2],
        sum_wF1 = acc[3],

        sum_wPrecision_worst = acc[4],
        sum_wRecall_worst = acc[5],
        sum_wF1_worst = acc[6],

        sum_wTop1 = acc[7],
        sum_wFull_mean = acc[8],

        cases = cases
    )
end

# -------------------------
# PCM log merge
# -------------------------
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
# Master
# -------------------------
function collect_all_results_weighted(pcm_log_dir::Union{Nothing,String})
    paths = Paths.project_paths()

    tasks_regret = [(utility, N, tw, m) for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]
    rules = (:maximin, :maximax)
    tasks_scan = [(rule, utility, N, tw, m) for rule in rules for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]

    total_tasks = length(tasks_regret) + length(tasks_scan)
    all_results = Vector{Any}(undef, total_tasks)

    @info "Starting weighted version processing" total_tasks threads=nthreads()

    @threads for idx in 1:total_tasks
        if idx <= length(tasks_regret)
            (utility, N, tw, method) = tasks_regret[idx]
            res = summarize_one_regret_weighted(paths, utility, N, tw, method; pcm_log_dir=pcm_log_dir)
            all_results[idx] = res
        else
            local_idx = idx - length(tasks_regret)
            (rule, utility, N, tw, method) = tasks_scan[local_idx]
            res = summarize_one_scan_weighted(paths, rule, utility, N, tw, method)
            all_results[idx] = res
        end

        if idx % 100 == 0
            @info "Progress" idx total_tasks
        end
    end

    valid_rows = [r for r in all_results if r !== nothing]

    rule_order = Dict("minimax_regret"=>1, "maximin"=>2, "maximax"=>3)
    sort!(valid_rows, by = r -> (get(rule_order, r.rule, 99), r.N, r.tw, r.utility, r.method))

    return valid_rows
end

function main()
    paths = Paths.project_paths()
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)

    pcm_log_dir = nothing
    if WRITE_PCM_LOG
        pcm_log_dir = joinpath(outdir, "pcm_logs_weighted")
        mkpath(pcm_log_dir)
    end

    rows = collect_all_results_weighted(pcm_log_dir)

    outpath = joinpath(outdir, "grid_summary_ALL_RULES_WEIGHTED.csv")

    open(outpath, "w") do io
        println(io, join([
            "rule","N","tw","utility","method",
            "sum_wPrecision","sum_wRecall","sum_wF1",
            "sum_wPrecision_worst","sum_wRecall_worst","sum_wF1_worst",
            "sum_wTop1","sum_wFull_mean",
            "cases"
        ], ','))

        for r in rows
            println(io, join([
                r.rule,
                string(r.N),
                r.tw,
                r.utility,
                r.method,

                @sprintf("%.10f", r.sum_wPrecision),
                @sprintf("%.10f", r.sum_wRecall),
                @sprintf("%.10f", r.sum_wF1),

                @sprintf("%.10f", r.sum_wPrecision_worst),
                @sprintf("%.10f", r.sum_wRecall_worst),
                @sprintf("%.10f", r.sum_wF1_worst),

                @sprintf("%.10f", r.sum_wTop1),
                @sprintf("%.10f", r.sum_wFull_mean),

                string(r.cases)
            ], ','))
        end
    end

    @info "Saved weighted summary" outpath rows=length(rows)

    if WRITE_PCM_LOG && pcm_log_dir !== nothing
        merged = joinpath(outdir, "pcm_summary_minimax_regret_WEIGHTED.csv")
        ok = merge_pcm_logs(pcm_log_dir, merged)
        @info "Merged weighted PCM logs" merged ok
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end