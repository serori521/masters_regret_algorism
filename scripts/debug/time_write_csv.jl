include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore

using CSV
using DataFrames
using Dates
using Statistics

# -------------------------
# 設定
# -------------------------
const BRUTE_STEPS     = 5000 #5000で50,10000で19の取りこぼし在り（10000回の実行のうち）
const repeat_num      = 100
const counts_utility  = 100

# tol = tol_factor * Δ
const TOL_FACTOR = 20

# ★ここが重要：epsは小さく（あなたの結論）
const EPS_BENCH = 1e-15

# 出力CSV（追記）
function bench_csv_path()
    return joinpath(@__DIR__, "..", "..", "results", "tmp", "bench_time_brute_vs_lps.csv")
end
function rank_at_t(utility, wL, wU, t; eps=EPS_BENCH)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, t; eps=eps)
    A = size(matrix,1)
    qstar = [SetRegretCore.argmax_regret_index(matrix, p, t; eps=eps) for p in 1:A]
    return SetRegretCore.snapshot_state(matrix, qstar, t; eps=eps).rank
end

function confirm_change(utility, wL, wU, x, Δ; eps=EPS_BENCH)
    δ = min(Δ/10, 1e-8)                 # 近傍を見る（適当でOK）
    r1 = rank_at_t(utility, wL, wU, x + δ; eps=eps)
    r2 = rank_at_t(utility, wL, wU, x - δ; eps=eps)
    return r1 != r2
end

# -------------------------
# 計算用関数
# -------------------------
function brute_scan_changes(utility, wL, wU, tL, tR; steps=BRUTE_STEPS, eps=EPS_BENCH)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, tR; eps=eps)
    Δ = (tR - tL) / steps
    ts = collect(tR:-Δ:tL)

    A = size(matrix, 1)
    changes = Float64[]

    # 初期状態
    t0 = ts[1]
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, t0; eps=eps)
    qstar0 = [SetRegretCore.argmax_regret_index(matrix, p, t0; eps=eps) for p in 1:A]
    prev_rank = SetRegretCore.snapshot_state(matrix, qstar0, t0; eps=eps).rank

    for k in 2:length(ts)
        t = ts[k]
        # tごとにモデル更新 (Brute forceの正確性担保)
        @inbounds for p in 1:A, q in 1:A
            p == q && continue
            SetRegretCore.set_linear_model_for_pair!(matrix[p, q], wL, wU, t; eps=eps)
        end
        qstar = [SetRegretCore.argmax_regret_index(matrix, p, t; eps=eps) for p in 1:A]
        snap = SetRegretCore.snapshot_state(matrix, qstar, t; eps=eps)
        if snap.rank != prev_rank
            push!(changes, t)
            prev_rank = snap.rank
        end
    end
    return changes
end

function lps_run_changes(utility, wL, wU, tL, tR; eps=EPS_BENCH)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    res = SetRegretCore.run_lps(matrix, wL, wU, tL, tR; eps=eps)
    return Vector{Float64}(res.changes)
end

function match_changes(brute_ts::Vector{Float64}, lps_ts::Vector{Float64}, tol::Float64)
    b = sort(copy(brute_ts); rev=true)
    l = sort(copy(lps_ts); rev=true)
    pairs = Tuple{Float64,Float64}[]
    miss_b = Float64[] # Missing in LPS
    miss_l = Float64[] # Extra in LPS

    i = 1; j = 1
    while i <= length(b) && j <= length(l)
        tb = b[i]; tl = l[j]
        if abs(tb - tl) <= tol
            push!(pairs, (tb, tl)); i += 1; j += 1
        elseif tb > tl + tol
            push!(miss_b, tb); i += 1
        else
            push!(miss_l, tl); j += 1
        end
    end
    while i <= length(b); push!(miss_b, b[i]); i += 1; end
    while j <= length(l); push!(miss_l, l[j]); j += 1; end
    return pairs, miss_b, miss_l
end

# -------------------------
# 追記CSVユーティリティ
# -------------------------
function append_row!(csvfile::String, row::DataFrame)
    isdir(dirname(csvfile)) || mkpath(dirname(csvfile))
    write_header = !isfile(csvfile)
    CSV.write(csvfile, row; append=true, writeheader=write_header)
end

# -------------------------
# メイン
# -------------------------
function main()
    paths = Paths.project_paths()

    utility_v = LoadInstance.read_utility_value(paths, "u1")
    methodW   = LoadInstance.read_method_weights(paths, "A/MMRW", repeat_num)
    eps = EPS_BENCH
    csvfile = bench_csv_path()
    println("Bench CSV: $csvfile")
    println("Start benchmark... repeat_num=$repeat_num, counts_utility=$counts_utility, BRUTE_STEPS=$BRUTE_STEPS, eps=$(EPS_BENCH)")
    utility = Matrix(utility_v[1])
    wL = methodW[1].L
    wU = methodW[1].R
    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)
    _ = brute_scan_changes(utility, wL, wU, tL, tR; steps=BRUTE_STEPS, eps=eps)
    _ = lps_run_changes(utility, wL, wU, tL, tR; eps=eps)
    # 全体集計（任意）
    brute_total_all = 0.0
    lps_total_all   = 0.0
    missing_all     = 0
    extra_all       = 0
    true_extra = 0
    for i in 1:repeat_num
        wL = methodW[i].L
        wU = methodW[i].R
        tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

        Δ   = (tR - tL) / BRUTE_STEPS
        tol = TOL_FACTOR * Δ


        brute_times = Float64[]  # 秒
        lps_times   = Float64[]  # 秒
        missing_cnt = 0
        extra_cnt   = 0

        # progress
        print("repeat $i / $repeat_num ... ")


        
        for idx in 1:counts_utility
            utility = Matrix(utility_v[idx])

            # brute time
            t0 = time_ns()
            brute_changes = brute_scan_changes(utility, wL, wU, tL, tR; steps=BRUTE_STEPS, eps=eps)
            push!(brute_times, (time_ns() - t0) / 1e9)

            # lps time
            t1 = time_ns()
            lps_changes = lps_run_changes(utility, wL, wU, tL, tR; eps=eps)
            push!(lps_times, (time_ns() - t1) / 1e9)


            # correctness check
            _, miss_b, miss_l = match_changes(brute_changes, lps_changes, tol)
            if !isempty(miss_l)
                # miss_l は "Extra in LPS"
                for x in miss_l
                    true_extra += confirm_change(utility, wL, wU, x, Δ; eps=eps) ? 1 : 0
                end
                
                # CSVに true_extra を書く、または println する
            end
            missing_cnt += length(miss_b)
            extra_cnt   += length(miss_l)
        end

        brute_total = sum(brute_times)
        lps_total   = sum(lps_times)

        brute_avg_ms = mean(brute_times) * 1000
        lps_avg_ms   = mean(lps_times) * 1000

        brute_total_all += brute_total
        lps_total_all   += lps_total
        missing_all     += missing_cnt
        extra_all       += extra_cnt

        dt = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

        row = DataFrame(
            datetime = [dt],
            repeat_index = [i],
            utility_count = [counts_utility],
            brute_steps = [BRUTE_STEPS],
            tol_factor = [TOL_FACTOR],
            delta = [Δ],
            tol = [tol],
            eps = [eps],
            brute_total_sec = [brute_total],
            brute_avg_ms = [brute_avg_ms],
            lps_total_sec = [lps_total],
            lps_avg_ms = [lps_avg_ms],
            missing_total = [missing_cnt],
            extra_total = [extra_cnt],
        )

        append_row!(csvfile, row)
        println("done. brute_total=$(round(brute_total; digits=3))s, lps_total=$(round(lps_total; digits=3))s, missing=$missing_cnt, extra=$extra_cnt")
    end

    # 全体まとめも追記（欲しければ）
    dt = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
    row_all = DataFrame(
        datetime = [dt],
        repeat_index = [0],  # 0 を "ALL" 扱い
        utility_count = [repeat_num * counts_utility],
        brute_steps = [BRUTE_STEPS],
        tol_factor = [TOL_FACTOR],
        delta = [NaN],
        tol = [NaN],
        eps = [EPS_BENCH],
        brute_total_sec = [brute_total_all],
        brute_avg_ms = [(brute_total_all / (repeat_num * counts_utility)) * 1000],
        lps_total_sec = [lps_total_all],
        lps_avg_ms = [(lps_total_all / (repeat_num * counts_utility)) * 1000],
        missing_total = [missing_all],
        extra_total = [extra_all],
    )
    append_row!(csvfile, row_all)
    println(true_extra)
    println("ALL done. brute_total=$(round(brute_total_all; digits=3))s, lps_total=$(round(lps_total_all; digits=3))s, missing=$missing_all, extra=$extra_all")
end

main()
