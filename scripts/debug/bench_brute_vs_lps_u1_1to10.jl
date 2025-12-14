# scripts/debug/bench_brute_vs_lps_u1_1to10_reps_pretty2.jl
#
# 変更点：
# - utility ごとの詳細CSV（brute/lps/pairs/missing）は
#   missing が「ある場合のみ」保存
# - かつ results/tmp/fail_cases/utility_u1_X/ にまとめて保存
#
# 主要出力（従来どおり）：
#   results/tmp/
#     bench_brute_vs_lps_key.csv
#     bench_brute_vs_lps_full.csv
#     bench_brute_vs_lps_failures.csv

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore

using CSV
using DataFrames

# -------------------------
# parameters
# -------------------------
const N_REPS      = 100
const BRUTE_STEPS = 5000
const MAX_LIST_TS = 50

# -------------------------
# utilities
# -------------------------
rank_str(rank::Vector{Int}) = join(rank, '|')

function ts_to_str(ts::Vector{Float64}; maxn::Int=MAX_LIST_TS)
    isempty(ts) && return ""
    s = sort(copy(ts); rev=true)
    if length(s) <= maxn
        return join(string.(s), '|')
    else
        return join(string.(s[1:maxn]), '|') * "|...(" * string(length(s)-maxn) * " more)"
    end
end

function match_changes(brute_ts::Vector{Float64}, lps_ts::Vector{Float64}, tol::Float64)
    b = sort(copy(brute_ts); rev=true)
    l = sort(copy(lps_ts); rev=true)
    pairs = Tuple{Float64,Float64}[]
    miss_b = Float64[]
    miss_l = Float64[]

    i = 1; j = 1
    while i <= length(b) && j <= length(l)
        tb = b[i]; tl = l[j]
        if abs(tb - tl) <= tol
            push!(pairs, (tb, tl))
            i += 1; j += 1
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
# brute / LPS (簡略：中身は前版と同じ)
# -------------------------
function brute_scan_changes(utility, wL, wU, tL, tR; steps=BRUTE_STEPS, eps=SetRegretCore.EPS_DEFAULT)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, tR; eps=eps)
    Δ = (tR - tL) / steps
    ts = collect(tR:-Δ:tL)

    A = size(matrix, 1)
    changes = Float64[]
    rank_map = Dict{Float64,Vector{Int}}()

    t0 = ts[1]
    @inbounds for p in 1:A, q in 1:A
        p == q && continue
        SetRegretCore.set_linear_model_for_pair!(matrix[p, q], wL, wU, t0; eps=eps)
    end
    qstar0 = [SetRegretCore.argmax_regret_index(matrix, p, t0; eps=eps) for p in 1:A]
    prev_rank = SetRegretCore.snapshot_state(matrix, qstar0, t0; eps=eps).rank

    for k in 2:length(ts)
        t = ts[k]
        @inbounds for p in 1:A, q in 1:A
            p == q && continue
            SetRegretCore.set_linear_model_for_pair!(matrix[p, q], wL, wU, t; eps=eps)
        end
        qstar = [SetRegretCore.argmax_regret_index(matrix, p, t; eps=eps) for p in 1:A]
        snap  = SetRegretCore.snapshot_state(matrix, qstar, t; eps=eps)
        if snap.rank != prev_rank
            push!(changes, t)
            rank_map[t] = snap.rank
            prev_rank = snap.rank
        end
    end
    return changes, rank_map
end

function lps_run_changes(utility, wL, wU, tL, tR; eps=SetRegretCore.EPS_DEFAULT)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    res = SetRegretCore.run_lps(matrix, wL, wU, tL, tR; eps=eps)
    ts = Vector{Float64}(res.changes)
    rank_map = Dict{Float64,Vector{Int}}(snap.t => snap.rank for snap in res.timeline)
    return ts, rank_map
end

# -------------------------
# main
# -------------------------
function main()
    paths = Paths.project_paths()
    outdir = joinpath(@__DIR__, "..", "..", "results", "tmp")
    faildir = joinpath(outdir, "fail_cases")
    mkpath(faildir)

    utility_v = LoadInstance.read_utility_value(paths, "u1")
    methodW   = LoadInstance.read_method_weights(paths, "A/MMRW", 1, 6)
    wL = methodW[1].L; wU = methodW[1].R
    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

    Δ   = (tR - tL) / BRUTE_STEPS
    tol = 20Δ
    eps = SetRegretCore.EPS_DEFAULT

    key = DataFrame(
        utility_idx=Int[], ok=Bool[],
        missing_in_LPS=Int[], extra_in_LPS=Int[],
        missing_in_LPS_ts=String[], extra_in_LPS_ts=String[]
    )

    for idx in 1:N_REPS
        utility = Matrix(utility_v[idx])
        brute_ts, brute_rank = brute_scan_changes(utility, wL, wU, tL, tR; eps=eps)
        lps_ts,   lps_rank   = lps_run_changes(utility, wL, wU, tL, tR; eps=eps)

        pairs, miss_b, miss_l = match_changes(brute_ts, lps_ts, tol)
        ok = isempty(miss_b) && isempty(miss_l)

        push!(key, (
            idx, ok,
            length(miss_b), length(miss_l),
            ts_to_str(miss_b), ts_to_str(miss_l)
        ))

        # ---- missing がある場合のみ詳細保存 ----
        if !ok
            udir = joinpath(faildir, "utility_u1_$(idx)")
            mkpath(udir)

            CSV.write(joinpath(udir, "brute_changes.csv"),
                DataFrame(t=brute_ts))
            CSV.write(joinpath(udir, "lps_changes.csv"),
                DataFrame(t=lps_ts))
            CSV.write(joinpath(udir, "missing.csv"),
                DataFrame(side = ["missing_in_LPS" for _ in miss_b],
                          t = miss_b))
            CSV.write(joinpath(udir, "extra.csv"),
                DataFrame(side = ["extra_in_LPS" for _ in miss_l],
                          t = miss_l))
        end
    end

    CSV.write(joinpath(outdir, "bench_brute_vs_lps_key.csv"), key)
    println("Saved key summary to results/tmp/bench_brute_vs_lps_key.csv")
end

main()
