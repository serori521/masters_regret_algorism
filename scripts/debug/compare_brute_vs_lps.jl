# scripts/debug/compare_brute_vs_lps.jl
#
# 目的：
# - results/tmp/brute_changes.csv（前に作ったやつ）を読む
# - 同じ入力で LPS を実行して lps_changes.csv を作る
# - 両者の change point を tol 付きで突合して差分を出す

include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SetRegretCore.jl"))  # SetRegretCore

using .Paths
using .LoadInstance
using .SetRegretCore

using CSV
using DataFrames

# -------------------------
# utilities
# -------------------------
rank_str(rank::Vector{Int}) = join(rank, '|')

function parse_rank(s::AbstractString)
    isempty(s) && return Int[]
    return parse.(Int, split(s, '|'))
end

"""
two-pointer matching (descending lists)
returns:
- pairs :: Vector{Tuple{Float64,Float64}}  (brute_t, lps_t)
- miss_brute :: Vector{Float64}
- miss_lps   :: Vector{Float64}
"""
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
            # bruteの方が右（大きい）にある → LPSが追いついてない
            push!(miss_b, tb)
            i += 1
        else
            # lpsの方が右（大きい）にある → bruteに無い
            push!(miss_l, tl)
            j += 1
        end
    end

    while i <= length(b)
        push!(miss_b, b[i]); i += 1
    end
    while j <= length(l)
        push!(miss_l, l[j]); j += 1
    end

    return pairs, miss_b, miss_l
end

function main()
    # -------------------------
    # paths / input
    # -------------------------
    paths = Paths.project_paths()

    utility_v = LoadInstance.read_utility_value(paths, "u1")
    utility   = Matrix(utility_v[51])

    methodW = LoadInstance.read_method_weights(paths, "A/MMRW", 1, 6)
    wL = methodW[1].L
    wU = methodW[1].R

    # t-range
    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

    # brute scan steps と同じにする（あなたの brute と一致させる）
    steps = 5000
    Δ = (tR - tL) / steps
    tol = 20Δ

    # -------------------------
    # read brute csv
    # -------------------------
    outdir = joinpath(@__DIR__, "..", "..", "results", "tmp")
    brute_csv = joinpath(outdir, "brute_changes.csv")
    isfile(brute_csv) || error("not found: $brute_csv  （先に brute_scan を回して生成してね）")

    dfB = CSV.read(brute_csv, DataFrame)
    brute_ts = Vector{Float64}(dfB.t)
    brute_rank_map = Dict{Float64,Vector{Int}}()
    for r in eachrow(dfB)
        brute_rank_map[r.t] = parse_rank(r.rank)
    end

    # -------------------------
    # run LPS and write lps csv
    # -------------------------
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    res = SetRegretCore.run_lps(matrix, wL, wU, tL, tR)

    lps_ts = Vector{Float64}(res.changes)

    # LPS側：各tでのrankを timeline から拾う（tが一致する前提。ズレる場合は近傍から拾う）
    lps_rank_map = Dict{Float64,Vector{Int}}()
    for snap in res.timeline
        lps_rank_map[snap.t] = snap.rank
    end
    println(lps_ts)
    lps_csv = joinpath(outdir, "lps_changes.csv")
    open(lps_csv, "w") do io
        println(io, "t,rank")
        for t in sort(lps_ts; rev=true)
            # ぴったり無ければ近いスナップから拾う
            rank = get(lps_rank_map, t, Int[])
            if isempty(rank)
                # 近傍検索
                best_dt = Inf
                best_rank = Int[]
                for (tt, rr) in lps_rank_map
                    d = abs(tt - t)
                    if d < best_dt
                        best_dt = d
                        best_rank = rr
                    end
                end
                rank = best_rank
            end
            println(io, "$(t),$(rank_str(rank))")
        end
    end

    # -------------------------
    # compare
    # -------------------------
    pairs, miss_brute, miss_lps = match_changes(brute_ts, lps_ts, tol)

    # pairs report csv
    pair_csv = joinpath(outdir, "compare_pairs.csv")
    open(pair_csv, "w") do io
        println(io, "brute_t,lps_t,abs_diff,brute_rank,lps_rank")
        for (tb, tl) in pairs
            rb = get(brute_rank_map, tb, Int[])
            rl = get(lps_rank_map, tl, Int[])
            println(io, "$(tb),$(tl),$(abs(tb-tl)),$(rank_str(rb)),$(rank_str(rl))")
        end
    end

    # missing report csv
    miss_csv = joinpath(outdir, "compare_missing.csv")
    open(miss_csv, "w") do io
        println(io, "side,t,rank")
        for tb in sort(miss_brute; rev=true)
            println(io, "missing_in_LPS,$(tb),$(rank_str(get(brute_rank_map, tb, Int[])))")
        end
        for tl in sort(miss_lps; rev=true)
            println(io, "extra_in_LPS,$(tl),$(rank_str(get(lps_rank_map, tl, Int[])))")
        end
    end

    # console summary
    println("=== Compare brute vs LPS ===")
    println("tL=$tL, tR=$tR, steps=$steps, Δ=$Δ, tol=$tol")
    println("brute changes = $(length(brute_ts))")
    println("lps   changes = $(length(lps_ts))")
    println("matched       = $(length(pairs))")
    println("missing_in_LPS= $(length(miss_brute))  (bruteにあるのにLPSに無い)")
    println("extra_in_LPS  = $(length(miss_lps))    (LPSにあるのにbruteに無い)")
    println("Saved:")
    println("  $lps_csv")
    println("  $pair_csv")
    println("  $miss_csv")
end

main()

