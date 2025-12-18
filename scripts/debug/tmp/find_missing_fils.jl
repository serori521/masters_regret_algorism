#!/usr/bin/env julia
# scripts/find_missing_files.jl
#
# 使い方:
#   julia --project=. scripts/find_missing_files.jl
#
# 出力:
#   results/tmp/missing_inputs_N4-8.csv
#   results/tmp/missing_inputs_N4-8.txt

# -------------------------
# Includes (same style as other scripts)
# -------------------------
using Dates
using Logging
using Base.Threads: @threads, nthreads, SpinLock

include(joinpath(@__DIR__, "..",  "src", "paths.jl"))
include(joinpath(@__DIR__, "..",  "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..",  "src", "SetRegretCore.jl"))

# -------------------------
# Config
# -------------------------
const NS = 4:8
const M  = 5
const REPEAT_NUM = 1000

const UTILITIES = ["u1", "u2"]
const TRUE_WEIGHT_TYPES = ["A", "B", "C", "D", "E"]

const ACTIVE_METHOD_DIRS = [
 "AMRD", "AMRLD", "AMRW", "AMRWW", "DMIN",
 "E-AMRD", "E-AMRW", "E-AMRWW", "E-AMRd", "E-AMRw",
 "E-MMRD", "E-MMRW", "E-MMRWW", "E-MMRd", "E-MMRw",
 "E-MSD", "E-MSW", "E-MSWW", "EV",
 "G-AMRD", "G-AMRW", "G-AMRWW", "G-AMRd", "G-AMRw",
 "G-MMRD", "G-MMRW", "G-MMRWW", "G-MMRd", "G-MMRw",
 "G-MSD", "G-MSW", "G-MSWW", "GM",
 "MMRD", "MMRLD", "MMRW", "MMRWW",
 "MSD", "MSW", "MSWW", "WMIN",
 "eAMRd", "eAMRdc", "eAMRw", "eAMRwc",
 "eMMRd", "eMMRdc", "eMMRw", "eMMRwc",
 "gAMRd", "gAMRdc", "gAMRw", "gAMRwc",
 "gMMRd", "gMMRdc", "gMMRw", "gMMRwc"
]

# python側と揃えるため leading "/" を付けた表現も許容（内部ではcleanにする）
const METHOD_DIRS = ["/" * m for m in ACTIVE_METHOD_DIRS]

# -------------------------
# Helpers
# -------------------------
method_clean(m::AbstractString) = startswith(m, "/") ? m[2:end] : String(m)

function out_paths()
    project_root = normpath(joinpath(@__DIR__, ".."))
    outdir = joinpath(project_root, "results", "tmp")
    mkpath(outdir)
    csv_path = joinpath(outdir, "missing_inputs_N4-8.csv")
    txt_path = joinpath(outdir, "missing_inputs_N4-8.txt")
    return csv_path, txt_path
end

function push_missing!(rows::Vector{String}, kind::String;
                       N::Union{Int,Nothing}=nothing,
                       utility::Union{String,Nothing}=nothing,
                       tw::Union{String,Nothing}=nothing,
                       method::Union{String,Nothing}=nothing,
                       err::Any=nothing)
    nstr = isnothing(N) ? "" : string(N)
    ustr = isnothing(utility) ? "" : utility
    tstr = isnothing(tw) ? "" : tw
    mstr = isnothing(method) ? "" : method
    estr = isnothing(err) ? "" : string(typeof(err))
    # CSV: kind,N,utility,tw,method,error_type
    push!(rows, string(kind, ",", nstr, ",", ustr, ",", tstr, ",", mstr, ",", estr))
end
function precheck_missing_parallel(paths; Ns=NS)
    rows = String[]
    push!(rows, "kind,N,utility,tw,method,error_type")  # header

    lk = ReentrantLock()

    function add_row!(s::String)
        lock(lk)
        try
            push!(rows, s)
        finally
            unlock(lk)
        end
    end

    # ---- 1) utility / true weights ----
    tasks1 = Tuple{Symbol,Int,String}[]
    for N in Ns, utility in UTILITIES
        push!(tasks1, (:utility, N, utility))
    end
    for N in Ns, tw in TRUE_WEIGHT_TYPES
        push!(tasks1, (:truew, N, tw))
    end

    with_logger(SimpleLogger(stderr, Logging.Error)) do
        @threads for i in eachindex(tasks1)
            kind, N, key = tasks1[i]
            if kind == :utility
                utility = key
                try
                    LoadInstance.read_utility_value(paths, utility; N=N, M=M)
                catch err
                    add_row!("MISSING_UTILITY,$N,$utility,,, $(typeof(err))")
                end
            else
                tw = key
                try
                    LoadInstance.read_true_weights(paths, tw; N=N)
                catch err
                    add_row!("MISSING_TRUEW,$N,,$tw,, $(typeof(err))")
                end
            end
        end
    end

    # ---- 2) method weights ----
    tasks2 = Tuple{Int,String,String}[]
    for N in Ns, tw in TRUE_WEIGHT_TYPES, method in METHOD_DIRS
        push!(tasks2, (N, tw, method_clean(method)))
    end

    with_logger(SimpleLogger(stderr, Logging.Error)) do
        @threads for i in eachindex(tasks2)
            N, tw, mclean = tasks2[i]
            filename = joinpath(tw, mclean)
            try
                LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
            catch err
                add_row!("MISSING_METHODW,$N,,$tw,$mclean,$(typeof(err))")
            end
        end
    end

    return rows
end

function precheck_missing(paths; Ns=NS)
    rows = String[]
    push!(rows, "kind,N,utility,tw,method,error_type")  # header

    # utility / true weights は Nごとにチェック
    for N in Ns
        for utility in UTILITIES
            try
                LoadInstance.read_utility_value(paths, utility; N=N, M=M)
            catch err
                push_missing!(rows, "MISSING_UTILITY"; N=N, utility=utility, err=err)
            end
        end

        for tw in TRUE_WEIGHT_TYPES
            try
                LoadInstance.read_true_weights(paths, tw; N=N)
            catch err
                push_missing!(rows, "MISSING_TRUEW"; N=N, tw=tw, err=err)
            end
        end
    end

    # method weights は (N, tw, method) ごとにチェック
    for N in Ns, tw in TRUE_WEIGHT_TYPES, method in METHOD_DIRS
        filename = joinpath(tw, method_clean(method))
        try
            # 読めなければ不足扱い（いちばん簡単で確実）
            LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
        catch err
            push_missing!(rows, "MISSING_METHODW"; N=N, tw=tw, method=method_clean(method), err=err)
        end
    end

    return rows
end

function summarize(rows::Vector{String})
    # header除外
    body = rows[2:end]
    counts = Dict{String,Int}()
    for line in body
        kind = split(line, ',')[1]
        counts[kind] = get(counts, kind, 0) + 1
    end
    return counts
end

# -------------------------
# Main
# -------------------------
function main()
    paths = Paths.project_paths()

    println("=== find_missing_files.jl ===")
    println("Ns = ", collect(NS))
    println("REPEAT_NUM = ", REPEAT_NUM)
    println("METHODS = ", length(METHOD_DIRS))
    println("Start precheck...")
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        rows = precheck_missing_parallel(paths; Ns=NS)
        counts = summarize(rows)
        csv_path, txt_path = out_paths()
        open(csv_path, "w") do io
            for r in rows
                println(io, r)
            end
        end

        # 人間が見やすいテキストも出す
        open(txt_path, "w") do io
            println(io, "missing check @ ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
            println(io, "Ns = ", collect(NS))
            println(io, "methods = ", length(METHOD_DIRS))
            println(io, "")
            println(io, "== summary ==")
            for (k,v) in sort(collect(counts); by=x->x[1])
                println(io, lpad(k, 16), " : ", v)
            end
            println(io, "")
            println(io, "== details (CSV lines) ==")
            for r in rows[2:end]
                println(io, r)
            end
        end

        println("Done.")
        println("-> CSV: ", csv_path)
        println("-> TXT: ", txt_path)

        if isempty(rows) || length(rows) == 1
            println("No missing inputs detected.")
        else
            println("Summary:")
            for (k,v) in sort(collect(counts); by=x->x[1])
                println("  ", k, " = ", v)
            end
            println("Fix missing items, then rerun.")
        end
    end
    


end

main()
