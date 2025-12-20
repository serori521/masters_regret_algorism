# scripts/run_maximinmaximax_lps_raw.jl
#
# maximin / maximax の raw（格子）出力。
# run_regret_lps_raw.jl と同じフォーマットで出力する。
#
# 出力先:
#   data/a3/<rule>/<utility>/N=<N>/<tw>/<method>/<utility>_<rule>_1000.csv
#   rule ∈ {maximin, maximax}
#
# 1ブロック（1つの utl_num と r）あたり:
#   line1: utl_num,r,true_cnt,m_cnt
#   line2: ,true_t1,true_t2,...,true_t_true_cnt
#   line3..: m_ti, concord(i,1), ..., concord(i,true_cnt)

include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using Base.Threads

# -------------------------
# Config (run_regret_lps_raw.jl に揃える)
# -------------------------
const NS = 4:8
const M = 5
const REPEAT_NUM = 1000
const UTILITY_MATRIX_NUM = 100

const UTILITIES = ["u1", "u2"]
const TRUE_WEIGHT_TYPES = ["A", "B", "C", "D", "E"]

const ACTIVE_METHOD_DIRS = [
    "AMRD", "AMRwc", "AMRW", "AMRWW", "DMIN",
    "E-AMRD", "E-AMRW", "E-AMRWW",
    "E-MMRD", "E-MMRW", "E-MMRWW",
    "E-DMIN", "E-WMIN", "E-WWMIN", "EV",
    "G-AMRD", "G-AMRW", "G-AMRWW",
    "G-MMRD", "G-MMRW", "G-MMRWW",
    "G-DMIN", "G-WMIN", "G-WWMIN", "GM",
    "MMRD",  "MMRwc", "MMRW", "MMRWW",
    "DMIN", "WMIN", "WWMIN", "WMIN",
    "eAMRd", "eAMRdc", "eAMRw", "eAMRwc",
    "eMMRd", "eMMRdc", "eMMRw", "eMMRwc",
    "gAMRd", "gAMRdc", "gAMRw", "gAMRwc",
    "gMMRd", "gMMRdc", "gMMRw", "gMMRwc"
]
const METHOD_DIRS = ["/" * m for m in ACTIVE_METHOD_DIRS]

# -------------------------
# Helpers
# -------------------------
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m

function out_csv_path(paths, rule::Symbol, utility::String, N::Int, tw::String, method::String)
    m = method_clean(method)
    rule_dir = rule == :maximin ? "maximin" : "maximax"
    dir = joinpath(paths.data, "a3", rule_dir, utility, "N=$(N)", tw, m)
    mkpath(dir)
    return joinpath(dir, "$(utility)_$(rule_dir)_$(REPEAT_NUM).csv")
end

function file_complete(path::String)
    isfile(path) || return false
    lines = readlines(path)
    isempty(lines) && return false
    s = max(1, length(lines) - 49)
    for line in lines[s:end]
        parts = split(line, ',')
        if length(parts) >= 2 &&
           parts[1] == string(UTILITY_MATRIX_NUM) &&
           parts[2] == string(REPEAT_NUM)
            return true
        end
    end
    return false
end

# rank1 の順序関係が rank2 と一致するペア数（Alt=5なら最大10）
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

# perm: 代替案ごとに基準の並び替え（maximinは昇順、maximaxは降順）
function build_perm(U::Matrix{Float64}, rule::Symbol)
    Alt, N = size(U)
    perm = Vector{Vector{Int}}(undef, Alt)
    rev = (rule == :maximax)
    @inbounds for a in 1:Alt
        perm[a] = sortperm(@view U[a, :]; rev=rev)
    end
    return perm
end

# C++ の maximin(u, yR, yL, totalU, z, perm, star) と同等（全代替案まとめて）
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

        # Process2
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

        # Process3
        j = perm[k][it]
        z[k, j] = 1.0 - cap + yL[j]
        star[k] = j
        it += 1

        # Process4
        while it <= N
            j = perm[k][it]
            z[k, j] = yL[j]
            it += 1
        end

        # total utility
        s = 0.0
        for i in 1:N
            s += U[k, i] * z[k, i]
        end
        totalU[k] = s
    end
    return nothing
end

# rank の中で a と b を入れ替える（a,b は代替案ID）
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

# C++ の while(t_snap>=tL) を移植:
# - 内部では「折れ点（slope change）」を使って探索するが、
#   返すのは「順位が変わった時刻」だけ（端点 tU,tL は必ず含む）。
function scan_timeline_maximinmaximax(U::Matrix{Float64}, wL::Vector{Float64}, wR::Vector{Float64}, rule::Symbol;
                                      epsi::Float64=1e-6, max_events::Int=200)
    Alt, N = size(U)
    tL, tU = SetRegretCore.find_optimal_trange(wL, wR)

    perm = build_perm(U, rule)

    yL = zeros(Float64, N)
    yR = zeros(Float64, N)
    yL2 = zeros(Float64, N)
    yR2 = zeros(Float64, N)

    totalU  = zeros(Float64, Alt)
    totalU2 = zeros(Float64, Alt)
    z  = zeros(Float64, Alt, N)
    z2 = zeros(Float64, Alt, N)
    star  = zeros(Int, Alt)
    star2 = zeros(Int, Alt)

    ts = Float64[tU]
    ranks = Vector{Vector{Int}}()

    # 初期 t=tU の rank
    @inbounds for i in 1:N
        yL[i] = wL[i] * tU
        yR[i] = wR[i] * tU
    end
    fill!(z, 0.0)
    maximin_totalU!(totalU, z, star, U, yL, yR, perm)
    rank = sortperm(totalU; rev=true)  # 1..Alt
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

        # 次に折れる点 r
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

        # U' (t_fold)
        @inbounds for i in 1:N
            yL2[i] = yL[i] * r
            yR2[i] = yR[i] * r
        end
        fill!(z2, 0.0)
        maximin_totalU!(totalU2, z2, star2, U, yL2, yR2, perm)

        # 交差候補（順位swap）
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
        sort!(crossings, by=x->x[1])  # r2 昇順

        current_rank = copy(ranks[end])
        if !isempty(crossings)
            for (r2, i, j) in reverse(crossings)  # 大きいr2から（降順tになる）
                t_cross = t_snap * r2
                push!(ts, t_cross)
                swap_in_rank!(current_rank, i, j)
                push!(ranks, copy(current_rank))
            end
        end

        # 折れ点へ進める（折れ点自体は出力しない）
        t_snap = t_fold
        if t_snap <= tL + 1e-15
            break
        end
    end

    # 端点 tL
    if isempty(ts) || ts[end] > tL + 1e-15
        push!(ts, tL)
        push!(ranks, copy(ranks[end]))
    elseif abs(ts[end] - tL) <= 1e-15
        if length(ranks) < length(ts)
            push!(ranks, copy(ranks[end]))
        end
    end

    return ts, ranks
end

# -------------------------
# Runner
# -------------------------
function run_one_file(paths, rule::Symbol, utility::String, N::Int, tw::String, method::String;
                      force::Bool=false, epsi::Float64=1e-6)

    outpath = out_csv_path(paths, rule, utility, N, tw, method)

    if !force && file_complete(outpath)
        println("skip")
        return :skip
    end

    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)

    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        @warn "NO_WEIGHT (read_method_weights failed)" rule utility N tw method REPEAT_NUM filename
        return :no_weight
    end
    repeat = min(REPEAT_NUM, length(methodW))

    open(outpath, "w") do io
        for utl_num in 1:UTILITY_MATRIX_NUM
            U = Matrix(utility_mats[utl_num])

            true_ts, true_ranks = scan_timeline_maximinmaximax(U, trueW.L, trueW.R, rule; epsi=epsi)
            true_cnt = length(true_ts)

            for r in 1:repeat
                wL = methodW[r].L
                wU = methodW[r].R

                m_ts, m_ranks = scan_timeline_maximinmaximax(U, wL, wU, rule; epsi=epsi)
                m_cnt = length(m_ts)

                println(io, join((utl_num, r, true_cnt, m_cnt), ','))        # 1行目
                println(io, join(vcat([""], string.(true_ts)), ','))         # 2行目（trueの横軸）

                for i in 1:m_cnt
                    concord = [count_concordant_pairs(m_ranks[i], true_ranks[j]) for j in 1:true_cnt]
                    println(io, join(vcat([string(m_ts[i])], string.(concord)), ','))
                end
            end
        end
    end

    return :ok
end

function main(; force::Bool=false)
    paths = Paths.project_paths()

    tasks = [(N, tw, m) for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]
    rules = (:maximin, :maximax)

    for rule in rules, utility in UTILITIES
        @threads for idx in eachindex(tasks)
            (N, tw, m) = tasks[idx]
            st = run_one_file(paths, rule, utility, N, tw, m; force=force)
            if st == :ok
                @info "done" rule utility N tw m tid=threadid()
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "run_maximinmaximax_lps_raw.jl start"
    main()
end
