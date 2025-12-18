# scripts/regret/write_minimax_regret_csv.jl
include(joinpath(@__DIR__, "..",  "src", "paths.jl"))
include(joinpath(@__DIR__, "..",  "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..",  "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using Base.Threads

# -------------------------
# Config
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

# python側と揃えるため leading "/" を付けた表現も許容（内部ではcleanにする）
const METHOD_DIRS = ["/" * m for m in ACTIVE_METHOD_DIRS]

# -------------------------
# Helpers
# -------------------------
@inline function method_clean(m::String)
    startswith(m, "/") ? m[2:end] : m
end

function out_csv_path(paths, utility::String, N::Int, tw::String, method::String)
    m = method_clean(method)
    dir = joinpath(paths.data, "a3", "regret", utility, "N=$(N)", tw, m)
    mkpath(dir)
    return joinpath(dir, "$(utility)_minimax_regret_$(REPEAT_NUM).csv")
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
# timeline から「評価点（両端含む）」を作る
@inline function points_from_res(res)
    ts    = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    return ts, ranks
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

# changes と timeline の対応：timeline[1]=tU, timeline[end]=tL, 途中が changes
@inline function change_ranks(res)
    k = length(res.changes)
    k == 0 && return Vector{Vector{Int}}()
    return [res.timeline[i+1].rank for i in 1:k]
end

function run_one_file(paths, utility::String, N::Int, tw::String, method::String;
                      force::Bool=false, eps::Float64=SetRegretCore.EPS_DEFAULT)

    outpath = out_csv_path(paths, utility, N, tw, method)

    if !force && file_complete(outpath)
        return :skip
    end

    # data 読み込み
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)

    # 真の重み
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)

    # 手法重み
    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        @warn "NO_WEIGHT (read_method_weights failed)" utility N tw method REPEAT_NUM filename
        return :no_weight
    end

    # （以下は元のまま）
    open(outpath, "w") do io
        for utl_num in 1:UTILITY_MATRIX_NUM
            U = Matrix(utility_mats[utl_num])

            matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
            res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
            true_ts, true_ranks = points_from_res(res_true)
            true_cnt = length(true_ts)

            for r in 1:REPEAT_NUM
                wL = methodW[r].L
                wU = methodW[r].R
                tL, tU = SetRegretCore.find_optimal_trange(wL, wU)

                matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
                res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
                m_ts, m_ranks = points_from_res(res_m)
                m_cnt = length(m_ts)

                println(io, join((utl_num, r, true_cnt, m_cnt), ','))
                println(io, join(vcat([""], string.(true_ts)), ','))

                for i in 1:m_cnt
                    concord = [count_concordant_pairs(m_ranks[i], true_ranks[j]) for j in 1:true_cnt]
                    println(io, join(vcat([string(m_ts[i])], string.(concord)), ','))
                end
            end
        end
    end

    return :ok
end

function precheck_missing(paths; Ns=NS)
    missing = String[]

    # utility / true weight は Nごとに一回だけチェック
    for N in Ns
        for utility in UTILITIES
            try
                LoadInstance.read_utility_value(paths, utility; N=N, M=M)
            catch err
                push!(missing, "MISSING_UTILITY,N=$N,utility=$utility,err=$(typeof(err))")
            end
        end
        for tw in TRUE_WEIGHT_TYPES
            try
                LoadInstance.read_true_weights(paths, tw; N=N)
            catch err
                push!(missing, "MISSING_TRUEW,N=$N,tw=$tw,err=$(typeof(err))")
            end
        end
    end

    # method weights は (N, tw, method) ごとにチェック
    for N in Ns, tw in TRUE_WEIGHT_TYPES, method in METHOD_DIRS
        filename = joinpath(tw, method_clean(method))
        try
            LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
        catch err
            push!(missing, "MISSING_METHODW,N=$N,tw=$tw,method=$(method_clean(method)),err=$(typeof(err))")
        end
    end

    # 出力：標準出力 + テキスト保存
    if !isempty(missing)
        @warn "PRECHECK found missing inputs" count=length(missing)
        out = joinpath(@__DIR__, "missing_inputs_N4-8.txt")
        open(out, "w") do io
            for s in missing
                println(io, s)
            end
        end
        @info "missing list written" out
    else
        @info "PRECHECK ok: no missing inputs"
    end
    return missing
end

# -------------------------
# Main
# -------------------------
function main(; force::Bool=false, do_precheck::Bool=true)
    paths = Paths.project_paths()

    if do_precheck
        missing = precheck_missing(paths; Ns=NS)
        if !isempty(missing)
            @warn "Abort run because missing inputs exist. Fix them, then rerun."  # 必要ならcontinue運用に変えてOK
            return
        end
    end

    tasks = [(N, tw, m) for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]

    for utility in UTILITIES
        @threads for idx in eachindex(tasks)
            (N, tw, m) = tasks[idx]
            st = run_one_file(paths, utility, N, tw, m; force=force)
            if st == :ok
                @info "done" utility N tw m tid=threadid()
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    @info "run_regret_lps_raw.jl start"  # これが出れば動いてる
    main()
end
