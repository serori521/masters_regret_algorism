# scripts/regret/write_minimax_regret_grid_summaries_v2.jl
#
# minimax regret (LPS) の raw 格子を保存せず、
# 7指標（Precision/Recall/F1/DiagonalMean/Top1/Top2Comp/Top2Include）の
# 「100*1000 の和」だけを (N,tw,utility,method) ごとに1行で保存する。
#
# 注意点:
# ① 端点を入れた都合で「右端列」「下端行」が重複する
#    → すべてのセル系評価は (m_cnt-1)×(true_cnt-1) の範囲だけで計算
# ② diagonal は C[k,k] ではなく、(true tU→tL) と (pred tU→tL) を結ぶ直線上のセルを拾う
#
# 出力:
#   data/metrics_julia/grid_summary_minimax_regret_v2.csv

include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using Base.Threads
using Printf

# -------------------------
# Config（元ファイル踏襲）
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

const EPS = SetRegretCore.EPS_DEFAULT

# -------------------------
# Helpers
# -------------------------
@inline function method_clean(m::String)
    startswith(m, "/") ? m[2:end] : m
end

@inline function max_pairs(Alt::Int)
    return Alt * (Alt - 1) ÷ 2
end

# timeline から「評価点（両端含む）」を作る
@inline function points_from_res(res)
    ts    = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    # 安全のため t 降順に揃える（LPS想定では元から降順のはず）
    p = sortperm(ts; rev=true)
    return ts[p], ranks[p]
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

@inline top1_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1])
@inline top2_comp_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1]) && (r1[2] == r2[2])
@inline function top2_include_ok(r1::Vector{Int}, r2::Vector{Int})
    a1,a2 = r1[1], r1[2]
    b1,b2 = r2[1], r2[2]
    return (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)
end

# ts は降順。区間 i は [ts[i], ts[i+1]]
function find_interval_index(ts::Vector{Float64}, t::Float64)
    I = max(length(ts) - 1,1)
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

# diagonal: (true tU→tL) と (pred tU→tL) の直線上セルを拾う
function diagonal_mean_on_line(true_ts::Vector{Float64}, m_ts::Vector{Float64},
                               true_ranks::Vector{Vector{Int}}, m_ranks::Vector{Vector{Int}})
    # 注意点①: interval数で計算
    J = max(length(true_ts) - 1,1)  # true columns
    I = max(length(m_ts) - 1,1)     # pred rows
    Alt = length(true_ranks[1])
    denom_pairs = max_pairs(Alt)

    tU_true = true_ts[1]
    tL_true = true_ts[end]
    tU_pred = m_ts[1]
    tL_pred = m_ts[end]

    diag_sum = 0.0
    cnt = 0

    # true の各区間 j の中点を pred 側へ写像 → 属する pred 区間 i を選ぶ
    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])

        α = (t_true_mid - tU_true) / (tL_true - tU_true)
        t_pred_mid = tU_pred + α * (tL_pred - tU_pred)

        # clamp
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

# 1ケース（1 utl_num, 1 r）で 7指標を返す
function case_metrics(true_ts::Vector{Float64}, true_ranks::Vector{Vector{Int}},
                      m_ts::Vector{Float64}, m_ranks::Vector{Vector{Int}})
    true_cnt = length(true_ranks)
    m_cnt = length(m_ranks)

    # 注意点①: 右端列・下端行が重複 → 有効範囲は -1
    J = max(true_cnt - 1,1)  # true columns
    I = max(m_cnt - 1,1)     # pred rows

    Alt = length(true_ranks[1])
    denom_pairs = max_pairs(Alt)

    # ① precision: mean_i max_j C[i,j]  (i=1..I, j=1..J)
    prec_sum = 0.0
    for i in 1:I
        best = -1
        ri = m_ranks[i]
        for j in 1:J
            c = count_concordant_pairs(ri, true_ranks[j])
            if c > best; best = c; end
        end
        prec_sum += best
    end
    precision = (prec_sum / I) / denom_pairs

    # ② recall: mean_j max_i C[i,j]
    rec_sum = 0.0
    for j in 1:J
        best = -1
        rj = true_ranks[j]
        for i in 1:I
            c = count_concordant_pairs(m_ranks[i], rj)
            if c > best; best = c; end
        end
        rec_sum += best
    end
    recall = (rec_sum / J) / denom_pairs

    # ③ F1
    f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0.0

    # ④ diagonal mean（注意点②）
    diag_mean = diagonal_mean_on_line(true_ts, m_ts, true_ranks, m_ranks)

    # ⑤⑥⑦ セル一致率（注意点①）
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
    cell_sum = 0
    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            cell_sum += count_concordant_pairs(ri, true_ranks[j])
        end
    end
    full_mean = (cell_sum / (I*J)) / denom_pairs

    return precision, recall, f1, diag_mean, full_mean, top1_ratio, top2c_ratio, top2i_ratio

end

# -------------------------
# Summarize one (utility,N,tw,method)
# -------------------------
function summarize_one(paths, utility::String, N::Int, tw::String, method::String; eps::Float64=EPS)
    # data 読み込み（1回）
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)

    trueW = LoadInstance.read_true_weights(paths, tw; N=N)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)

    filename = joinpath(tw, method_clean(method))
    methodW = try
        LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    catch
        @warn "NO_WEIGHT" utility N tw method
        return nothing
    end
    repeat = min(REPEAT_NUM, length(methodW))
    repeat == 0 && return nothing

    # threads 集計（utl_numを並列）
    T = nthreads()
    sumFull = zeros(Float64, T)
    sumP   = zeros(Float64, T)
    sumR   = zeros(Float64, T)
    sumF   = zeros(Float64, T)
    sumD   = zeros(Float64, T)
    sumT1  = zeros(Float64, T)
    sumT2c = zeros(Float64, T)
    sumT2i = zeros(Float64, T)
    cases  = zeros(Int, T)

    @threads for utl_num in 1:UTILITY_MATRIX_NUM
        tid = threadid()
        U = Matrix(utility_mats[utl_num])

        # 真（この U では固定）
        matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
        res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
        true_ts, true_ranks = points_from_res(res_true)

        for r in 1:repeat
            wL = methodW[r].L
            wU = methodW[r].R
            tL, tU = SetRegretCore.find_optimal_trange(wL, wU)

            matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
            res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
            m_ts, m_ranks = points_from_res(res_m)

            p, rr, f1, d,full, t1, t2c, t2i = case_metrics(true_ts, true_ranks, m_ts, m_ranks)
            sumFull[tid] += full
            sumP[tid]   += p
            sumR[tid]   += rr
            sumF[tid]   += f1
            sumD[tid]   += d
            sumT1[tid]  += t1
            sumT2c[tid] += t2c
            sumT2i[tid] += t2i
            cases[tid]  += 1
        end
    end

    total_cases = sum(cases)
    total_cases == 0 && return nothing

    return (
        sum_precision    = sum(sumP),
        sum_recall       = sum(sumR),
        sum_f1           = sum(sumF),
        sum_diag_mean    = sum(sumD),
        sum_full_mean    = sum(sumFull),
        sum_top1         = sum(sumT1),
        sum_top2_comp    = sum(sumT2c),
        sum_top2_include = sum(sumT2i),
        cases            = total_cases
    )
end

# -------------------------
# Main
# -------------------------
const WRITE_LOCK = ReentrantLock()

function main()
    paths = Paths.project_paths()

    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)
    outpath = joinpath(outdir, "grid_summary_minimax_regret_v2.csv")

    # ヘッダだけ先に作る（追記運用）
    if !isfile(outpath)
        open(outpath, "w") do io
            println(io, join([
                "rule","N","tw","utility","method",
                "sum_precision","sum_recall","sum_f1","sum_diag_mean","sum_full_mean",
                "sum_top1","sum_top2_comp","sum_top2_include",
                "cases"
            ], ','))
        end
    end

    tasks = [(utility, N, tw, m) for utility in UTILITIES for N in NS for tw in TRUE_WEIGHT_TYPES for m in METHOD_DIRS]

    @threads for idx in eachindex(tasks)
        (utility, N, tw, method) = tasks[idx]
        res = summarize_one(paths, utility, N, tw, method)
        res === nothing && continue

        lock(WRITE_LOCK) do
            open(outpath, "a") do io
                println(io, join([
                    "minimax_regret",
                    string(N),
                    tw,
                    utility,
                    method_clean(method),
                    @sprintf("%.10f", res.sum_precision),
                    @sprintf("%.10f", res.sum_recall),
                    @sprintf("%.10f", res.sum_f1),
                    @sprintf("%.10f", res.sum_diag_mean),
                    @sprintf("%.10f", res.sum_full_mean),
                    @sprintf("%.10f", res.sum_top1),
                    @sprintf("%.10f", res.sum_top2_comp),
                    @sprintf("%.10f", res.sum_top2_include),
                    string(res.cases)
                ], ','))
            end
        end

        @info "done" utility N tw method cases=res.cases tid=threadid()
    end

    @info "saved" outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "write_minimax_regret_grid_summaries_v2.jl start"
    main()
end
