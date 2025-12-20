# scripts/run_maximinmaximax_grid_summaries_v2.jl
#
# Maximin / Maximax: 格子rawを保存せず、7指標のみを「100*1000の和」で保存する。
# 重要な修正点:
#  (1) tL端を入れた都合で、右端列・下端行が重複する → 集計では最後の行/列を除外 (I=m_cnt-1, J=true_cnt-1)
#  (2) diagonal は C[k,k] ではなく、(true tU→tL) と (pred tU→tL) を結ぶ直線に沿うセルを拾う
#
# 出力:
#   data/metrics_julia/grid_summary_maximinmaximax_v2.csv
#
# 1行 = (rule,N,tw,utility,method) ごとの合計:
#   sum_precision, sum_recall, sum_f1, sum_diag_mean, sum_top1, sum_top2_comp, sum_top2_include
#   cases = 100*repeat (通常100*1000)

include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using Base.Threads
using Printf

# -------------------------
# Config（run_regret_lps_raw.jl に寄せる）
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

const EPSI = 1e-6

# -------------------------
# Helpers
# -------------------------
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m

@inline function max_pairs(Alt::Int)
    return Alt * (Alt - 1) ÷ 2
end

# rank1の順序関係が rank2 と一致するペア数（Alt=5なら最大10）
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

# C++ の maximin() 相当（全代替案まとめて totalU と star を出す）
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

# C++スキャンを移植（返すのは「順位が変わった時刻＋端点」と順位列）
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

    # 初期 t=tU
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

        # 次の折れ点
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

        # t_fold側
        @inbounds for i in 1:N
            yL2[i] = yL[i] * r
            yR2[i] = yR[i] * r
        end
        fill!(z2, 0.0)
        maximin_totalU!(totalU2, z2, star2, U, yL2, yR2, perm)

        # 交差候補
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
        sort!(crossings, by=x->x[1])  # r2昇順

        current_rank = copy(ranks[end])
        if !isempty(crossings)
            # tは減少方向なので、大きいr2（=大きいt_cross）から入れる
            for (r2, i, j) in reverse(crossings)
                t_cross = t_snap * r2
                push!(ts, t_cross)
                swap_in_rank!(current_rank, i, j)
                push!(ranks, copy(current_rank))
            end
        end

        # 折れ点へ進める（折れ自体は保存しない）
        t_snap = t_fold
        if t_snap <= tL + 1e-15
            break
        end
    end

    # 端点 tL
    if isempty(ts) || ts[end] > tL + 1e-15
        push!(ts, tL)
        push!(ranks, copy(ranks[end]))
    end

    return ts, ranks
end

# tsは降順。区間 i は [ts[i], ts[i+1]] とみなす。
# t がどの区間に属するか i を返す（1..I）。外側は端に丸める。
function find_interval_index(ts::Vector{Float64}, t::Float64)
    I = max(length(ts) - 1,1)
    # clamp
    t_hi = ts[1]
    t_lo = ts[end]
    if t >= t_hi
        return 1
    elseif t <= t_lo
        return I
    end
    # 二分探索（降順）
    lo = 1
    hi = I
    while lo <= hi
        mid = (lo + hi) >>> 1
        if ts[mid] >= t >= ts[mid+1]
            return mid
        elseif t > ts[mid]          # もっと上（小さいindex）へ
            hi = mid - 1
        else                         # t < ts[mid+1] 方向へ
            lo = mid + 1
        end
    end
    return clamp(lo, 1, I)
end

# (true tU→tL) と (pred tU→tL) を結ぶ直線に沿う diagonal セル列を拾って平均
function diagonal_mean_on_line(true_ts::Vector{Float64}, m_ts::Vector{Float64},
                               true_ranks::Vector{Vector{Int}}, m_ranks::Vector{Vector{Int}})
    # 注意点①: 右端列・下端行を除外 → interval数で扱う

    J = max(length(true_ts) - 1,1)  # true columns
    I = max(length(m_ts) - 1,1)     # pred rows

    tU_true = true_ts[1]
    tL_true = true_ts[end]
    tU_pred = m_ts[1]
    tL_pred = m_ts[end]

    denom_pairs = max_pairs(length(true_ranks[1]))

    diag_sum = 0.0
    cnt = 0

    # true の各区間 j の中点を直線で pred に写して、その点が属する pred 区間 i を取る
    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])

        # 直線写像: t_pred(t)
        # t_pred = tU_pred + (t - tU_true)/(tL_true - tU_true) * (tL_pred - tU_pred)
        α = (t_true_mid - tU_true) / (tL_true - tU_true)
        t_pred_mid = tU_pred + α * (tL_pred - tU_pred)

        # 数値誤差のclamp（端点外に出ないように）
        if t_pred_mid > tU_pred
            t_pred_mid = tU_pred
        elseif t_pred_mid < tL_pred
            t_pred_mid = tL_pred
        end

        i = find_interval_index(m_ts, t_pred_mid)
        # ここも注意点①: i=1..I, j=1..J の範囲で rank を比較（末尾は除外）
        # ranks配列は tsと同長で入っているので、区間iに対応する代表として ranks[i] を使う

        c = count_concordant_pairs(m_ranks[i], true_ranks[j])
        diag_sum += c
        cnt += 1
    end

    return (diag_sum / cnt) / denom_pairs
end

# 1ケース（1つのU, 1つの推定重み）で7指標を返す（raw格子保存なし）
function case_metrics(true_ts::Vector{Float64}, true_ranks::Vector{Vector{Int}},
                      m_ts::Vector{Float64}, m_ranks::Vector{Vector{Int}})
    true_cnt = length(true_ranks)
    m_cnt = length(m_ranks)

    # 注意点①: 右端列・下端行が重複 → 有効な範囲は -1
    J = max(true_cnt - 1,1)  # true columns
    I = max(m_cnt - 1,1)     # pred rows

    Alt = length(true_ranks[1])
    denom_pairs = max_pairs(Alt)  # 10

    # ① precision: mean_i max_j C[i,j] (i=1..I, j=1..J)
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

    # ③ F1: ケースごとに計算してから合計する（あなたの要望どおり）
    f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0.0

    # ④ diagonal mean（注意点②: 直線上のセル）
    diag_mean = diagonal_mean_on_line(true_ts, m_ts, true_ranks, m_ranks)

    # ⑤⑥⑦ セル割合（注意点①: i=1..I, j=1..J のみ）
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
# Main: summarize (no raw)
# -------------------------
function main()
    paths = Paths.project_paths()
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)
    outpath = joinpath(outdir, "grid_summary_maximinmaximax_v3.csv")

    open(outpath, "w") do io
        println(io, join([
            "rule","N","tw","utility","method",
            "sum_precision","sum_recall","sum_f1","sum_diag_mean","sum_full_mean",
            "sum_top1","sum_top2_comp","sum_top2_include",
            "cases"
        ], ','))

        rules = (:maximin, :maximax)

        for rule in rules, utility in UTILITIES, N in NS, tw in TRUE_WEIGHT_TYPES, method in METHOD_DIRS
            # 入力
            utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
            trueW = LoadInstance.read_true_weights(paths, tw; N=N)

            filename = joinpath(tw, method_clean(method))
            methodW = try
                LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
            catch
                @warn "NO_WEIGHT" rule utility N tw method
                continue
            end
            repeat = min(REPEAT_NUM, length(methodW))

            # threads集計（utl_num方向に並列化）
            T = nthreads()
            sumFull = zeros(Float64, T)
            sumP = zeros(Float64, T)
            sumR = zeros(Float64, T)
            sumF = zeros(Float64, T)
            sumD = zeros(Float64, T)
            sumT1 = zeros(Float64, T)
            sumT2c = zeros(Float64, T)
            sumT2i = zeros(Float64, T)
            cases = zeros(Int, T)

            @threads for utl_num in 1:UTILITY_MATRIX_NUM
                tid = threadid()
                U = Matrix(utility_mats[utl_num])

                # 真側（このUでは固定）
                true_ts, true_ranks = scan_timeline_maximinmaximax(U, trueW.L, trueW.R, rule; epsi=EPSI)

                for r in 1:repeat
                    wL = methodW[r].L
                    wU = methodW[r].R

                    m_ts, m_ranks = scan_timeline_maximinmaximax(U, wL, wU, rule; epsi=EPSI)
                    p, rr, f1, d, fm, t1, t2c, t2i = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

                    sumFull[tid] += fm
                    sumP[tid] += p
                    sumR[tid] += rr
                    sumF[tid] += f1
                    sumD[tid] += d
                    sumT1[tid] += t1
                    sumT2c[tid] += t2c
                    sumT2i[tid] += t2i
                    cases[tid] += 1
                end
            end

            total_cases = sum(cases)
            if total_cases == 0
                @warn "no cases" rule utility N tw method
                continue
            end

            # 100*1000 の「和」を出す（あなたの要望どおり）
            println(io, join([
                String(rule), string(N), tw, utility, method_clean(method),
                @sprintf("%.10f", sum(sumP)),
                @sprintf("%.10f", sum(sumR)),
                @sprintf("%.10f", sum(sumF)),
                @sprintf("%.10f", sum(sumD)),
                @sprintf("%.10f", sum(sumFull)),
                @sprintf("%.10f", sum(sumT1)),
                @sprintf("%.10f", sum(sumT2c)),
                @sprintf("%.10f", sum(sumT2i)),
                string(total_cases)
            ], ','))

            @info "done" rule N tw utility method total_cases
        end
    end

    @info "saved" outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "run_maximinmaximax_grid_summaries_v2.jl start"
    main()
end
