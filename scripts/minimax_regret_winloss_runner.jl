# scripts/minimax_regret_winloss_runner.jl
#
# Minimax Regret (LPS) — Win/Loss logger against EV / GM per "problem".
#
# "Problem" here is (PCM repeat r) × (utility matrix utl_num), i.e. 1000×100 cases per (utility,N,tw).
#
# This runner computes, for each (utility, N, true_weight_type, method):
#   - win/loss/tie counts vs EV and vs GM
#   - average margin (e.g., F1(method) - F1(EV)) and loss-only stats
#   - optional per-utility-matrix (utl_num) aggregates
#
# Important design:
#   - True timeline is computed once per utl_num (NOT per method) to avoid redundant work.
#   - EV/GM and each method are computed on the same (utl_num, r) so comparisons are aligned.
#
# Run:
#   julia --project=. scripts/minimax_regret_winloss_runner.jl

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

# Methods to evaluate (include EV/GM as baselines; others compared against them)
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

# eps for LPS
const EPS_REGRET = SetRegretCore.EPS_DEFAULT

# Logging switches
const WRITE_BY_UTILITY_MATRIX = true   # writes per-utl_num aggregates (more rows)

# -------------------------
# Shared helpers
# -------------------------

"""/ を消して method 名を正規化する。"""
@inline method_clean(m::String) = startswith(m, "/") ? m[2:end] : m

"""代替案数 Alt のペア数 C(Alt,2)。minimax regret では Alt=M=5 なので 10。"""
@inline max_pairs(Alt::Int) = Alt * (Alt - 1) ÷ 2

"""rank1 と rank2 の一致ペア数（concordant pairs）を数える。"""
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

"""t がどの区間 [ts[k], ts[k+1]] に属するか（ts は降順）を二分探索で返す。"""
function find_interval_index(ts::Vector{Float64}, t::Float64)
    I = max(length(ts) - 1, 1)
    t_hi, t_lo = ts[1], ts[end]
    if t >= t_hi; return 1; elseif t <= t_lo; return I; end
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

"""true の区間上の中点を、予測側の t に線形対応させたときの diagonal mean を計算する。

注意：I,J で割る（区間数依存を消す）のは必須。
ここでは denom_pairs では割らず、一致ペア数スケール（0..denom_pairs）で返す。
"""
function diagonal_mean_on_line(true_ts::Vector{Float64}, m_ts::Vector{Float64},
                               true_ranks::Vector{Vector{Int}}, m_ranks::Vector{Vector{Int}})
    J = max(length(true_ts) - 1, 1)
    tU_true, tL_true = true_ts[1], true_ts[end]
    tU_pred, tL_pred = m_ts[1], m_ts[end]

    diag_sum = 0.0
    cnt = 0
    for j in 1:J
        t_true_mid = 0.5 * (true_ts[j] + true_ts[j+1])
        α = (t_true_mid - tU_true) / (tL_true - tU_true)
        t_pred_mid = tU_pred + α * (tL_pred - tU_pred)
        t_pred_mid = clamp(t_pred_mid, tL_pred, tU_pred)
        i = find_interval_index(m_ts, t_pred_mid)
        diag_sum += count_concordant_pairs(m_ranks[i], true_ranks[j])
        cnt += 1
    end
    return diag_sum / max(cnt, 1)
end

"""1ケース(=ある true timeline と method timeline)の指標を計算する。

- Precision / Recall: best-match を取り、I/J で平均（区間数依存を消す）。
- denom_pairs では割らない：一致ペア数のまま返す（Alt=5なら 0..10）。
- F1 は上の P,R から計算（スケールは P,R と同じ）。
- FullMean / Top1/Top2: 全セル平均・率も返す（率は0..1）。

戻り値:
  precision, recall, f1, diag_mean, full_mean, top1_rate, top2c_rate, top2i_rate
"""
function case_metrics(true_ts::Vector{Float64}, true_ranks::Vector{Vector{Int}},
                      m_ts::Vector{Float64}, m_ranks::Vector{Vector{Int}})
    J = max(length(true_ranks) - 1, 1)
    I = max(length(m_ranks) - 1, 1)

    # Precision (平均は I で取る)
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
    precision = prec_sum / I

    # Recall (平均は J で取る)
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
    recall = rec_sum / J

    f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0.0
    diag_mean = diagonal_mean_on_line(true_ts, m_ts, true_ranks, m_ranks)

    total_cells = I * J
    top1_cnt = 0; top2c_cnt = 0; top2i_cnt = 0
    cell_sum = 0.0

    @inline top1_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1])
    @inline top2_comp_ok(r1::Vector{Int}, r2::Vector{Int}) = (r1[1] == r2[1]) && (r1[2] == r2[2])
    @inline function top2_include_ok(r1::Vector{Int}, r2::Vector{Int})
        a1,a2 = r1[1], r1[2]
        b1,b2 = r2[1], r2[2]
        return (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)
    end

    for i in 1:I
        ri = m_ranks[i]
        for j in 1:J
            rj = true_ranks[j]
            top1_cnt  += top1_ok(ri, rj) ? 1 : 0
            top2c_cnt += top2_comp_ok(ri, rj) ? 1 : 0
            top2i_cnt += top2_include_ok(ri, rj) ? 1 : 0
            cell_sum  += count_concordant_pairs(ri, rj)
        end
    end

    full_mean = cell_sum / total_cells
    return precision, recall, f1, diag_mean, full_mean,
           top1_cnt/total_cells, top2c_cnt/total_cells, top2i_cnt/total_cells
end

"""LPS の res から (ts, ranks) を取り出して、t降順に揃える。"""
function points_from_res(res)
    ts = [e.t for e in res.timeline]
    ranks = [e.rank for e in res.timeline]
    p = sortperm(ts; rev=true)
    return ts[p], ranks[p]
end

# -------------------------
# Win/Loss accumulators
# -------------------------

"""勝敗カウントとマージン統計（平均との差など）を保持する。"""
mutable struct WLStats
    wins::Int
    losses::Int
    ties::Int
    sum_margin::Float64
    sum_pos_margin::Float64
    sum_neg_margin::Float64
    max_win_margin::Float64
    max_loss_margin::Float64
end

function WLStats()
    return WLStats(0,0,0, 0.0,0.0,0.0, -Inf, +Inf)
end

"""margin=score(method)-score(base) で勝敗と統計を更新する。"""
@inline function update_wl!(s::WLStats, margin::Float64; eps_tie::Float64=1e-12)
    s.sum_margin += margin
    if margin > eps_tie
        s.wins += 1
        s.sum_pos_margin += margin
        s.max_win_margin = max(s.max_win_margin, margin)
    elseif margin < -eps_tie
        s.losses += 1
        s.sum_neg_margin += margin
        s.max_loss_margin = min(s.max_loss_margin, margin)  # most negative
    else
        s.ties += 1
    end
end

"""複数指標（F1, FullMean, Top1）について勝敗をまとめて保持する。"""
mutable struct WLBundle
    f1_vs_ev::WLStats
    f1_vs_gm::WLStats
    full_vs_ev::WLStats
    full_vs_gm::WLStats
    top1_vs_ev::WLStats
    top1_vs_gm::WLStats
end

function WLBundle()
    return WLBundle(WLStats(), WLStats(), WLStats(), WLStats(), WLStats(), WLStats())
end

# -------------------------
# Core runner
# -------------------------

"""(utility,N,tw) 固定で、全methodの win/loss を計算する。

ループ単位:
  utl_num(=効用行列) ごとに true timeline を1回
  その後 r(=PCM repeat) ごとに EV/GM + method を同一Uで計算し、勝敗を更新

返り値:
  bundles::Dict{String, WLBundle}  (method => stats)
  (optional) byU::Dict{Tuple{String,Int}, Dict{String, WLBundle}}  ((method,utl_num)ごとのstats)
"""
function winloss_for_setting(paths, utility::String, N::Int, tw::String;
                             eps::Float64=EPS_REGRET, write_byU::Bool=WRITE_BY_UTILITY_MATRIX)

    # --- Load all U and true weights ---
    utility_mats = LoadInstance.read_utility_value(paths, utility; N=N, M=M)
    trueW = LoadInstance.read_true_weights(paths, tw; N=N)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)

    # --- Load all method weights (vector length = REPEAT_NUM) ---
    methodW_map = Dict{String, Any}()
    for m in METHOD_DIRS
        filename = joinpath(tw, method_clean(m))
        methodW = try
            LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
        catch
            nothing
        end
        methodW === nothing && continue
        repeat = min(REPEAT_NUM, length(methodW))
        repeat == 0 && continue
        methodW_map[method_clean(m)] = methodW
    end

    # Require EV and GM as baselines
    if !haskey(methodW_map, "EV") || !haskey(methodW_map, "GM")
        error("EV/GM weights not found for (utility=$utility, N=$N, tw=$tw).")
    end

    # Determine repeat count from EV (assumed aligned across methods)
    repeat = min(REPEAT_NUM, length(methodW_map["EV"]))

    bundles = Dict{String, WLBundle}()
    for (m, _) in methodW_map
        bundles[m] = WLBundle()
    end

    byU = write_byU ? Dict{Tuple{String,Int}, WLBundle}() : Dict{Tuple{String,Int}, WLBundle}()

    # --- Main loops ---
    for utl_num in 1:UTILITY_MATRIX_NUM
        U = Matrix(utility_mats[utl_num])

        # True timeline (computed once per U)
        matrix_true = SetRegretCore.create_minimax_R_Matrix(U)
        res_true = SetRegretCore.run_lps(matrix_true, trueW.L, trueW.R, tL_true, tU_true; eps=eps)
        true_ts, true_ranks = points_from_res(res_true)

        # For each repeat r, compute EV/GM once, then compare each method
        for r in 1:repeat
            # --- EV ---
            wL_ev, wU_ev = methodW_map["EV"][r].L, methodW_map["EV"][r].R
            tL_ev, tU_ev = SetRegretCore.find_optimal_trange(wL_ev, wU_ev)
            matrix_ev = SetRegretCore.create_minimax_R_Matrix(U)
            res_ev = SetRegretCore.run_lps(matrix_ev, wL_ev, wU_ev, tL_ev, tU_ev; eps=eps)
            ev_ts, ev_ranks = points_from_res(res_ev)
            _, _, f1_ev, _, full_ev, top1_ev, _, _ = case_metrics(true_ts, true_ranks, ev_ts, ev_ranks)

            # --- GM ---
            wL_gm, wU_gm = methodW_map["GM"][r].L, methodW_map["GM"][r].R
            tL_gm, tU_gm = SetRegretCore.find_optimal_trange(wL_gm, wU_gm)
            matrix_gm = SetRegretCore.create_minimax_R_Matrix(U)
            res_gm = SetRegretCore.run_lps(matrix_gm, wL_gm, wU_gm, tL_gm, tU_gm; eps=eps)
            gm_ts, gm_ranks = points_from_res(res_gm)
            _, _, f1_gm, _, full_gm, top1_gm, _, _ = case_metrics(true_ts, true_ranks, gm_ts, gm_ranks)

            # --- Others (including EV/GM themselves; margin will be 0 for self) ---
            for (m, methodW) in methodW_map
                wL, wU = methodW[r].L, methodW[r].R
                tL, tU = SetRegretCore.find_optimal_trange(wL, wU)
                matrix_m = SetRegretCore.create_minimax_R_Matrix(U)
                res_m = SetRegretCore.run_lps(matrix_m, wL, wU, tL, tU; eps=eps)
                m_ts, m_ranks = points_from_res(res_m)

                _, _, f1_m, _, full_m, top1_m, _, _ = case_metrics(true_ts, true_ranks, m_ts, m_ranks)

                b = bundles[m]
                update_wl!(b.f1_vs_ev,   f1_m   - f1_ev)
                update_wl!(b.f1_vs_gm,   f1_m   - f1_gm)
                update_wl!(b.full_vs_ev, full_m - full_ev)
                update_wl!(b.full_vs_gm, full_m - full_gm)
                update_wl!(b.top1_vs_ev, top1_m - top1_ev)
                update_wl!(b.top1_vs_gm, top1_m - top1_gm)

                if write_byU
                    key = (m, utl_num)
                    bu = get!(byU, key, WLBundle())
                    update_wl!(bu.f1_vs_ev,   f1_m   - f1_ev)
                    update_wl!(bu.f1_vs_gm,   f1_m   - f1_gm)
                    update_wl!(bu.full_vs_ev, full_m - full_ev)
                    update_wl!(bu.full_vs_gm, full_m - full_gm)
                    update_wl!(bu.top1_vs_ev, top1_m - top1_ev)
                    update_wl!(bu.top1_vs_gm, top1_m - top1_gm)
                end
            end
        end
    end

    return bundles, byU, repeat
end

"""WLStats をCSVに書ける形（wins/losses/ties, mean marginなど）に展開する。"""
function wl_to_fields(s::WLStats, total_cases::Int)
    mean_margin = s.sum_margin / max(total_cases, 1)
    mean_pos = s.wins > 0 ? (s.sum_pos_margin / s.wins) : 0.0
    mean_neg = s.losses > 0 ? (s.sum_neg_margin / s.losses) : 0.0
    max_win = isfinite(s.max_win_margin) ? s.max_win_margin : 0.0
    max_loss = isfinite(s.max_loss_margin) ? s.max_loss_margin : 0.0
    return (s.wins, s.losses, s.ties, mean_margin, mean_pos, mean_neg, max_win, max_loss)
end

# -------------------------
# Master entry
# -------------------------

function main()
    paths = Paths.project_paths()
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)

    out_summary = joinpath(outdir, "minimax_regret_winloss_summary_v1.csv")
    out_byU = joinpath(outdir, "minimax_regret_winloss_byU_v1.csv")

    open(out_summary, "w") do io
        println(io, join([
            "rule","utility","N","tw","method","cases",
            # F1 vs EV
            "f1_wins_vs_EV","f1_losses_vs_EV","f1_ties_vs_EV","f1_mean_margin_vs_EV","f1_mean_win_margin_vs_EV","f1_mean_loss_margin_vs_EV","f1_max_win_margin_vs_EV","f1_max_loss_margin_vs_EV",
            # F1 vs GM
            "f1_wins_vs_GM","f1_losses_vs_GM","f1_ties_vs_GM","f1_mean_margin_vs_GM","f1_mean_win_margin_vs_GM","f1_mean_loss_margin_vs_GM","f1_max_win_margin_vs_GM","f1_max_loss_margin_vs_GM",
            # FullMean vs EV
            "full_wins_vs_EV","full_losses_vs_EV","full_ties_vs_EV","full_mean_margin_vs_EV","full_mean_win_margin_vs_EV","full_mean_loss_margin_vs_EV","full_max_win_margin_vs_EV","full_max_loss_margin_vs_EV",
            # FullMean vs GM
            "full_wins_vs_GM","full_losses_vs_GM","full_ties_vs_GM","full_mean_margin_vs_GM","full_mean_win_margin_vs_GM","full_mean_loss_margin_vs_GM","full_max_win_margin_vs_GM","full_max_loss_margin_vs_GM",
            # Top1 vs EV
            "top1_wins_vs_EV","top1_losses_vs_EV","top1_ties_vs_EV","top1_mean_margin_vs_EV","top1_mean_win_margin_vs_EV","top1_mean_loss_margin_vs_EV","top1_max_win_margin_vs_EV","top1_max_loss_margin_vs_EV",
            # Top1 vs GM
            "top1_wins_vs_GM","top1_losses_vs_GM","top1_ties_vs_GM","top1_mean_margin_vs_GM","top1_mean_win_margin_vs_GM","top1_mean_loss_margin_vs_GM","top1_max_win_margin_vs_GM","top1_max_loss_margin_vs_GM"
        ], ','))

        for utility in UTILITIES
            for N in NS
                for tw in TRUE_WEIGHT_TYPES
                    @info "Win/Loss setting" utility N tw
                    bundles, byU, repeat = winloss_for_setting(paths, utility, N, tw)
                    total_cases = UTILITY_MATRIX_NUM * repeat

                    # Write summary per method
                    for (m, b) in sort(collect(bundles); by=x->x[1])
                        f1ev = wl_to_fields(b.f1_vs_ev, total_cases)
                        f1gm = wl_to_fields(b.f1_vs_gm, total_cases)
                        fev  = wl_to_fields(b.full_vs_ev, total_cases)
                        fgm  = wl_to_fields(b.full_vs_gm, total_cases)
                        t1ev = wl_to_fields(b.top1_vs_ev, total_cases)
                        t1gm = wl_to_fields(b.top1_vs_gm, total_cases)

                        println(io, join([
                            "minimax_regret", utility, string(N), tw, m, string(total_cases),
                            # f1 vs EV
                            string(f1ev[1]), string(f1ev[2]), string(f1ev[3]), @sprintf("%.10f", f1ev[4]), @sprintf("%.10f", f1ev[5]), @sprintf("%.10f", f1ev[6]), @sprintf("%.10f", f1ev[7]), @sprintf("%.10f", f1ev[8]),
                            # f1 vs GM
                            string(f1gm[1]), string(f1gm[2]), string(f1gm[3]), @sprintf("%.10f", f1gm[4]), @sprintf("%.10f", f1gm[5]), @sprintf("%.10f", f1gm[6]), @sprintf("%.10f", f1gm[7]), @sprintf("%.10f", f1gm[8]),
                            # full vs EV
                            string(fev[1]), string(fev[2]), string(fev[3]), @sprintf("%.10f", fev[4]), @sprintf("%.10f", fev[5]), @sprintf("%.10f", fev[6]), @sprintf("%.10f", fev[7]), @sprintf("%.10f", fev[8]),
                            # full vs GM
                            string(fgm[1]), string(fgm[2]), string(fgm[3]), @sprintf("%.10f", fgm[4]), @sprintf("%.10f", fgm[5]), @sprintf("%.10f", fgm[6]), @sprintf("%.10f", fgm[7]), @sprintf("%.10f", fgm[8]),
                            # top1 vs EV
                            string(t1ev[1]), string(t1ev[2]), string(t1ev[3]), @sprintf("%.10f", t1ev[4]), @sprintf("%.10f", t1ev[5]), @sprintf("%.10f", t1ev[6]), @sprintf("%.10f", t1ev[7]), @sprintf("%.10f", t1ev[8]),
                            # top1 vs GM
                            string(t1gm[1]), string(t1gm[2]), string(t1gm[3]), @sprintf("%.10f", t1gm[4]), @sprintf("%.10f", t1gm[5]), @sprintf("%.10f", t1gm[6]), @sprintf("%.10f", t1gm[7]), @sprintf("%.10f", t1gm[8])
                        ], ','))
                    end

                    # Optional: by utility matrix index
                    if WRITE_BY_UTILITY_MATRIX
                        open(out_byU, "a") do iou
                            # If file is empty, write header first
                            if filesize(out_byU) == 0
                                println(iou, join([
                                    "rule","utility","N","tw","utl_num","method","cases",
                                    "f1_wins_vs_EV","f1_losses_vs_EV","f1_ties_vs_EV","f1_mean_margin_vs_EV",
                                    "f1_wins_vs_GM","f1_losses_vs_GM","f1_ties_vs_GM","f1_mean_margin_vs_GM",
                                    "full_wins_vs_EV","full_losses_vs_EV","full_ties_vs_EV","full_mean_margin_vs_EV",
                                    "full_wins_vs_GM","full_losses_vs_GM","full_ties_vs_GM","full_mean_margin_vs_GM",
                                    "top1_wins_vs_EV","top1_losses_vs_EV","top1_ties_vs_EV","top1_mean_margin_vs_EV",
                                    "top1_wins_vs_GM","top1_losses_vs_GM","top1_ties_vs_GM","top1_mean_margin_vs_GM"
                                ], ','))
                            end

                            for utl_num in 1:UTILITY_MATRIX_NUM
                                casesU = repeat
                                for (m, _) in sort(collect(bundles); by=x->x[1])
                                    bu = get(byU, (m, utl_num), WLBundle())
                                    f1ev = wl_to_fields(bu.f1_vs_ev, casesU)
                                    f1gm = wl_to_fields(bu.f1_vs_gm, casesU)
                                    fev  = wl_to_fields(bu.full_vs_ev, casesU)
                                    fgm  = wl_to_fields(bu.full_vs_gm, casesU)
                                    t1ev = wl_to_fields(bu.top1_vs_ev, casesU)
                                    t1gm = wl_to_fields(bu.top1_vs_gm, casesU)

                                    println(iou, join([
                                        "minimax_regret", utility, string(N), tw, string(utl_num), m, string(casesU),
                                        string(f1ev[1]), string(f1ev[2]), string(f1ev[3]), @sprintf("%.10f", f1ev[4]),
                                        string(f1gm[1]), string(f1gm[2]), string(f1gm[3]), @sprintf("%.10f", f1gm[4]),
                                        string(fev[1]), string(fev[2]), string(fev[3]), @sprintf("%.10f", fev[4]),
                                        string(fgm[1]), string(fgm[2]), string(fgm[3]), @sprintf("%.10f", fgm[4]),
                                        string(t1ev[1]), string(t1ev[2]), string(t1ev[3]), @sprintf("%.10f", t1ev[4]),
                                        string(t1gm[1]), string(t1gm[2]), string(t1gm[3]), @sprintf("%.10f", t1gm[4])
                                    ], ','))
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @info "Saved" out_summary
    if WRITE_BY_UTILITY_MATRIX
        @info "Saved" out_byU
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
