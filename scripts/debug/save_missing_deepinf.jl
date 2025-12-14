include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore

using CSV
using DataFrames
using Plots
using Plots: vline!

# -------------------------
# 1. 計算用関数 (bench_brute_vs_lps_u1_1to10.jl より)
# -------------------------
const BRUTE_STEPS = 5000
const repeat_num = 1
const counts_utility = 100

function brute_scan_changes(utility, wL, wU, tL, tR; steps=BRUTE_STEPS, eps=SetRegretCore.EPS_DEFAULT)
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

function lps_run_changes(utility, wL, wU, tL, tR; eps=SetRegretCore.EPS_DEFAULT)
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    res = SetRegretCore.run_lps(matrix, wL, wU, tL, tR; eps=eps)
    return Vector{Float64}(res.changes)
end

function match_changes(brute_ts::Vector{Float64}, lps_ts::Vector{Float64}, tol::Float64)
    b = sort(copy(brute_ts); rev=true)
    l = sort(copy(lps_ts); rev=true)
    pairs = Tuple{Float64,Float64}[]
    miss_b = Float64[] # LPSが見逃したもの (Missing in LPS)
    miss_l = Float64[] # LPSが余計に見つけたもの (Extra in LPS)

    i = 1
    j = 1
    while i <= length(b) && j <= length(l)
        tb = b[i]
        tl = l[j]
        if abs(tb - tl) <= tol
            push!(pairs, (tb, tl))
            i += 1
            j += 1
        elseif tb > tl + tol
            push!(miss_b, tb)
            i += 1
        else
            push!(miss_l, tl)
            j += 1
        end
    end
    while i <= length(b)
        push!(miss_b, b[i])
        i += 1
    end
    while j <= length(l)
        push!(miss_l, l[j])
        j += 1
    end
    return pairs, miss_b, miss_l
end

# -------------------------
# 2. 描画用関数 (凡例を外に出す修正版)
# -------------------------
function plot_regret_bruteforce(
    utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64},
    idx::Int; # タイトル用
    n::Int=400
)
    tL, tR = SetRegretCore.find_optimal_trange(L, R)
    ts = range(tR, tL; length=n)
    A = size(utility, 1)

    matrix_eval = SetRegretCore.create_minimax_R_Matrix(utility)

    vals = zeros(A, n)
    for (k, t) in enumerate(ts)
        SetRegretCore.initialize_linear_models!(matrix_eval, L, R, t)
        MR = SetRegretCore.max_regret_vector(matrix_eval, t)
        vals[:, k] .= MR
    end

    # 凡例を外に出す (:outertopright)
    p = plot(
        legend=:outertopright,
        xlabel="t",
        ylabel="MR_p(t)",
        title="u1_$(idx) : Missing Check",
        size=(900, 600),
        margin=5Plots.mm
    )

    for i in 1:A
        plot!(p, ts, vals[i, :], label="p=$i", lw=1.5)
    end
    return p
end

function overlay_change_points!(
    p::Plots.Plot,
    cps::Vector{Float64};
    color=:red,
    alpha=0.45,
    linestyle=:solid,
    label::String=""
)
    isempty(cps) && return p
    first = true
    for t in cps
        vline!(p, [t], lc=color, ls=linestyle, alpha=alpha, label=(first ? label : ""))
        first = false
    end
    return p
end

# -------------------------
# 3. メイン処理
# -------------------------
function main()
    paths = Paths.project_paths()

    # 保存先ディレクトリ作成
    outdir = joinpath(@__DIR__, "..", "..", "results", "tmp", "missings_png_$(repeat_num)")
    if !isdir(outdir)
        mkpath(outdir)
        println("Created directory: $outdir")
    else
        println("Output directory: $outdir")
    end

    # データ読み込み
    utility_v = LoadInstance.read_utility_value(paths, "u1")
    methodW = LoadInstance.read_method_weights(paths, "A/MMRW", repeat_num)
    wL = methodW[repeat_num].L
    wU = methodW[repeat_num].R

    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)
    Δ = (tR - tL) / BRUTE_STEPS
    tol = 20Δ
    eps = SetRegretCore.EPS_DEFAULT

    println("Start scanning u1 (1 to 100)...")

    count_saved = 0

    for idx in 1:counts_utility
        utility = Matrix(utility_v[idx])

        # 1. 計算
        brute_changes = brute_scan_changes(utility, wL, wU, tL, tR; steps=BRUTE_STEPS, eps=eps)
        lps_changes = lps_run_changes(utility, wL, wU, tL, tR; eps=eps)

        # 2. 比較
        pairs, miss_b, miss_l = match_changes(brute_changes, lps_changes, tol)

        # 3. LPSの見逃し (missing_in_LPS) がある場合のみプロット保存
        if !isempty(miss_b)
            x0 = miss_b[1]  # brute側にあってLPSに無い “最初のmissing” をfocusにする

            trace_dir = joinpath(@__DIR__, "..", "..", "results", "tmp", "missings_trace_$(repeat_num)")
            isdir(trace_dir) || mkpath(trace_dir)

            idx_str = lpad(idx, 3, "0")
            trace_csv = joinpath(trace_dir, "u1_$(idx_str)_trace.csv")
            lines_csv = joinpath(trace_dir, "u1_$(idx_str)_lines.csv")
            rej_csv = joinpath(trace_dir, "u1_$(idx_str)_reject.csv")

            matrix2 = SetRegretCore.create_minimax_R_Matrix(utility)
            SetRegretCore.run_lps(matrix2, wL, wU, tL, tR;
                eps=eps,
                trace_path=trace_csv,   # ループごとの概要（E1/E2/t_next/nS/順位など）
                lines_path=lines_csv,   # 各tでの qstar/hat の slope/intercept
                focus_x0=x0,            # missing交点の近傍だけ詳しく理由を取る
                reject_path=rej_csv,    # REJECT_LOWER/UPPER と “支配項(dom)” を出す
                near_tol=tol            # 近傍判定幅（あなたのtolでOK）
            )
            println("  [Found Missing] index=$idx, missing_count=$(length(miss_b))")

            # MR曲線を描画
            p = plot_regret_bruteforce(utility, wL, wU, idx)

            # 線を重ねる
            # Brute (正解) = 赤
            overlay_change_points!(p, brute_changes; color=:red, linestyle=:solid, alpha=0.5, label="Brute")
            # LPS (計算結果) = 青
            overlay_change_points!(p, lps_changes; color=:blue, linestyle=:dash, alpha=0.5, label="LPS")
            # Missing (見逃し箇所) = 緑の太線で強調
            overlay_change_points!(p, miss_b; color=:green, linestyle=:solid, alpha=0.8, label="MISSING")

            # 保存
            # 3桁ゼロ埋め (001, 002...)
            idx_str = lpad(idx, 3, "0")
            fname = joinpath(outdir, "u1_$(idx_str)_missing.png")
            savefig(p, fname)

            count_saved += 1
        end

        if idx % 10 == 0
            print(".")
        end
    end
    println("\nDone.")
    println("Total saved images: $count_saved")
end

main()