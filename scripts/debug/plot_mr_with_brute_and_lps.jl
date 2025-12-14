include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cp_regret_replace.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))  # 読み込みに使う場合
# include(joinpath(@__DIR__, "..", "..", "src", "file_operate.jl")) # もし load_instance が合わないならこっち
using .Paths
using .SetRegretCore
using .LoadInstance

using CSV
using DataFrames
using Plots
using Plots: vline!, ylims

# -------------------------
# 1) brute: 各tで一次モデル再構成してMR曲線を描く
# -------------------------
function plot_regret_bruteforce(
    utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64};
    n::Int=400
)
    tL, tR = SetRegretCore.find_optimal_trange(L, R)
    ts = range(tR, tL; length=n)
    A = size(utility, 1)

    matrix_eval = SetRegretCore.create_minimax_R_Matrix(utility)

    vals = zeros(A, n)
    for (k, t) in enumerate(ts)
        # 各tで(A,B,t*)を作り直す（brute）
        SetRegretCore.initialize_linear_models!(matrix_eval, L, R, t)
        MR = SetRegretCore.max_regret_vector(matrix_eval, t)
        vals[:, k] .= MR
    end

    p = plot(legend=:topright, xlabel="t", ylabel="MR_p(t)",
             title="MR (bruteforce) + change points")
    for i in 1:A
        plot!(p, ts, vals[i, :], label="p=$i")
    end
    return p, ts, vals, tL, tR
end

# -------------------------
# 2) 縦線オーバレイ（凡例が増えすぎないよう最初の1本だけラベル）
# -------------------------
function overlay_change_points!(
    p::Plots.Plot,
    cps::Vector{Float64};
    color=:red,
    alpha=0.45,
    label::String=""
)
    isempty(cps) && return p
    first = true
    for t in cps
        vline!(p, [t], lc=color, alpha=alpha, label=(first ? label : ""))
        first = false
    end
    return p
end

# -------------------------
# 3) CSVからLPS changeを読む
#    lps_changes.csv の列名が t じゃない場合もあるので吸収
# -------------------------
function read_lps_changes_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    if :t ∈ names(df)
        return Vector{Float64}(df.t)
    else
        # 先頭列をtとして読む（列名が違っても動く）
        col = names(df)[1]
        return Vector{Float64}(df[!, col])
    end
end

# -------------------------
# 4) メイン（今回：bruteと同じ問題設定）
# -------------------------
function main(; n::Int=500)
    # ★ここは「compare_brute_vs_lps.jl と同じ設定」に合わせる
    # load_instance の関数シグネチャが環境で違うことがあるので、まずは以下を試す：
    paths = Paths.project_paths()

    utility_v = LoadInstance.read_utility_value(paths, "u1")
    utility   = Matrix(utility_v[5])

    methodW = LoadInstance.read_method_weights(paths, "A/MMRW", 1, 6)
    wL = methodW[1].L
    wU = methodW[1].R

    # bruteの変化点（あなたが出したリストをそのまま使う：確実）
    brute_changes = Float64[
        1.151459029922891,
        1.1144845286710632,
        1.0458855343871256,
        0.9790698761764272,
        0.8938264118755896,
        0.864817478417564
    ]

    # LPSの変化点（前に compare が吐いたCSV）
    lps_csv = joinpath("results", "tmp", "lps_changes.csv")
    lps_changes = read_lps_changes_csv(lps_csv)

    # MR曲線
    p, ts, vals, tL, tR = plot_regret_bruteforce(utility, wL, wU; n=n)

    # 縦線：brute=赤, LPS=青
    overlay_change_points!(p, brute_changes; color=:red,  alpha=0.55, label="brute chg")
    overlay_change_points!(p, lps_changes;   color=:blue, alpha=0.35, label="LPS chg")

    # 保存
    out_png = joinpath("results", "tmp", "mr_brute_with_lps_changes.png")
    savefig(p, out_png)
    println("Saved: ", out_png)

    display(p)
    return nothing
end

main()
