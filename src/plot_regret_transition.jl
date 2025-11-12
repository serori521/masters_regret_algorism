# plot_regret_transition.jl
# 1ファイル統合：ブルートフォース描画 + イベント（change points）オーバレイ
include("set_regret.jl")
using .SetRegretCore
include("tracker_rtr.jl")
using .RegretRTRTracker

using Plots
using Plots: vline!, ylims

"ブルートフォース: 右→左で等間隔サンプルし、各tで一次モデルを再構成してMRを描画"
function plot_regret_bruteforce(utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64}; n::Int=400)
    tL, tR = SetRegretCore.find_optimal_trange(L, R)
    ts = range(tR, tL; length=n)
    A = size(utility, 1)

    # 評価用の独立コピー
    matrix_eval = SetRegretCore.create_minimax_R_Matrix(utility)

    vals = zeros(A, n)
    for (k, t) in enumerate(ts)
        # 各 (p,q) の一次モデルを「そのtで」再構成（= ブルートフォース）
        SetRegretCore.initialize_linear_models!(matrix_eval, L, R, t)
        MR = SetRegretCore.max_regret_vector(matrix_eval)
        vals[:, k] .= MR
    end

    p = plot(legend=:topright, xlabel="t", ylabel="MR_p(t)", title="MR (bruteforce)")
    for i in 1:A
        plot!(p, ts, vals[i, :], label="p=$i")
    end
    return p, ts, vals
end

"イベントオーバレイ: 検出した change points を縦線で重ねる"
function overlay_change_points!(p::Plots.Plot, cps::Vector{Float64}; color=:red, alpha=0.4)
    isempty(cps) && return p
    ymin, ymax = ylims(p)
    for t in cps
        vline!(p, [t], label="", lc=color, alpha=alpha)
    end
    return p
end

"""
メイン関数：検算プロット + 変化点の縦線
戻り値: (plot, change_points)
"""
function plot_regret_transition(utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64}; n::Int=400)
    p, _, _ = plot_regret_bruteforce(utility, L, R; n=n)
    cps = RegretRTRTracker.find_change_points(utility, L, R)
    overlay_change_points!(p, cps; color=:red, alpha=0.45)
    display(p)
    return p, cps
end
