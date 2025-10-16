# plot_regret_transition_rtr.jl (ブルートフォース版・修正済み)
# 可視化: 各tで正しい一次モデルを「再計算」して最大リグレットを描く
include("RegretRTR.jl")
using .RegretRTR
using Plots

"""
plot_regret_bruteforce(utility, L, R; n=200)

- 右→左（TR→TL）に沿って t を等linspaceで取り、
- 各 t 地点で、全ペアの一次モデルを RegretRTR.set_linear_model_for_pair! で再計算してから評価する。
"""
function plot_regret_bruteforce(utility::Matrix{Float64},
    L::Vector{Float64},
    R::Vector{Float64}; n::Int=200)

    # 初期状態はtの範囲取得と、評価用行列のテンプレートとして使用
    st = RegretRTR.build_state(utility, L, R)
    tL, tR = st.tL, st.tR

    # 右→左グリッド
    ts = collect(range(tR, tL; length=n))

    A = size(utility, 1)
    vals = zeros(A, n)

    # 評価用の行列を準備
    matrix_for_eval = st.matrix

    # 各 t について、毎回モデルを再計算して評価
    for (k, t) in enumerate(ts)
        # 全ての(p,q)ペアについて、現在のtにおける一次モデルを強制的に再設定
        for i in 1:A, j in 1:A
            if i == j
                continue
            end
            # ★修正点1: RegretRTRモジュール内の関数を明示的に呼び出す
            RegretRTR.set_linear_model_for_pair!(matrix_for_eval[i, j], L, R, t)
        end

        # ★修正点2: 同様に、RegretRTRモジュール内の関数を呼び出す
        MR = RegretRTR.max_regret_vector(matrix_for_eval)
        vals[:, k] .= MR
    end

    p = plot(xlabel="t", ylabel="Max Regret", title="Max Regret per Alternative (Brute-force)",
        legend=:outerright, size=(900, 520), grid=true)
    colors = [:red, :blue, :green, :purple, :orange, :brown, :pink, :gray, :cyan, :magenta]
    for a in 1:A
        plot!(p, ts, vals[a, :],
            label="alt $a",
            color=colors[a > length(colors) ? mod1(a, length(colors)) : a],
            linewidth=2)
    end
    vline!(p, [tL, tR], label="t range", linestyle=:dash, color=:black, alpha=0.5)

    return p
end