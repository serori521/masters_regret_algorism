# plot_regret_transition_rtr.jl
# RegretRTR を用いて、各代替案の最大リグレット推移を可視化
using Plots
include("RegretRTR.jl")

"""
plot_regret_transition_rtr(utility, L, R; n=200)

- utility: A×N 効用値行列
- L, R: 長さNの下端・上端
- 右→左（TR→TL）に沿って t をサンプリングし、各代替案の最大リグレットを描画
"""
function plot_regret_transition_rtr(utility::Matrix{Float64},
                                    L::Vector{Float64},
                                    R::Vector{Float64}; n::Int=200)
    st = RegretRTR.build_state(utility, L, R)
    tL, tR = st.tL, st.tR
    ts = collect(range(tR, tL; length=n))

    A = size(utility,1)
    vals = zeros(A, length(ts))

    for (k,t) in enumerate(ts)
        MR = RegretRTR.max_regret_vector_at(st.matrix, t)
        vals[:,k] .= MR
    end

    p = plot(xlabel="t", ylabel="Max Regret", title="Max Regret per Alternative",
             legend=:outerright, size=(900, 520), grid=true)
    colors = [:red, :blue, :green, :purple, :orange, :brown, :pink, :gray, :cyan, :magenta]

    for a in 1:A
        plot!(p, ts, vals[a,:],
              label="alt $a",
              color=colors[a > length(colors) ? mod1(a,length(colors)) : a],
              linewidth=2)
    end

    vline!(p, [tL, tR], label="t range", linestyle=:dash, color=:black, alpha=0.5)
    return p, st
end
