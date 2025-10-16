
# top1_envelope.jl
# ------------------------------------------------------------
# Julia extension: "Top-1 envelope" tracker for maximum-regret,
# working alongside your existing files:
#   - calc_IPW.jl
#   - module_regret.jl
#   - file_operate.jl  (for I/O if needed)
#
# This file DOES NOT modify your originals. It imports functions
# from RankChangeInterval and provides:
#   - find_rank_change_in_interval_top1  (interval-local)
#   - find_rank_change_points_top1_from_tR (full [tL,tR])
#   - plot_top1_envelope! (optional, requires Plots.jl)
#
# Drop this file next to your existing .jl files and do:
#   include("module_regret.jl")
#   include("top1_envelope.jl")
#   using .RankChangeInterval
#
# Then call find_rank_change_points_top1_from_tR(...).
# ------------------------------------------------------------

module Top1EnvelopeExt

using .RankChangeInterval: regret_value, linear_regret_parameters, compute_next_t, create_minimax_R_Matrix
using Statistics

export find_rank_change_in_interval_top1,
       find_rank_change_points_top1_from_tR,
       plot_top1_envelope!

# ------------------------------------------------------------
# Interval-local: track only the Top-1 (max) opponent for a fixed o_o
# over [t_lower, t_upper], assuming linearity in this interval.
#
# Returns a NamedTuple compatible with your downstream code:
#   (change_points, max_regret_alts, max_regret_values,
#    r_max_in_interval, r_min_in_interval)
# ------------------------------------------------------------
function find_rank_change_in_interval_top1(o_o::Int, t_lower::Float64, t_upper::Float64, matrix, methodW;
                                           eps::Float64=1e-10)
    @assert t_lower < t_upper "t_lower must be < t_upper"

    n = size(matrix, 1)

    # Precompute linear params for all opponents q ≠ o_o in this interval
    A = zeros(Float64, n)
    B = zeros(Float64, n)
    for q in 1:n
        if q == o_o; continue; end
        A[q], B[q] = linear_regret_parameters(o_o, q, t_upper, t_lower, matrix, methodW)
    end

    # Helper to evaluate regret vs q at t
    evalR = (q, t) -> (q == o_o ? -Inf : A[q]*t + B[q])

    # Left endpoint winner
    vals_left = [evalR(q, t_lower) for q in 1:n]
    p = argmax(vals_left)   # current top opponent index
    t = t_lower

    change_points = Float64[]
    max_regret_alts = Int[]
    max_regret_values = Float64[]

    # Track min/max regret over interval (for reporting)
    r_min = +Inf
    r_max = -Inf

    push!(max_regret_alts, p)
    push!(max_regret_values, evalR(p, t))

    while t < t_upper - eps
        # Candidates that can overtake p must have larger slope
        candidates = Int[]
        ap = A[p]
        for j in 1:n
            if j==o_o || j==p; continue; end
            if A[j] > ap + eps
                push!(candidates, j)
            end
        end

        # Find nearest intersection in (t, t_upper]
        τ = +Inf
        q_star = 0
        for j in candidates
            denom = (A[p] - A[j])
            if abs(denom) <= eps; continue; end
            t_star = (B[j] - B[p]) / denom
            if t_star > t + eps && t_star <= t_upper + eps
                if t_star < τ; τ = t_star; q_star = j; end
            end
        end

        if q_star == 0 || τ == +Inf
            # No change inside; record stats to the end and stop
            # Update r_min/r_max by checking endpoints (linear in between)
            r_end = evalR(p, t_upper)
            r_min = min(r_min, evalR(p, t), r_end)
            r_max = max(r_max, evalR(p, t), r_end)
            t = t_upper
            break
        else
            # From t..τ the winner is p; update stats and then switch to q_star at τ
            r_min = min(r_min, evalR(p, t), evalR(p, τ))
            r_max = max(r_max, evalR(p, t), evalR(p, τ))

            push!(change_points, τ)
            p = q_star
            t = τ
            push!(max_regret_alts, p)
            push!(max_regret_values, evalR(p, t))
        end
    end

    # If never updated r_min/r_max (degenerate), set by endpoints
    if !isfinite(r_min) || !isfinite(r_max)
        vL = evalR(p, t_lower)
        vU = evalR(p, t_upper)
        r_min = min(vL, vU)
        r_max = max(vL, vU)
    end

    return (
        change_points = change_points,
        max_regret_alts = max_regret_alts,
        max_regret_values = max_regret_values,
        r_max_in_interval = r_max,
        r_min_in_interval = r_min
    )
end

# ------------------------------------------------------------
# Full [t_L, t_R] with AvailSpace-based linear horizon.
# This mirrors RankChangeInterval.find_rank_change_points_from_tR
# but uses the Top-1 tracker per interval.
# ------------------------------------------------------------
function find_rank_change_points_top1_from_tR(o_o::Int, t_L::Float64, t_R::Float64, matrix, methodW;
                                              eps::Float64=1e-10)
    all_change_points = Float64[]
    all_max_regret_alts = Int[]
    all_max_regret_values = Float64[]
    intervals = Tuple{Float64,Float64}[]
    interval_data_collected = Any[]

    current_t = t_R
    while current_t > t_L + eps
        next_t, _ = compute_next_t(current_t, matrix, methodW)
        next_t = max(next_t, t_L)

        if isapprox(current_t, next_t; atol=1e-12)
            if current_t > t_L + eps
                push!(intervals, (t_L, current_t))
                res = find_rank_change_in_interval_top1(o_o, t_L, current_t, matrix, methodW; eps=eps)
                push!(interval_data_collected, res)
                append!(all_change_points, res.change_points)
                append!(all_max_regret_alts, res.max_regret_alts)
                append!(all_max_regret_values, res.max_regret_values)
            end
            break
        end

        push!(intervals, (next_t, current_t))
        res = find_rank_change_in_interval_top1(o_o, next_t, current_t, matrix, methodW; eps=eps)
        push!(interval_data_collected, res)
        append!(all_change_points, res.change_points)
        append!(all_max_regret_alts, res.max_regret_alts)
        append!(all_max_regret_values, res.max_regret_values)

        current_t = next_t
    end

    # Overall min/max in all intervals
    r_M_overall = -Inf
    r_m_overall = +Inf
    if !isempty(interval_data_collected)
        r_M_overall = maximum([res.r_max_in_interval for res in interval_data_collected])
        r_m_overall = minimum([res.r_min_in_interval for res in interval_data_collected])
    end

    return (
        change_points = all_change_points,
        max_regret_alts = all_max_regret_alts,
        max_regret_values = all_max_regret_values,
        intervals = intervals,
        interval_data = interval_data_collected,
        r_M = r_M_overall,
        r_m = r_m_overall
    )
end


# ------------------------------------------------------------
# Optional plotting (requires Plots.jl).
# Given o_o and a specific interval, draw the max-regret lines vs opponents
# and highlight the Top-1 envelope and change points.
# ------------------------------------------------------------
function plot_top1_envelope!(plt, o_o::Int, t_lower::Float64, t_upper::Float64, matrix, methodW;
                             points::Vector{Float64}=Float64[], eps::Float64=1e-10)
    @eval using Plots
    n = size(matrix, 1)

    # Linear params
    A = zeros(Float64, n); B = zeros(Float64, n)
    for q in 1:n
        if q == o_o; continue; end
        A[q], B[q] = linear_regret_parameters(o_o, q, t_upper, t_lower, matrix, methodW)
    end
    evalR = (q, t) -> (q == o_o ? NaN : A[q]*t + B[q])

    ts = range(t_lower, t_upper; length=400)
    # Plot each opponent line lightly
    for q in 1:n
        if q==o_o; continue; end
        ys = [evalR(q,t) for t in ts]
        plot!(plt, ts, ys, label="q=$(q)", alpha=0.3)
    end

    # Shade change points
    for x in points
        v = maximum([evalR(q,x) for q in 1:n if q!=o_o])
        scatter!(plt, [x], [v], label="switch at $(round(x,digits=4))")
    end

    return plt
end

end # module
