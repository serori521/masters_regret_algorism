###############################
# イベント検知（疑う層）
###############################
function collect_outer_changes(
    matrix::Array{minimax_regret_tuple,2},
    qstar::Vector{Int},
    x_p_max::Vector{Float64},
    t_min::Float64, t_max::Float64;
    eps::Float64=EPS_DEFAULT
)
    A = length(qstar)
    changes = Float64[]

    @inbounds for p1 in 1:A-1
        for p2 in p1+1:A
            q1 = qstar[p1]
            q2 = qstar[p2]
            (q1 == 0 || q2 == 0) && continue

            line1 = matrix[p1, q1]
            line2 = matrix[p2, q2]
            Adelta = line1.slope - line2.slope
            abs(Adelta) <= eps && continue

            x = (line2.intercept - line1.intercept) / Adelta
            lower = maximum((t_min, line1.tstar, line2.tstar, x_p_max[p1], x_p_max[p2]))
            if lower <= x && x <= t_max - eps
                push!(changes, x)
            end
        end
    end

    sort!(changes; rev=true)
    return changes
end

function next_coefficient_event(
    matrix::Array{minimax_regret_tuple,2},
    t_L::Float64, t_cur::Float64;
    eps::Float64=EPS_DEFAULT
)
    best = t_L
    pairs = Tuple{Int,Int}[]
    A = size(matrix, 1)

    @inbounds for i in 1:A, j in 1:A
        i == j && continue
        tstar = matrix[i, j].tstar
        if !(t_L + eps < tstar < t_cur - eps)
            continue
        end
        if tstar > best + eps
            best = tstar
            empty!(pairs)
            push!(pairs, (i, j))
        elseif abs(tstar - best) <= eps
            push!(pairs, (i, j))
        end
    end

    return best, pairs
end

function next_inner_event(
    x_p_max::Vector{Float64},
    t_L::Float64, t_cur::Float64;
    eps::Float64=EPS_DEFAULT
)
    best = t_L
    idxs = Int[]
    @inbounds for (p, x) in enumerate(x_p_max)
        if !(t_L + eps < x < t_cur - eps)
            continue
        end
        if x > best + eps
            best = x
            empty!(idxs)
            push!(idxs, p)
        elseif abs(x - best) <= eps
            push!(idxs, p)
        end
    end
    return best, idxs
end
