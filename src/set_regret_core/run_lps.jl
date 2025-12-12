###############################
# ログ用スナップショット
###############################
function snapshot_state(
    matrix::Array{minimax_regret_tuple,2},
    qstar::Vector{Int},
    t::Float64;
    eps::Float64=EPS_DEFAULT
)
    A = length(qstar)
    MR = Vector{Float64}(undef, A)
    @inbounds for p in 1:A
        q = qstar[p]
        MR[p] = q == 0 ? -Inf : evaluate_regret(matrix[p, q], t)
    end
    winners = findall(x -> x <= minimum(MR) + eps, MR)
    return (t=t, MR=MR, rank=ranking_from_MR(copy(MR)), winners=Vector{Int}(winners))
end

function push_snapshot!(changes::Vector{Float64}, timeline::Vector{SnapshotEntry},
    matrix::Array{minimax_regret_tuple,2}, qstar::Vector{Int},
    t::Float64; eps::Float64=EPS_DEFAULT, detect_change::Bool=false)
    snap = snapshot_state(matrix, qstar, t; eps=eps)
    if detect_change && !isempty(timeline)
        prev = timeline[end]
        if prev.rank != snap.rank
            duplicated = any(abs(tc - t) <= eps for tc in changes)
            duplicated || push!(changes, t)
        end
    end
    push!(timeline, snap)
end

function refresh_inner_state!(
    matrix::Array{minimax_regret_tuple,2},
    p::Int,
    t::Float64,
    t_L::Float64,
    qstar::Vector{Int},
    hat_q::Vector{Int},
    x_p_max::Vector{Float64};
    eps::Float64=EPS_DEFAULT,
    preferred::Int=0
)
    preferred = preferred == 0 ? qstar[p] : preferred
    best = argmax_regret_index(matrix, p, t; preferred=preferred, eps=eps)
    qstar[p] = best
    challenger, xmax = find_inner_crossing(matrix, p, best, t_L, t; eps=eps)
    hat_q[p] = challenger
    x_p_max[p] = challenger == 0 ? t_L : xmax
end


###############################
# 8. KDSベース左向き走査メインループ
###############################
function run_lps(
    matrix::Array{minimax_regret_tuple,2},
    wL::Vector{Float64}, wU::Vector{Float64},
    t_L::Float64, t_U::Float64;
    eps::Float64=EPS_DEFAULT
)
    initialize_linear_models!(matrix, wL, wU, t_U; eps=eps)

    A = size(matrix, 1)
    qstar = zeros(Int, A)
    hat_q = zeros(Int, A)
    x_p_max = fill(t_L, A)

    @inbounds for p in 1:A
        qstar[p] = argmax_regret_index(matrix, p, t_U; eps=eps)
        h, xp = find_inner_crossing(matrix, p, qstar[p], t_L, t_U; eps=eps)
        hat_q[p] = h
        x_p_max[p] = h == 0 ? t_L : xp
    end

    Tchg = Float64[]
    timeline = SnapshotEntry[]
    push_snapshot!(Tchg, timeline, matrix, qstar, t_U; eps=eps, detect_change=false)

    t = t_U
    while t > t_L + eps
        E1, active_pairs = next_coefficient_event(matrix, t_L, t; eps=eps)
        E2, inner_indices = next_inner_event(x_p_max, t_L, t; eps=eps)
        t_next = max(max(E1, E2), t_L)

        if t_next >= t - eps
            break
        end

        S = collect_outer_changes(matrix, qstar, x_p_max, t_next, t; eps=eps)
        for x in S
            push!(Tchg, x)
            push_snapshot!(Tchg, timeline, matrix, qstar, x; eps=eps, detect_change=false)
        end

        affected = Int[]

        if !isempty(active_pairs) && abs(t_next - E1) <= eps
            for (p, q) in active_pairs
                set_linear_model_for_pair!(matrix[p, q], wL, wU, t_next; eps=eps)
                push!(affected, p)
            end
        end

        if !isempty(inner_indices) && abs(t_next - E2) <= eps
            for p in inner_indices
                if hat_q[p] != 0
                    qstar[p] = hat_q[p]
                end
                push!(affected, p)
            end
        end

        t = t_next

        isempty(affected) && break

        unique!(affected)
        for p in affected
            refresh_inner_state!(matrix, p, t, t_L, qstar, hat_q, x_p_max;
                eps=eps, preferred=qstar[p])
        end

        push_snapshot!(Tchg, timeline, matrix, qstar, t; eps=eps, detect_change=true)
    end

    return (changes=Tchg, timeline=timeline)
end
