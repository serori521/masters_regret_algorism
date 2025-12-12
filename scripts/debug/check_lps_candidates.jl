include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "cp_regret_replace.jl"))  # SetRegretCore

using .Paths
using .LoadInstance
using .SetRegretCore

function main()
    paths = Paths.project_paths()

    utility_v = LoadInstance.read_utility_value(paths, "u1")
    utility   = Matrix(utility_v[5])

    methodW = LoadInstance.read_method_weights(paths, "A/MMRW", 1, 6)
    wL = methodW[1].L
    wU = methodW[1].R

    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, tR)

    A = size(matrix, 1)
    qstar = zeros(Int, A)
    hat_q = zeros(Int, A)
    x_p_max = fill(tL, A)

    # --- 初期の qstar/hat_q/x_p_max を作る（run_lps冒頭と同じ） ---
    @inbounds for p in 1:A
        qstar[p] = SetRegretCore.argmax_regret_index(matrix, p, tR)
        h, xp = SetRegretCore.find_inner_crossing(matrix, p, qstar[p], tL, tR)
        hat_q[p] = h
        x_p_max[p] = h == 0 ? tL : xp
    end

    # --- ここが重要：E1/E2候補が出てるか ---
    E1, active_pairs = SetRegretCore.next_coefficient_event(matrix, tL, tR)
    E2, inner_indices = SetRegretCore.next_inner_event(x_p_max, tL, tR)

    println("tL=$tL, tR=$tR")
    println("init: qstar = ", qstar)
    println("init: hat_q = ", hat_q)
    println("init: x_p_max = ", x_p_max)

    println("E1(best) = $E1, #pairs = $(length(active_pairs))")
    if !isempty(active_pairs)
        println("  sample pairs: ", active_pairs[1:min(end,5)])
    end

    println("E2(best) = $E2, #idxs = $(length(inner_indices))")
    if !isempty(inner_indices)
        println("  idxs = ", inner_indices)
    end

    # 追加：tstarが有効なペア数をざっくり
    cnt_valid = 0
    tstar_max = -Inf
    @inbounds for i in 1:A, j in 1:A
        i == j && continue
        ts = matrix[i,j].tstar
        if (tL < ts < tR)
            cnt_valid += 1
            tstar_max = max(tstar_max, ts)
        end
    end
    println("#(tstar in (tL,tR)) = $cnt_valid, max tstar = $tstar_max")
end

main()
