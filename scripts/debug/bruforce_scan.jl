include(joinpath(@__DIR__, "..", "..", "src", "cp_regret_replace.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "file_operate.jl"))

using .SetRegretCore

function refresh_all_pairs!(matrix, wL, wU, t)
    A = size(matrix, 1)
    @inbounds for p in 1:A, q in 1:A
        p == q && continue
        SetRegretCore.set_linear_model_for_pair!(matrix[p, q], wL, wU, t)
    end
end

function main()
    utility_v = read_utility_value("u1")
    utility   = Matrix(utility_v[5])

    methodW = read_method_weights("A/MMRW", 1, 6)
    wL = methodW[1].L
    wU = methodW[1].R

    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

    matrix = SetRegretCore.create_minimax_R_Matrix(utility)

    Δ = (tR - tL) / 5000
    ts = collect(tR:-Δ:tL)

    N = size(matrix, 1)
    changes = Float64[]

    # 初期
    t0 = ts[1]
    refresh_all_pairs!(matrix, wL, wU, t0)
    qstar0 = [SetRegretCore.argmax_regret_index(matrix, p, t0) for p in 1:N]
    snap0 = SetRegretCore.snapshot_state(matrix, qstar0, t0)
    prev_rank = snap0.rank
    println(prev_rank)

    # 走査
    for k in 2:length(ts)
        t = ts[k]
        refresh_all_pairs!(matrix, wL, wU, t)

        qstar = [SetRegretCore.argmax_regret_index(matrix, p, t) for p in 1:N]
        snap = SetRegretCore.snapshot_state(matrix, qstar, t)
        rank = snap.rank

        if rank != prev_rank
            push!(changes, t)
            prev_rank = rank
        end
    end

    println("Bruteforce change points:")
    println(changes)
end

main()
