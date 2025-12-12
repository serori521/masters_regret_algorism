include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "SetRegretCore.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore


function refresh_all_pairs!(matrix, wL, wU, t)
    A = size(matrix, 1)
    @inbounds for p in 1:A, q in 1:A
        p == q && continue
        SetRegretCore.set_linear_model_for_pair!(matrix[p, q], wL, wU, t)
    end
end

function main()
    paths = Paths.project_paths()

    utility_v = LoadInstance.read_utility_value(paths, "u1")   # ← pathsを渡す
    utility = Matrix(utility_v[5])

    methodW = LoadInstance.read_method_weights(paths, "A/MMRW", 1, 6)  # ←ここもpaths
    wL = methodW[1].L
    wU = methodW[1].R


    # ---------- t範囲 ----------
    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

    # ---------- モデル初期化 ----------
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, tR)

    # ---------- brute scan ----------
    steps = 5000
    Δ = (tR - tL) / steps
    ts = collect(tR:-Δ:tL)

    A = size(matrix, 1)
    changes = Float64[]
    ranks_at_changes = Vector{Vector{Int}}()

    # 初期
    t0 = ts[1]
    qstar0 = [SetRegretCore.argmax_regret_index(matrix, p, t0) for p in 1:A]
    snap0 = SetRegretCore.snapshot_state(matrix, qstar0, t0)
    prev_rank = snap0.rank

    # ログ出力先
    outdir = joinpath(@__DIR__, "..", "..", "results", "tmp")
    isdir(outdir) || mkpath(outdir)
    out_csv = joinpath(outdir, "brute_changes.csv")

    # ヘッダ
    open(out_csv, "w") do io
        println(io, "k,t,rank")  # rankは "3|1|2|..." 形式で保存
    end

    # 走査
    for k in 2:length(ts)
        t = ts[k]

        # ★超重要：tごとに線形モデルを更新（bruteの正しさのため）
        @inbounds for p in 1:A, q in 1:A
            p == q && continue
            SetRegretCore.set_linear_model_for_pair!(matrix[p, q], wL, wU, t)
        end

        qstar = [SetRegretCore.argmax_regret_index(matrix, p, t) for p in 1:A]
        snap = SetRegretCore.snapshot_state(matrix, qstar, t)
        rank = snap.rank

        if rank != prev_rank
            push!(changes, t)
            push!(ranks_at_changes, rank)
            prev_rank = rank

            # 1行追記
            open(out_csv, "a") do io
                println(io, "$(k),$(t),$(join(rank, '|'))")
            end
        end
    end

    println("tL=$tL, tR=$tR, Δ=$Δ")
    println("Bruteforce change points ($(length(changes))):")
    println(changes)
    println("Saved: $out_csv")
end

main()