using DataFrames, CSV
using IntervalArithmetic
using LaTeXStrings
using Statistics
using DataStructures
using JuMP
import HiGHS
using Base.Threads # スレッド機能を使うために必要

# --- 共通ライブラリの読み込み ---
# これらはグローバルスコープで一度読み込んでおきます
include("../new_libs/display-latex.jl")
include("../new_libs/crisp-pcm.jl")
include("../new_libs/analysis-indicators.jl")
include("../new_libs/solve-deterministic-ahp.jl")

# =================================================================
# 設定部分: 手法と使用するファイル・関数のマッピング
# =================================================================
# Key: 出力フォルダ名 (手法名)
# Value: (ファイルパス, 関数名のシンボル, method引数(EV/GM/nothing))
methods_map = OrderedDict(
    # --- Original Methods (引数はPCMのみ) ---
    "AMRwc" => ("../new_libs/oAMRwc.jl", :AMRwc, nothing),
    "MMRwc" => ("../new_libs/oMMRwc.jl", :MMRwc, nothing),

    # --- Extended Methods (引数はPCM, method) ---
    "eAMRd" => ("../new_libs/xAMRd.jl", :xAMRd, EV),
    "gAMRd" => ("../new_libs/xAMRd.jl", :xAMRd, GM), 
    "eAMRdc" => ("../new_libs/xAMRdc2.jl", :xAMRdc2, EV),
    "gAMRdc" => ("../new_libs/xAMRdc2.jl", :xAMRdc2, GM), 
    "eMMRd" => ("../new_libs/xMMRd.jl", :xMMRd, EV),
    "gMMRd" => ("../new_libs/xMMRd.jl", :xMMRd, GM), 
    "eMMRdc" => ("../new_libs/xMMRdc2.jl", :xMMRdc2, EV),
    "gMMRdc" => ("../new_libs/xMMRdc2.jl", :xMMRdc2, GM),

    # 必要であれば以下もコメントアウトを外して実行できます
    "eAMRw" => ("../new_libs/xAMRw.jl", :xAMRw, EV),
    "gAMRw" => ("../new_libs/xAMRw.jl", :xAMRw, GM), 
    "eAMRwc" => ("../new_libs/xAMRwc2.jl", :xAMRwc2, EV),
    "gAMRwc" => ("../new_libs/xAMRwc2.jl", :xAMRwc2, GM), 
    "eMMRw" => ("../new_libs/xMMRw.jl", :xMMRw, EV),
    "gMMRw" => ("../new_libs/xMMRw.jl", :xMMRw, GM), 
    "eMMRwc" => ("../new_libs/xMMRwc2.jl", :xMMRwc2, EV),
    "gMMRwc" => ("../new_libs/xMMRwc2.jl", :xMMRwc2, GM),
)

# シミュレーション設定
num_criteria = [4, 5, 6, 7, 8]
true_importance = ["A", "B", "C", "D", "E"]
wanted = Set{Tuple{Int,String,String}}([
    (4,"A","eAMRwc"),
    (4,"A","eMMRwc"),
    (4,"A","gAMRwc"),
    (4,"A","gMMRwc"),
    (4,"C","eAMRwc"),
    (4,"C","eMMRwc"),
    (4,"C","gAMRwc"),
    (4,"C","gMMRwc"),
    (4,"D","eAMRwc"),
    (4,"D","eMMRwc"),
    (4,"D","gAMRwc"),
    (4,"D","gMMRwc"),
    (5,"A","eAMRwc"),
    (5,"A","eMMRwc"),
    (5,"A","gAMRwc"),
    (5,"A","gMMRwc"),
    (5,"B","eAMRdc"),
    (5,"B","eAMRwc"),
    (5,"B","eMMRdc"),
    (5,"B","eMMRwc"),
    (5,"B","gAMRdc"),
    (5,"B","gMMRdc"),
    (5,"C","eAMRwc"),
    (5,"C","eMMRwc"),
    (5,"C","gAMRwc"),
    (5,"C","gMMRwc"),
    (5,"D","eAMRwc"),
    (5,"E","eAMRwc"),
    (5,"E","eMMRwc"),
    (5,"E","gAMRwc"),
    (5,"E","gMMRwc"),
    (6,"A","eAMRwc"),
    (6,"A","eMMRwc"),
    (6,"A","gAMRwc"),
    (6,"A","gMMRwc"),
    (6,"B","eAMRdc"),
    (6,"B","eAMRwc"),
    (6,"B","eMMRdc"),
    (6,"B","eMMRwc"),
    (6,"B","gAMRdc"),
    (6,"B","gAMRwc"),
    (6,"B","gMMRdc"),
    (6,"B","gMMRwc"),
    (6,"C","eAMRwc"),
    (6,"C","eMMRwc"),
    (6,"C","gAMRwc"),
    (6,"C","gMMRwc"),
    (6,"D","eAMRwc"),
    (6,"D","eMMRwc"),
    (6,"D","gAMRdc"),
    (6,"D","gAMRwc"),
    (6,"D","gMMRdc"),
    (6,"D","gMMRwc"),
    (6,"E","eAMRwc"),
    (6,"E","eMMRwc"),
    (6,"E","gAMRwc"),
    (6,"E","gMMRwc"),
    (7,"A","eAMRwc"),
    (7,"A","eMMRwc"),
    (7,"A","gAMRwc"),
    (7,"A","gMMRwc"),
    (7,"B","eAMRwc"),
    (7,"B","eMMRwc"),
    (7,"B","gAMRwc"),
    (7,"B","gMMRwc"),
    (7,"C","eAMRwc"),
    (7,"C","eMMRwc"),
    (7,"D","gAMRwc"),
    (7,"D","gMMRwc"),
    (7,"E","gAMRwc"),
    (7,"E","gMMRwc"),
    (8,"A","eAMRwc"),
    (8,"A","eMMRwc"),
    (8,"A","gAMRwc"),
    (8,"A","gMMRwc"),
    (8,"C","eAMRwc"),
    (8,"C","eMMRwc"),
    (8,"C","gAMRwc"),
    (8,"C","gMMRwc"),
    (8,"E","gAMRwc"),
    (8,"E","gMMRwc"),
])
# =================================================================
# Step 1: 関数ハンドルの準備 (直列実行)
# =================================================================
println("Loading modules and functions...")

# 計算に必要な情報をまとめる構造体的なリストを作る
method_tasks = []

for (target_name, (file_path, func_sym, method_type)) in methods_map
    # モジュールの作成と読み込み（ここは安全のため直列でやる）
    mod_name = Symbol("Mod_" * target_name)
    @eval module $mod_name
    using IntervalArithmetic
    using IntervalArithmetic.Symbols
    using JuMP
    import HiGHS
    using Statistics

    # 共通ライブラリの再読み込みは不要（コメントアウト済み前提）
    # include("./libs/crisp-pcm.jl")

    include($file_path)
    end

    # 関数オブジェクトを取得
    func_handle = getfield(@eval($mod_name), func_sym)

    # 並列処理用にタスクリストに追加
    push!(method_tasks, (name=target_name, func=func_handle, method=method_type))
end

# =================================================================
# Step 1.5: 実行タスクの構築 (method × N × true_importance を事前に分割)
#   - 上の wanted に含まれる組み合わせだけを tasks に積む
# =================================================================
tasks = []
for mt in method_tasks
    mname = mt.name
    for N in num_criteria
        for tw in true_importance
            if (N, tw, mname) ∈ wanted
                push!(tasks, (name=mname, func=mt.func, method=mt.method, N=N, setting=tw))
            end
        end
    end
end

println("Filtered tasks: $(length(tasks)) (from wanted=$(length(wanted)))")

println("Preparation complete. Starting parallel simulation with $(Threads.nthreads()) threads...")

# =================================================================
# Step 2: シミュレーション実行 (並列実行)
# =================================================================
# @threads は配列に対して有効なので、tasks配列を回す

for task in tasks
    target_name = task.name
    target_func = task.func
    method_type = task.method
    N = task.N
    setting = task.setting

    println("Processing: $target_name N=$N tw=$setting on thread $(Threads.threadid()) ...")


            Simp_columns = OrderedDict()
            push!(Simp_columns, "Num" => Int[])
            for i in 1:N
                push!(Simp_columns, "wL[$(i-1)]" => Float64[])
                push!(Simp_columns, "wR[$(i-1)]" => Float64[])
            end
            push!(Simp_columns, "Sum_of_Width" => Float64[])
            # スレッドセーフにするため、DataFrame構築はループ内で行う（問題なし）
            Simp = DataFrame(Simp_columns)

            path_true = "/workspaces/inulab_julia_devcontainer/data/PCM_set/N=" * string(N) * "/a3/" * setting * "/Given_interval_weight.csv"
            if !isfile(path_true)
                continue
            end
            df_true = CSV.File(path_true, header=false) |> DataFrame
            # T = ... (Tは使っていないようであれば省略可)

            path_pcm = "/workspaces/inulab_julia_devcontainer/data/PCM_set/N=" * string(N) * "/a3/" * setting * "/PCM_int.csv"
            if !isfile(path_pcm)
                continue
            end
            df_pcm = CSV.File(path_pcm, header=false) |> DataFrame
            subdfs = split_dataframe(df_pcm, N)

            for (i, subdf) in enumerate(subdfs)
                A = Matrix(subdf)
                solution = nothing

                if method_type === nothing
                    solution = target_func(A)
                else
                    solution = target_func(A, method_type)
                end
                
                E = solution.W
                E_data = Vector{Float64}()
                for j in 1:N
                    push!(E_data, inf(E[j]))
                    push!(E_data, sup(E[j]))
                end
                sum_width = sum(diam.(E))
                push!(Simp, (i, E_data..., sum_width))
                
            end

            # ファイル書き出し (パスが手法ごとに異なるので競合しない)
            output_folder_name = target_name
            root_folder = "/workspaces/inulab_julia_devcontainer/data/Simp/N=" * string(N) * "/a3/" * setting
            output_path = root_folder * "/" * output_folder_name

            if !isdir(output_path)
                mkpath(output_path)
            end
            CSV.write(output_path * "/Simp.csv", Simp)

    println("Finished: $target_name N=$N tw=$setting")
end