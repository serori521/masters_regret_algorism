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

# =================================================================
# Step 1: 関数ハンドルの準備 (直列実行)
# =================================================================
println("Loading modules and functions...")

# 計算に必要な情報をまとめる構造体的なリストを作る
tasks = []

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
    push!(tasks, (name=target_name, func=func_handle, method=method_type))
end
# =================================================================
# Step 1: 関数ハンドルの準備 (直列実行)
# =================================================================
println("Loading modules and functions...")

# 計算に必要な情報をまとめる構造体的なリストを作る
tasks = []

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
    push!(tasks, (name=target_name, func=func_handle, method=method_type))
end

println("Preparation complete. Starting parallel simulation with $(Threads.nthreads()) threads...")

# =================================================================
# Step 2: シミュレーション実行 (並列実行)
# =================================================================
# @threads は配列に対して有効なので、tasks配列を回す

@threads for task in tasks
    target_name = task.name
    target_func = task.func
    method_type = task.method

    println("Processing: $target_name on thread $(Threads.threadid()) ...")

    # ここから下の計算ロジックは以前と同じ（ただし変数はtaskから取得したものを使う）
    for N in num_criteria
        for setting in true_importance

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
                try
                    if method_type === nothing
                        solution = target_func(A)
                    else
                        solution = target_func(A, method_type)
                    end
                catch e
                    # 並列処理中のエラー表示
                    println("Error in $target_name N = $N ,setting = $setting (Thread $(Threads.threadid())): $e")
                    # 並列ループを壊さないためにcontinueなどが良いが、ここではログだけ
                end

                if solution !== nothing
                    E = solution.W
                    E_data = Vector{Float64}()
                    for j in 1:N
                        push!(E_data, inf(E[j]))
                        push!(E_data, sup(E[j]))
                    end
                    sum_width = sum(diam.(E))
                    push!(Simp, (i, E_data..., sum_width))
                end
            end

            # ファイル書き出し (パスが手法ごとに異なるので競合しない)
            output_folder_name = target_name
            root_folder = "/workspaces/inulab_julia_devcontainer/data/Simp/N=" * string(N) * "/a3/" * setting
            output_path = root_folder * "/" * output_folder_name

            if !isdir(output_path)
                mkpath(output_path)
            end
            CSV.write(output_path * "/Simp.csv", Simp)
        end
    end
    println("Finished: $target_name")
end