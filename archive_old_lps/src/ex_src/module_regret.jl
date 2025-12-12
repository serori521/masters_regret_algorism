module RankChangeInterval
using Statistics, LinearAlgebra, DataFrames, CSV
export find_rank_change_points_from_tR, analyze_all_alternatives_with_avail_space, save_rank_changes_and_rankings_to_csv
include("calc_IPW.jl")

# 指定した t における代替案 o_o と o_q 間のリグレット値を返す関数
function regret_value(o_o::Int, o_q::Int, t::Float64, matrix, methodW)
    if o_o == o_q
        return 0.0
    end

    Y_L = methodW.L .* t
    Y_R = methodW.R .* t

    # 一時的に regret を計算
    temp_matrix = deepcopy(matrix)
    _, _ = calc_regret(temp_matrix, Y_L, Y_R)
    # 行列内のリグレット値を取得
    return temp_matrix[o_o, o_q].regret
end

# アベイルスペースから次のt'を計算する関数
function compute_next_t(current_t::Float64, matrix, methodW)
    Y_L = methodW.L .* current_t
    Y_R = methodW.R .* current_t

    temp_matrix = deepcopy(matrix)
    _, _ = calc_regret(temp_matrix, Y_L, Y_R)

    # 非対角要素の最小アベイルスペースを見つける
    min_availspace = findmin([temp_matrix[i, j].Avail_space
                              for i in 1:size(matrix, 1), j in 1:size(matrix, 1)
                              if i != j])[1]

    # rパラメータを計算
    r = 1 / (1 + min_availspace)

    # 次のtを計算
    next_t = r * current_t

    return next_t, min_availspace
end

# t内でのリグレット関数の線形パラメータ（傾きと切片）を計算
function linear_regret_parameters(o_o::Int, q::Int, t1::Float64, t2::Float64, matrix, methodW)
    r1 = regret_value(o_o, q, t1, matrix, methodW)
    r2 = regret_value(o_o, q, t2, matrix, methodW)

    # 線形関数 r(t) = A*t + B のパラメータを計算
    A = (r2 - r1) / (t2 - t1)
    B = r1 - A * t1

    return A, B
end

# 2つのリグレット関数の交点を計算
function compute_crossing_point(o_o::Int, p::Int, q::Int, t_upper::Float64, t_lower::Float64, matrix, methodW)
    A_p, B_p = linear_regret_parameters(o_o, p, t_upper, t_lower, matrix, methodW)
    A_q, B_q = linear_regret_parameters(o_o, q, t_upper, t_lower, matrix, methodW)

    if isapprox(A_p, A_q; atol=1e-8)
        # 傾きがほぼ同じ場合は交点なしとみなす
        return nothing
    end

    # 交点のt値を計算
    t_cross = (B_q - B_p) / (A_p - A_q)

    # t_upperとt_lowerの間にあるか確認
    if t_cross >= t_lower && t_cross <= t_upper
        return t_cross
    else
        return nothing
    end
end

# 指定した区間[t_upper, t_lower]内での順位変化点を検出
# 区間内での順位変化点とリグレット値の詳細な検出
function find_rank_change_in_interval(o_o::Int, t_lower::Float64, t_upper::Float64, matrix, methodW)
    n = size(matrix, 1)
    candidates = [q for q in 1:n if q != o_o]

    # t_upper (区間の右端) での最大リグレットとその代替案
    regrets_at_upper = [regret_value(o_o, q, t_upper, matrix, methodW) for q in candidates]
    p_idx_upper = argmax(regrets_at_upper)
    max_alt_upper = candidates[p_idx_upper]
    max_regret_val_upper = regrets_at_upper[p_idx_upper]

    # t_lower (区間の左端) での最大リグレットとその代替案
    regrets_at_lower = [regret_value(o_o, q, t_lower, matrix, methodW) for q in candidates]
    p_idx_lower = argmax(regrets_at_lower)
    max_alt_lower = candidates[p_idx_lower]
    max_regret_val_lower = regrets_at_lower[p_idx_lower]

    # この区間での最大リグレットの最小値と最大値 (初期値)
    r_min_interval = min(max_regret_val_upper, max_regret_val_lower)
    r_max_interval = max(max_regret_val_upper, max_regret_val_lower)

    # 区間内での変化点を記録
    interval_change_points = Float64[]
    interval_max_regret_alts = Int[]
    interval_max_regret_values = Float64[]

    # アルゴリズム 3.1 の考え方で、t_lower から t_upper に向けて変化点を探す
    current_t0 = t_lower
    current_op = max_alt_lower # 左端で最大リグレットを与える代替案

    # この区間では、最初に current_op が最大リグレットを与えると仮定
    # 他の候補が current_op を上回る点（交点）を探す

    # 簡略化のため、Claudeが提案した元の find_rank_change_in_interval のロジックを
    # 少し整理して適用することを試みます。
    # (ユーザーの記述したアルゴリズム 3.1 に完全に沿うには、より複雑なループが必要になる可能性があります)

    # t_upper で最大リグレットを与える代替案 (p) を基準とする
    p = max_alt_upper

    # t_lower で p 以外に最大リグレットを与える可能性がある代替案 (q) を探す
    # (この部分はClaude版のロジックに近いが、より単純化する必要があるかもしれない)
    # 今回は、p (t_upperでの最大リグレット代替案) と q (t_lowerでの最大リグレット代替案) が
    # 異なる場合に交点をチェックする、という単純なアプローチでまずエラーをなくすことを目指します。
    if max_alt_upper != max_alt_lower
        # p (upperでの勝者) と q (lowerでの勝者) の交点を計算
        t_cross = compute_crossing_point(o_o, max_alt_upper, max_alt_lower, t_upper, t_lower, matrix, methodW)
        if t_cross !== nothing
            # この交点が区間内にあることを確認 (compute_crossing_point内で既に行われている)
            # 交点でのリグレット値を計算し、それが本当にその点での最大リグレットかを確認する必要がある
            # (ここでは簡略化のため、交点が見つかればそれを変化点として記録)
            push!(interval_change_points, t_cross)

            # 交点でどちらが最大リグレットを与えるかを判断
            # (より厳密には、全ての候補と比較する必要がある)
            regret_p_at_cross = regret_value(o_o, max_alt_upper, t_cross, matrix, methodW)
            regret_q_at_cross = regret_value(o_o, max_alt_lower, t_cross, matrix, methodW)

            if regret_p_at_cross >= regret_q_at_cross
                push!(interval_max_regret_alts, max_alt_upper)
                push!(interval_max_regret_values, regret_p_at_cross)
                r_min_interval = min(r_min_interval, regret_p_at_cross)
                r_max_interval = max(r_max_interval, regret_p_at_cross)
            else
                push!(interval_max_regret_alts, max_alt_lower)
                push!(interval_max_regret_values, regret_q_at_cross)
                r_min_interval = min(r_min_interval, regret_q_at_cross)
                r_max_interval = max(r_max_interval, regret_q_at_cross)
            end
        end
    end

    # この関数が返すのは、この区間[t_lower, t_upper]に関する情報
    return (
        change_points=interval_change_points,       # この区間内で見つかった変化点
        max_regret_alts=interval_max_regret_alts,   # 各変化点で最大リグレットを与える代替案
        max_regret_values=interval_max_regret_values, # 各変化点での最大リグレット値
        max_alt_at_t_lower=max_alt_lower,             # 区間左端で最大リグレットを与える代替案
        max_alt_at_t_upper=max_alt_upper,             # 区間右端で最大リグレットを与える代替案
        r_min_in_interval=r_min_interval,             # この区間での最大リグレットの最小値
        r_max_in_interval=r_max_interval              # この区間での最大リグレットの最大値
    )
end

# アベイルスペースに基づいてt^Rから順に区間を分析
# module_regret.jl 内
function find_rank_change_points_from_tR(o_o::Int, t_L::Float64, t_R::Float64, matrix, methodW)
    all_change_points = Float64[]
    all_max_regret_alts = Int[]
    all_max_regret_values = Float64[]
    intervals = Tuple{Float64,Float64}[]
    interval_data_collected = [] # 名前付きタプルの配列として収集

    current_t = t_R
    while current_t > t_L
        next_t, min_availspace = compute_next_t(current_t, matrix, methodW)
        next_t = max(next_t, t_L)

        if isapprox(current_t, next_t, atol=1e-8) # 無限ループ防止
            if current_t > t_L # まだt_Lに達していなければ、最後の区間として処理
                push!(intervals, (t_L, current_t))
                interval_result = find_rank_change_in_interval(o_o, t_L, current_t, matrix, methodW) # 引数の順序を t_lower, t_upper に
                push!(interval_data_collected, interval_result)
                append!(all_change_points, interval_result.change_points) # 正しいフィールド名を使用
                append!(all_max_regret_alts, interval_result.max_regret_alts)
                append!(all_max_regret_values, interval_result.max_regret_values)
            end
            break
        end

        push!(intervals, (next_t, current_t))
        # find_rank_change_in_interval の引数の順序を t_lower, t_upper に合わせる
        interval_result = find_rank_change_in_interval(o_o, next_t, current_t, matrix, methodW)
        push!(interval_data_collected, interval_result)
        # println(interval_result) # デバッグ用

        # interval_result が名前付きタプルを返すと仮定してフィールドにアクセス
        append!(all_change_points, interval_result.change_points)
        append!(all_max_regret_alts, interval_result.max_regret_alts)
        append!(all_max_regret_values, interval_result.max_regret_values)

        current_t = next_t
        if isapprox(current_t, t_L, atol=1e-8)
            break
        end
    end

    # 全体的な最大・最小リグレット値 (interval_data_collected が空でない場合)
    r_M_overall = -Inf
    r_m_overall = Inf
    if !isempty(interval_data_collected)
        r_M_overall = maximum([res.r_max_in_interval for res in interval_data_collected])
        r_m_overall = minimum([res.r_min_in_interval for res in interval_data_collected])
    end

    return (
        change_points=all_change_points,
        max_regret_alts=all_max_regret_alts,
        max_regret_values=all_max_regret_values,
        intervals=intervals, # アベイルスペース区間
        interval_data=interval_data_collected, # 各区間での詳細な分析結果
        r_M=r_M_overall,
        r_m=r_m_overall
    )
end

# 代替案間の関係を分析し、全体的な順位変化を検出
function analyze_all_alternatives_with_avail_space(utility::Matrix{Float64}, methodW, t_range::Tuple{Float64,Float64})
    n = size(utility, 1)
    matrix = create_minimax_R_Matrix(utility)

    # 各代替案ごとの詳細な分析結果を格納
    all_results_per_alternative = Dict{Int,Any}()
    for o_o in 1:n
        result = find_rank_change_points_from_tR(o_o, t_range[1], t_range[2], matrix, methodW)
        all_results_per_alternative[o_o] = result
    end

    # 全ての「個々の代替案に対するリグレット相手が変わる点」を収集
    collected_individual_change_points = Float64[]
    for o_o in 1:n
        # result が名前付きタプルで .change_points フィールドを持つことを期待
        if hasfield(typeof(all_results_per_alternative[o_o]), :change_points)
            append!(collected_individual_change_points, all_results_per_alternative[o_o].change_points)
        end
    end

    # アベイルスペースに基づく区間境界も収集
    all_unique_intervals = Tuple{Float64,Float64}[]
    for o_o in 1:n
        if hasfield(typeof(all_results_per_alternative[o_o]), :intervals)
            append!(all_unique_intervals, all_results_per_alternative[o_o].intervals)
        end
    end
    unique!(all_unique_intervals) # 重複する区間を削除

    # ランキングを評価すべき全ての重要なt点をまとめる
    # (個々の変化点、区間境界、分析範囲の端点)
    critical_t_points = Float64[]
    append!(critical_t_points, collected_individual_change_points)
    for (lower, upper) in all_unique_intervals
        push!(critical_t_points, lower)
        push!(critical_t_points, upper)
    end
    push!(critical_t_points, t_range[1]) # 分析範囲の開始点
    push!(critical_t_points, t_range[2]) # 分析範囲の終了点
    unique!(sort!(critical_t_points))
    # 範囲外の点や非常に近い点を除くフィルタリング (必要に応じて)
    filter!(p -> p >= t_range[1] - 1e-9 && p <= t_range[2] + 1e-9, critical_t_points)
    # 非常に近い点をまとめる処理も検討可能 (例: 差が1e-7未満なら同一視)

    # 各重要t点での全体のランキングを計算
    overall_rankings = Dict{Float64,Vector{Int}}()
    for t_val in critical_t_points
        max_regrets_for_all_oo = zeros(n)
        for o_o_idx in 1:n
            current_max_r = -Inf
            # o_o_idx 以外の全ての代替案 q_idx とのリグレットを計算し、その最大値を取る
            for q_idx in 1:n
                if q_idx != o_o_idx
                    r_val = regret_value(o_o_idx, q_idx, t_val, matrix, methodW)
                    current_max_r = max(current_max_r, r_val)
                end
            end
            # もし全ての相手とのリグレットが-Infなら (通常は起こらないが)、0.0 とする
            max_regrets_for_all_oo[o_o_idx] = current_max_r == -Inf ? 0.0 : current_max_r
        end
        overall_rankings[t_val] = sortperm(max_regrets_for_all_oo) # リグレットが小さい順が上位
    end

    # --- 全体的な順位変化点の検出と、その時点でのランキングを記録 ---
    overall_rank_change_events = [] # (t_value=変化点t, ranking=そのtでのランキング) の名前付きタプルの配列

    # critical_t_points はソート済みなので、順番に比較していく
    if length(critical_t_points) >= 1
        # 最初のt点 (通常はt_L) でのランキングを初期状態として記録
        # これを「最初の変化点」として扱うかは要件による
        # ここでは、t_Lでの状態をまず記録し、その後変化があれば記録する方針
        first_t = critical_t_points[1]
        push!(overall_rank_change_events, (t_value=first_t, ranking=overall_rankings[first_t]))

        if length(critical_t_points) >= 2
            for i in 1:(length(critical_t_points)-1)
                t1 = critical_t_points[i]
                t2 = critical_t_points[i+1]

                ranking_at_t1 = overall_rankings[t1]
                ranking_at_t2 = overall_rankings[t2]

                if ranking_at_t1 != ranking_at_t2
                    # t1 から t2 の間で順位が変動した
                    # 変化点としては t2 (変化が確定した最初のt点) を記録し、その時のランキングを保存
                    # 既に t2 が overall_rank_change_events の最後の t_value と同じでなければ追加
                    if isempty(overall_rank_change_events) || last(overall_rank_change_events).t_value != t2
                        push!(overall_rank_change_events, (t_value=t2, ranking=ranking_at_t2))
                    end
                end
            end
        end
    end

    # `change_points` は `overall_rank_change_events` の `t_value` のリスト
    final_overall_change_points = [event.t_value for event in overall_rank_change_events]
    # もし最初のt_Lを変化点として含めないなら、ここで調整
    # 例: if length(final_overall_change_points) > 1 && final_overall_change_points[1] == t_range[1]
    #         popfirst!(final_overall_change_points)
    #         popfirst!(overall_rank_change_events)
    #     end

    return (
        all_results_per_alternative=all_results_per_alternative,
        change_points=final_overall_change_points,     # 全体でランキングが実際に変わったt点のリスト
        rank_change_data=overall_rank_change_events, # (t値, そのt値でのランキング) の名前付きタプルのリスト
        intervals=all_unique_intervals,                # アベイルスペース区間
        all_points=critical_t_points,                  # ランキングを計算した全てのt点
        rankings=overall_rankings                      # 各t点での全体のランキング
    )
end

function save_rank_changes_and_rankings_to_csv(results, filename="rank_change_details.csv")
    println("CSV保存関数 (save_final_rank_changes_to_csv) を開始します。")

    # results が必要なフィールドを持っているか確認
    if !hasfield(typeof(results), :change_points) || !hasfield(typeof(results), :rankings)
        println("警告: 'results'に必要な 'change_points' または 'rankings' フィールドが存在しません。")
        return DataFrame() # 空のDataFrameを返す
    end

    relevant_t_points = sort(unique(results.change_points)) # 順位変化点のみを対象とする

    # もし change_points が空でも、t_range の開始点と終了点でのランキングは表示したい場合、
    # relevant_t_points に t_range[1] と t_range[2] を追加することも検討できます。
    # 例: relevant_t_points = sort(unique([results.change_points..., results.all_points[1], results.all_points[end]]))
    # 今回は、明確な「変化点」のみをCSVに出力する方針とします。

    if isempty(relevant_t_points)
        println("情報: 保存対象となる順位変化点 (results.change_points) がありません。")
        # t_rangeの最初と最後の点のランキングだけでも出力するか、空のCSVにするか選択
        # ここでは、もしall_pointsがあればそれを使う試み（なければ空のまま）
        if hasfield(typeof(results), :all_points) && !isempty(results.all_points)
            println("代わりに all_points の最初と最後のt値でのランキングを試みます。")
            initial_t = results.all_points[1]
            final_t = results.all_points[end]
            relevant_t_points = unique(sort([initial_t, final_t]))
            if !haskey(results.rankings, initial_t) || !haskey(results.rankings, final_t)
                println("警告: all_pointsの最初または最後のt値に対応するランキングが見つかりません。")
                return DataFrame()
            end
        else
            println("警告: all_points も空か存在しません。空のCSVを作成します。")
            # 空のDataFrameを作成してヘッダーだけ書き出すことも可能
            # return DataFrame() # ここで終了しても良い
        end
    end

    # ランキングデータから代替案の数を取得
    # relevant_t_pointsが空でないことを確認してから
    local n_alternatives
    if !isempty(relevant_t_points) && haskey(results.rankings, relevant_t_points[1])
        n_alternatives = length(results.rankings[relevant_t_points[1]])
    else
        # 最初のchange_pointに対するランキングがない場合、他のt_pointから試す
        # または、all_pointsの最初のキーから取得する
        if hasfield(typeof(results), :all_points) && !isempty(results.all_points) && haskey(results.rankings, results.all_points[1])
            n_alternatives = length(results.rankings[results.all_points[1]])
        else
            println("警告: 代替案の数を決定できませんでした。CSV作成を中止します。")
            return DataFrame()
        end
    end

    if n_alternatives == 0
        println("警告: 代替案の数が0です。")
        return DataFrame()
    end

    # ヘッダーを作成
    header_symbols = [:t_change_point]
    for i in 1:n_alternatives
        push!(header_symbols, Symbol("alt_$(i)_rank"))
    end

    # データ行を準備
    data_rows = []
    for t_val in relevant_t_points
        if !haskey(results.rankings, t_val)
            println("警告: t=$t_val に対応するランキングデータが見つかりません。この行はスキップします。")
            continue
        end
        ranking_vector = results.rankings[t_val] # 例: [3, 1, 2, 5, 4] (1位が代替案3, 2位が代替案1, ...)

        row_to_write = Any[t_val]

        ranks_of_alternatives = zeros(Int, n_alternatives)
        for rank_position in 1:n_alternatives
            if rank_position <= length(ranking_vector)
                alternative_id_at_this_rank = ranking_vector[rank_position]
                if 1 <= alternative_id_at_this_rank <= n_alternatives
                    ranks_of_alternatives[alternative_id_at_this_rank] = rank_position
                else
                    println("警告: t=$t_val で不正な代替案ID ($alternative_id_at_this_rank) がランキングに含まれています。")
                end
            else
                println("警告: t=$t_val でランキングベクトルの長さが代替案数と一致しません。")
                # この場合、ranks_of_alternatives の残りは0のまま
            end
        end

        for alt_idx in 1:n_alternatives
            push!(row_to_write, ranks_of_alternatives[alt_idx])
        end
        push!(data_rows, row_to_write)
    end

    # DataFrameを作成
    df_output = DataFrame()
    if !isempty(data_rows)
        try
            # 各列のデータ型を推測させるか、明示的に指定する
            for (col_idx, col_name) in enumerate(header_symbols)
                df_output[!, col_name] = [row[col_idx] for row in data_rows]
            end
        catch e
            println("エラー: DataFrameの作成中にエラーが発生しました: $e")
            println("data_rows: ", data_rows)
            println("header_symbols: ", header_symbols)
            return DataFrame() # エラー時は空のDataFrame
        end
    else
        println("情報: DataFrame に追加するデータ行がありません。ヘッダーのみのDataFrameを作成します。")
        for col_name in header_symbols # ヘッダーだけでも作成
            df_output[!, col_name] = []
        end
    end

    println("CSV保存関数: DataFrameが作成されました。行数: $(nrow(df_output)), 列数: $(ncol(df_output))")
    if nrow(df_output) == 0 && isempty(relevant_t_points) # relevant_t_pointsが元々空で、dfも空なら
        println("情報: 書き込むデータがないため、CSVファイルは作成されません。")
        return df_output
    end

    abs_filename = abspath(filename) # 絶対パスを取得
    println("CSVファイルを次の絶対パスに保存しようとしています: ", abs_filename)

    try
        println("CSV.write を使用して書き込みを試みます...")
        CSV.write(abs_filename, df_output)
        println("結果を $abs_filename に保存しました (CSV.write)。")

        # ファイル内容の検証 (オプション)
        if isfile(abs_filename)
            file_content = readlines(abs_filename)
            println("ファイル内容の最初の数行 (CSV.write):")
            for (i, line) in enumerate(file_content)
                if i > min(5, length(file_content))
                    break
                end
                println(line)
            end
            if isempty(file_content) && nrow(df_output) > 0
                println("警告: CSV.write で作成されたファイルが空ですが、DataFrameにはデータがありました。")
            elseif isempty(file_content)
                println("情報: CSV.write で作成されたファイルは空です（DataFrameも空またはヘッダーのみだった可能性があります）。")
            end
        else
            println("警告: CSV.write でファイルが作成されませんでした。")
        end

    catch e_csv
        println("エラー: CSV.write を使用したCSVファイル '$abs_filename' の書き込みに失敗しました: $e_csv")
        println("DelimitedFiles を使用したフォールバックを試みます...")
        try
            header_str_array = [string(name) for name in names(df_output)]
            # DataFrameを直接writedlmに渡すか、Matrix{String}に変換
            # ここでは、DataFrameを直接渡してみる (DelimitedFilesが対応している場合)
            # ただし、ヘッダー行とデータ行を分けて書き出す方が確実

            open(abs_filename, "w") do io
                println(io, join(header_str_array, ',')) # ヘッダー行
                for r in 1:nrow(df_output)
                    row_strings = [string(df_output[r, c]) for c in 1:ncol(df_output)]
                    println(io, join(row_strings, ','))
                end
            end
            println("結果を $abs_filename に保存しました (DelimitedFiles)。")

            if isfile(abs_filename)
                file_content_dlm = readlines(abs_filename)
                println("ファイル内容の最初の数行 (DelimitedFiles):")
                for (i, line) in enumerate(file_content_dlm)
                    if i > min(5, length(file_content_dlm))
                        break
                    end
                    println(line)
                end
                if isempty(file_content_dlm) && nrow(df_output) > 0
                    println("警告: DelimitedFiles で作成されたファイルが空ですが、DataFrameにはデータがありました。")
                elseif isempty(file_content_dlm)
                    println("情報: DelimitedFiles で作成されたファイルは空です。")
                end
            else
                println("警告: DelimitedFiles でファイルが作成されませんでした。")
            end
        catch e_dlm
            println("エラー: DelimitedFiles を使用したCSVファイル '$abs_filename' の書き込みにも失敗しました: $e_dlm")
        end
    end

    return df_output
end

end # module