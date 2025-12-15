# src/load_instance.jl
module LoadInstance

using DelimitedFiles
using StringEncodings
using CSV
using DataFrames
using StringEncodings # enc"SHIFT_JIS" を使うために必要
# Paths.project_paths() を LPSv2 から呼ぶ想定なので、ここでは joinpath を使うだけ

"効用値行列を読む: data/効用値行列/<utility>/N=6_M=5/u.csv"
function read_utility_value(paths, utility::String; N::Int=6, M::Int=5)
    csv_path = joinpath(paths.data, "効用値行列", utility, "N=$(N)_M=$(M)", "u.csv")
    data = readdlm(csv_path, ',', Float64)
    mats = Matrix{Float64}[]
    for i in 1:M:size(data, 1)
        push!(mats, Matrix(data[i:i+M-1, :]))
    end
    return mats
end

"""
手法重みを読む:
data/Simp/N=6/a3/<filename>/Simp.csv をSHIFT_JISで読み、必要部分を抽出
"""


"""
    read_method_weights(paths, filename, repeat_num, criteria_num=6; a3="a3")

指定されたCSVファイルを読み込み、重みパラメータを抽出します。
CSV.jlのストリーム処理を使用することで、ファイル全体をメモリに読み込む無駄を省き、
ヘッダ行数が異なるフォーマット（日本語ヘッダ/英語ヘッダ）の両方に柔軟に対応します。
"""
function read_method_weights(paths, filename::String, repeat_num::Int, criteria_num::Int=6;
                             a3::String="a3")
    
    # パスの構築
    # paths.data が文字列であることを前提としています
    csv_path = joinpath(paths.data, "Simp", "N=$(criteria_num)", a3, filename, "Simp.csv")
    
    if !isfile(csv_path)
        error("File not found: $csv_path")
    end

    # --- 内部関数: ストリームからデータを抽出する ---
    # io: ファイルIO または ファイルパス
    # 戻り値: (成功フラグ, 結果のVector)
    function extract_data(io_or_path)
        data = Vector{NamedTuple}()
        found_start = false
        
        # header=false: ヘッダ行数に関わらず全行をデータ候補として扱う
        # reusebuffer=true: メモリ割り当てを減らす最適化
        rows = CSV.Rows(io_or_path; header=false, reusebuffer=true)

        for row in rows
            # 必要な数だけ集まったら終了
            if length(data) >= repeat_num
                break
            end

            # 1列目の値を確認（CSV.Rowsの値は文字列なのでパースを試みる）
            # データ行は必ずID（数値の1）から始まると仮定
            if ismissing(row[1])
                continue
            end
            
            val1_str = strip(string(row[1]))
            val1 = tryparse(Float64, val1_str)

            if !found_start
                # データ開始行を探す: 1列目が 1.0 である行
                if val1 == 1.0
                    found_start = true
                    # この行もデータ処理対象なので下へ続く
                else
                    continue # ヘッダ行または空行とみなしてスキップ
                end
            end

            # --- データ行の処理 ---
            if found_start
                # 必要なカラム数をチェック
                # 構成: [ID, wL0, wR0, ..., wL(N-1), wR(N-1), adjacent]
                # ID(1) + 2*N + adjacent(1) = 2N + 2
                # ※ CSV.Rows は行によって長さが変わる可能性を考慮して length(row) チェック
                if length(row) < 2 * criteria_num + 2
                    continue # カラム不足行はスキップ
                end

                try
                    wL = Vector{Float64}(undef, criteria_num)
                    wR = Vector{Float64}(undef, criteria_num)

                    for k in 1:criteria_num
                        # CSVのカラムは1始まり。ID列(1)の次から重みデータ
                        # wL[k] -> col index: 2 + (k-1)*2
                        # wR[k] -> col index: 3 + (k-1)*2
                        idx_L = 2 + (k-1)*2
                        idx_R = 3 + (k-1)*2
                        
                        wL[k] = parse(Float64, strip(string(row[idx_L])))
                        wR[k] = parse(Float64, strip(string(row[idx_R])))
                    end
                    
                    # adjacent は重みデータの次
                    idx_adj = 2 + 2 * criteria_num + 1 # 元コードの論理では +1 だが、位置的には wRの次
                    # 元コード: row[2 + 2*criteria_num] は0始まりインデックス配列の末尾を指していたと思われる
                    # ここでは正確な位置: ID(1) + 2*N(個) の次の列(adjacent) = 1 + 2N + 1 = 2N + 2 列目
                    # しかしアップロードされたファイルを見ると adjacent は「区間幅の総和」として末尾にある
                    # 念の為、元のロジック `row[2 + 2*criteria_num]` (1始まりなら row[1 + 2*criteria_num + 1]?) を踏襲するなら
                    # ユーザーコード: vals = row[2 : 1 + 2*criteria_num] -> 長さ 2N
                    # adjacent = row[2 + 2*criteria_num] -> これは vals の次の要素 (インデックス 2N+2)
                    
                    idx_adj = 2 + 2 * criteria_num
                    # もし adjacent が最後の列なら、上記の計算で合っているか確認が必要ですが、
                    # ここでは元のコードの意図（IDの次から2N個読み、その次を読む）に従います。
                    
                    adjacent = parse(Float64, strip(string(row[idx_adj])))

                    push!(data, (L=wL, R=wR, adjacent=adjacent))
                catch e
                    # パースエラー（数値変換できない等）があればスキップ
                    continue
                end
            end
        end
        
        return (!isempty(data), data)
    end

    # --- メイン処理: エンコーディング対応 ---
    
    # 戦略:
    # 1. まず標準的な UTF-8 で読み込みを試みる (Simp - コピー.csv 向け)
    # 2. 失敗（データが見つからない、またはエラー）したら Shift_JIS で試みる (Simp.csv 向け)

    result = Vector{NamedTuple}()
    success = false

    # 1. UTF-8 で試行
    try
        is_ok, res = extract_data(csv_path)
        if is_ok
            result = res
            success = true
        end
    catch
        # UTF-8 でのエラーは無視して次へ
    end

    # 2. Shift_JIS で試行 (UTF-8でダメだった場合)
    if !success
        try
            open(csv_path, enc"SHIFT_JIS", "r") do io
                is_ok, res = extract_data(io)
                if is_ok
                    result = res
                    success = true
                end
            end
        catch e
            # Shift_JIS でもエラーなら、最終的にエラーを投げるためにここでは何もしない
        end
    end

    if !success
        error("Failed to read weights from $csv_path. Tried UTF-8 and SHIFT_JIS.")
    end

    if length(result) < repeat_num
        @warn "Requested $repeat_num rows, but only got $(length(result)) in $csv_path"
    end

    return result
end


"真の区間重要度: data/true_interval_weight_set/N=6/<generate_method>/Given_interval_weight.csv"
function read_true_weights(paths, generate_method::String; N::Int=6)
    csv_path = joinpath(paths.data, "true_interval_weight_set", "N=$(N)", generate_method, "Given_interval_weight.csv")
    data = readdlm(csv_path, ',', Float64)
    n = length(data)
    return (
        L = [data[i] for i in 1:2:n-1],
        R = [data[i] for i in 2:2:n]
    )
end

end # module
