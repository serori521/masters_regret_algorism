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
function read_method_weights(paths, filename::String, repeat_num::Int, criteria_num::Int=6; a3::String="a3")

    csv_path = joinpath(paths.data, "Simp", "N=$(criteria_num)", a3, filename, "Simp.csv")
    isfile(csv_path) || error("File not found: $csv_path")

    needed_cols = 2 * criteria_num + 2   # ID + 2N + adjacent
    start_id = 1.0

    # 1行ぶんを NamedTuple にできたら返す、ダメなら nothing
    function parse_weight_row(row)
        length(row) >= needed_cols || return nothing

        # 先頭が数値1.0か（開始以降のデータ行）
        v1 = tryparse(Float64, strip(string(row[1])))
        v1 == start_id || return nothing

        wL = Vector{Float64}(undef, criteria_num)
        wR = Vector{Float64}(undef, criteria_num)

        @inbounds for k in 1:criteria_num
            idxL = 2 + (k-1)*2
            idxR = 3 + (k-1)*2
            wL[k] = parse(Float64, strip(string(row[idxL])))
            wR[k] = parse(Float64, strip(string(row[idxR])))
        end

        # adjacent は「ID(1) + 2N の次」= 2N+2 列目
        idx_adj = 2 * criteria_num + 2
        adjacent = parse(Float64, strip(string(row[idx_adj])))

        return (L=wL, R=wR, adjacent=adjacent)
    end

    function extract_data(io_or_path; label::String)
        data = NamedTuple[]
        found_start = false

        rows = CSV.Rows(io_or_path; header=false, reusebuffer=true, skipto=2)

        for row in rows
            length(data) >= repeat_num && break

            # 開始行（ID=1.0）を見つけるまでスキップ
            if !found_start
                v1 = ismissing(row[1]) ? nothing : tryparse(Float64, strip(string(row[1])))
                if v1 == start_id
                    found_start = true
                else
                    continue
                end
            end

            nt = try
                # found_start以降は「列数不足行」を飛ばしつつ parse
                length(row) < needed_cols ? nothing : begin
                    wL = Vector{Float64}(undef, criteria_num)
                    wR = Vector{Float64}(undef, criteria_num)

                    @inbounds for k in 1:criteria_num
                        idxL = 2 + (k-1)*2
                        idxR = 3 + (k-1)*2
                        wL[k] = parse(Float64, strip(string(row[idxL])))
                        wR[k] = parse(Float64, strip(string(row[idxR])))
                    end

                    idx_adj = 2 * criteria_num + 2
                    adjacent = parse(Float64, strip(string(row[idx_adj])))

                    (L=wL, R=wR, adjacent=adjacent)
                end
            catch
                nothing
            end

            nt === nothing && continue
            push!(data, nt)
        end

        return data
    end

    # UTF-8 → ダメなら Shift_JIS
    data = try
        extract_data(csv_path; label="utf8")
    catch
        NamedTuple[]
    end

    if isempty(data)
        try
            open(csv_path, enc"SHIFT_JIS", "r") do io
                data = extract_data(io; label="shift_jis")
            end
        catch
            # ここでは握りつぶして最後に error
        end
    end

    isempty(data) && error("Failed to read weights from $csv_path. Tried UTF-8 and SHIFT_JIS.")
    length(data) < repeat_num && @warn "Requested $repeat_num rows, but only got $(length(data))" csv_path=csv_path

    return data
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
