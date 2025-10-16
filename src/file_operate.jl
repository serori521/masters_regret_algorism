# 修正ステップ1: CSVパッケージの除去
using Pkg
import Pkg;
using DelimitedFiles, StatsBase
using DataFrames  # CSVを削除

using StringEncodings, Logging

# 修正ステップ2: DataFrame生成方法の変更
@inline function read_utility_value(utility::String)
    csv_path = "/workspaces/inulab_julia_devcontainer/data/効用値行列/" * utility * "/N=6_M=5/u.csv"
    data = readdlm(csv_path, ',', Float64)

    # Matrixから直接DataFrameを生成（中間処理削減）
    utility_for_alta = DataFrame[]
    for i in 1:5:size(data, 1)
        push!(utility_for_alta, DataFrame(data[i:i+4, :], :auto))
    end
    return utility_for_alta
end

# 修正ステップ3: 文字コード処理の最適化
function read_method_weights(filename::String, repeat_num::Int, criteria_num::Int)
    csv_path = "/workspaces/inulab_julia_devcontainer/data/Simp/N=6/a3/" * filename * "/Simp.csv"

    # StringEncodingsを直接使用
    io = open(csv_path, enc"SHIFT_JIS", "r")
    data = readdlm(io, ',', Float64; skipstart=3)
    close(io)

    # メモリ効率の良いデータ生成
    result = Vector{NamedTuple}(undef, repeat_num)
    for i in 1:repeat_num
        result[i] = (
            L=data[i, 2:2:2+criteria_num*2-1],
            R=data[i, 3:2:2+criteria_num*2],
            adjacent=data[i, 2+criteria_num*2]
        )
    end
    return result
end



#真の区間重要度を読み込む関数
function read_true_weights(generate_method::String)
    csv_path = "/workspaces/inulab_julia_devcontainer/data/true_interval_weight_set/N=6/" * generate_method * "/Given_interval_weight.csv"
    data = readdlm(csv_path, ',', Float64)
    n = length(data)

    return (
        L=[data[i] for i in 1:2:n-1],
        R=[data[i] for i in 2:2:n]
    )
end
