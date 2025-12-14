# src/load_instance.jl
module LoadInstance

using DelimitedFiles
using StringEncodings

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
function read_method_weights(paths, filename::String, repeat_num::Int, criteria_num::Int=6;
                             a3::String="a3")
    csv_path = joinpath(paths.data, "Simp", "N=$(criteria_num)", a3, filename, "Simp.csv")

    io = open(csv_path, enc"SHIFT_JIS", "r")
    data = readdlm(io, ',', Float64; skipstart=3)
    close(io)

    result = Vector{NamedTuple}(undef, repeat_num)
    for i in 1:repeat_num
        result[i] = (
            L = data[i, 2:2:2+criteria_num*2-1],
            R = data[i, 3:2:2+criteria_num*2],
            adjacent = data[i, 2+criteria_num*2]
        )
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
