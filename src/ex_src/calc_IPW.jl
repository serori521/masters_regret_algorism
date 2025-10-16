#一番重要なリグレットを計算するファイル
#複数存在する最適解のtの範囲を求める関数
function find_optimal_trange(L::Array{Float64,1}, R::Array{Float64,1})
    max_sum = -Inf
    min_sum = Inf
    n = length(L)

    for j in 1:n
        sum_ij_R = sum(L[i] for i in 1:n if i != j) + R[j]
        sum_ij_L = sum(R[i] for i in 1:n if i != j) + L[j]
        max_sum = max(max_sum, sum_ij_R)
        min_sum = min(min_sum, sum_ij_L)
    end
    t_R = 1 / max_sum
    t_L = 1 / min_sum
    return min(t_L, t_R), max(t_L, t_R)
end

# タプルの要素として使用するデータ構造を定義する
mutable struct minimax_regret_tuple
    difference_U::Vector{Float64}   #各評価基準の効用値の差1*6
    rank::Vector{Int}               #効用値の差の降順に並び替えたときのインデックス1*6
    regret::Float64                 #各代替案の組み合わせにおけるregret
    interm_index::Int               #途中代替案のインデックス
    Avail_space::Float64            #現在の途中のものと最大値との差
end

mutable struct maxregret_miniAvail_tuple
    max_regret::Vector{Float64}
    maxR_index::Vector{Int}
    mini_Avail::Vector{Float64}
    miniA_index::Vector{Int}
end

function create_minimax_R_Matrix(utility::Matrix)   #Matrixは5*6の行列
    #タプルを要素として持つ5*5の行列を定義
    matrix = [minimax_regret_tuple([], [], 0.0, 0, 0.0) for _ in 1:5, _ in 1:5]
    #各評価基準の効用値の差を計算
    difference_U = []
    for i in 1:5, j in i+1:5
        diff = vec(utility[j, :] - utility[i, :])   #i,j行目の効用値の差を計算
        rank = sortperm(diff, rev=true)             #効用値の差の降順に並び替えたときのインデックスを返す
        matrix[i, j].difference_U = diff
        matrix[i, j].rank = rank
    end
    return matrix
end


@inline function calc_regret(matrix::Array{minimax_regret_tuple,2}, Y_L::Array{Float64,1}, Y_R::Array{Float64,1})

    for i in 1:5, j in i:5
        if i == j
            matrix[i, j].regret = 0.0
        else
            #右上三角行列のみ計算
            rank = matrix[i, j].rank
            diff = matrix[i, j].difference_U
            #各評価基準の効用値の差の降順に並び替えたときのインデックスを取得
            regret = sum(diff[rank[t]] .* Y_L[rank[t]] for t in 1:6)
            z = 1 - sum(Y_L[rank[t]] for t in 1:6)
            width_Y = Y_R .- Y_L
            index = 1
            Avail_space = 0.0
            for index = 1:6
                if (width_Y[rank[index]] < z)
                    regret = regret + diff[rank[index]] * width_Y[rank[index]]
                    z = z - width_Y[rank[index]]
                else
                    regret = regret + diff[rank[index]] * z
                    Avail_space = width_Y[rank[index]] - z
                    if Avail_space < 0.0000001
                        Avail_space = width_Y[rank[index+1]]
                    end
                    break
                end
            end
            matrix[i, j].regret = regret
            matrix[i, j].interm_index = rank[index]
            matrix[i, j].Avail_space = Avail_space

            #左下三角行列の要素を右上三角行列の要素と反対に計算する
            #各評価基準の効用値の差の降順に並び替えたときのインデックスを取得
            index_r = length(rank)
            regret_r = -sum(diff[rank[s]] .* Y_L[rank[s]] for s in 1:6)
            z_r = 1 - sum(Y_L[rank[s]] for s in 1:6)
            Avail_space_r = 0.0
            for index_r in 6:-1:1
                if (width_Y[rank[index_r]] < z_r)
                    regret_r = regret_r - diff[rank[index_r]] * width_Y[rank[index_r]]
                    z_r = z_r - width_Y[rank[index_r]]
                else
                    regret_r = regret_r - diff[rank[index_r]] * z_r
                    Avail_space_r = width_Y[rank[index_r]] - z_r
                    if Avail_space_r < 0.0000001
                        Avail_space_r = width_Y[rank[index_r-1]]
                    end
                    break
                end
            end
            matrix[j, i].regret = regret_r
            matrix[j, i].interm_index = rank[index_r]
            matrix[j, i].Avail_space = Avail_space_r
        end
    end
    #最大リグレットを小さい順に並べる
    i = 1
    regret_max = Vector{Any}(undef, 5)
    for i in 1:5
        regret_max[i] = findmax([matrix[i, j].regret for j in 1:5])
    end
    rank_regret = sortperm(sortperm(regret_max, rev=false))
    return regret_max, rank_regret
end