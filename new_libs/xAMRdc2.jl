"""
区間重要度推定法 eAMRd/c, gAMRd/c の関数
step3の目的関数を1~n全体を対象に変更
xAMRdc(PCM, method)とすることで，区間重要度が求められる．
In: PCM:Matrix型, method:EV or GM
Out:LPResult_Individual型(下で定義)
"""

using IntervalArithmetic
using IntervalArithmetic.Symbols
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")
include("./solve-deterministic-ahp.jl")

# Phase 1-4の結果を格納する型
# MMRD_Individual_kai = @NamedTuple{
#     s::T,
#     centers::Matrix{T},
#     l::Matrix{T},
#     wᴸ::Vector{T}, wᵁ::Vector{T},
#     W::Vector{Interval{T}} # ([wᵢᴸ, wᵢᵁ])
# } where {T<:Real}

LPResult_Individual = @NamedTuple{
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}},
} where {T<:Real}

# 任意の行と列を削除
@inline function remove_row_col(A::Matrix{T}, row::Int, col::Int)::Matrix{T} where {T<:Real}
    m, n = size(A)

    # 行を除外
    new_matrix = A[setdiff(1:m, row), :]
    # 列を除外
    result_matrix = new_matrix[:, setdiff(1:n, col)]

    return result_matrix
end

@inline function phase1_kai(A::Matrix{T}, method::Function)::Matrix{T} where {T<:Real}
    m, n = size(A)

    # Phase 1
    Wᶜ = Matrix{T}(undef, m, n)
    for k = 1:n
        removed_matrix = remove_row_col(A, k, k)
        W = method(removed_matrix)

        # Wᶜₖを1として挿入
        insert!(W, k, 1.0)
        Wᶜ[:, k] = W
    end

    return Wᶜ
end

# Phase3の戻り値
phase3_jump_mmrd_result2 = @NamedTuple{
    μₖ⃰::T, l⃰::Vector{T}
} where {T<:Real}

# Phase2のループの中の部分
@inline function phase2_jump_amrd(A::Matrix{T}, Wᶜ::Matrix{T}, k::Int, n::Int)::T where {T<:Real}
    ε = 1e-6
    tolerance = 1e-6  # 許容誤差

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, L[i=1:n] ≥ 0)
        @variable(model, t ≥ 1 + ε)
        @variable(model, D[i=1:n, j=1:n] ≥ 0)
        Lₖ = L[k]

        for j = filter(j -> j != k, 1:n)
            for i = filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᶜ = Wᶜ[i, k]
                wⱼᶜ = Wᶜ[j, k]
                Lᵢ = L[i]
                Lⱼ = L[j]
                @constraint(model, sqrt(aᵢⱼ) * (wⱼᶜ - Lⱼ) + D[i, j] == sqrt(1 / aᵢⱼ) * (wᵢᶜ + Lᵢ))
                # 等式制約の誤差許容範囲toleranceを設定
                # @constraint(model, sqrt(aᵢⱼ) * (wⱼᶜ - Lⱼ) + D[i, j] <= sqrt(1 / aᵢⱼ) * (wᵢᶜ + Lᵢ) + ε)
                # @constraint(model, sqrt(aᵢⱼ) * (wⱼᶜ - Lⱼ) + D[i, j] >= sqrt(1 / aᵢⱼ) * (wᵢᶜ + Lᵢ) - ε)
            end
        end

        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            Lⱼ = L[j]
            wⱼᶜ = Wᶜ[j, k]
            @constraint(model, sqrt(aₖⱼ) * (wⱼᶜ - Lⱼ) + D[k, j] == sqrt(1 / aₖⱼ) * (t - 1 + Lₖ))
            # 等式制約の誤差許容範囲toleranceを設定
            # @constraint(model, sqrt(aₖⱼ) * (wⱼᶜ - Lⱼ) + D[k, j] <= sqrt(1 / aₖⱼ) * (t - 1 + Lₖ) + ε)
            # @constraint(model, sqrt(aₖⱼ) * (wⱼᶜ - Lⱼ) + D[k, j] >= sqrt(1 / aₖⱼ) * (t - 1 + Lₖ) - ε)
        end

        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i, k]
            Lᵢ = L[i]
            wᵢᶜ = Wᶜ[i, k]
            @constraint(model, sqrt(aᵢₖ) * (t - 1 - Lₖ) + D[i, k] == sqrt(1 / aᵢₖ) * (wᵢᶜ + Lᵢ))
            # 等式制約の誤差許容範囲toleranceを設定
            # @constraint(model, sqrt(aᵢₖ) * (t - 1 - Lₖ) + D[i, k] <= sqrt(1 / aᵢₖ) * (wᵢᶜ + Lᵢ) + ε)
            # @constraint(model, sqrt(aᵢₖ) * (t - 1 - Lₖ) + D[i, k] >= sqrt(1 / aᵢₖ) * (wᵢᶜ + Lᵢ) - ε)
        end

        # 正規性条件
        # Σᵢ≠ⱼLᵢ - Lⱼ ≥ 0, j ∈ N 
        for j in 1:n
            @constraint(model, sum(L[i] for i in 1:n if i != j) - L[j] ≥ 0)
        end

        for i = filter(i -> i != k, 1:n)
            @constraint(model, Wᶜ[i, k] - L[i] >= ε)
        end
        @constraint(model, (t - 1) - Lₖ ≥ ε)

        # 目的関数
        @objective(model, Min, sum(D[i, j] for i in filter(x -> x != k, 1:n)
                                   for j in filter(x -> x != i && x != k, 1:n)))

        optimize!(model)
        return objective_value(model)

    finally
        empty!(model)
    end
end

# Phase3のループの中の部分
@inline function phase3_jump_amrd2(A::Matrix{T}, Wᶜ::Matrix{T}, d⃰::T, k::Int, n::Int)::phase3_jump_mmrd_result2{T} where {T<:Real}
    ε = 1e-6
    tolerance = 1e-6  # 許容誤差

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 目的関数をΣᵢⱼdᵢⱼに変更
    # Phase2とは異なり変形なしにLPとなる

    try
        @variable(model, l[i=1:n] ≥ ε)
        @variable(model, d[i=1:n, j=1:n] ≥ 0)
        @variable(model, ε ≤ μₖ ≤ 1 - ε)
        lₖ = l[k]

        # 制約条件を不等式に変更
        # dᵢⱼ = √aⱼᵢ (μₖwᵢᶜ + lᵢ) - √aᵢⱼ(μₖwⱼᶜ - lⱼ) 
        for j in filter(j -> j != k, 1:n)
            for i in filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᶜ = Wᶜ[i, k]
                wⱼᶜ = Wᶜ[j, k]
                lᵢ = l[i]
                lⱼ = l[j]
                dᵢⱼ = d[i, j]
                # @constraint(model, sqrt(aᵢⱼ) * (μₖ * wⱼᶜ - lⱼ) + dᵢⱼ - sqrt(1 / aᵢⱼ) * (μₖ * wᵢᶜ + lᵢ) == 0)
                @constraint(model, sqrt(aᵢⱼ) * (μₖ * wⱼᶜ - lⱼ) + dᵢⱼ - sqrt(1 / aᵢⱼ) * (μₖ * wᵢᶜ + lᵢ) ≤ ε)
                @constraint(model, sqrt(aᵢⱼ) * (μₖ * wⱼᶜ - lⱼ) + dᵢⱼ - sqrt(1 / aᵢⱼ) * (μₖ * wᵢᶜ + lᵢ) ≥ -ε)
            end
        end

        # dₖⱼ = √aⱼₖ (1 - μₖ + lᵢ) - √aₖⱼ(μₖwⱼᶜ - lⱼ)
        for j in filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            lⱼ = l[j]
            wⱼᶜ = Wᶜ[j, k]
            dₖⱼ = d[k, j]
            # @constraint(model, sqrt(aₖⱼ) * (μₖ * wⱼᶜ - lⱼ) + dₖⱼ - sqrt(1 / aₖⱼ) * (1 - μₖ + lₖ) == 0)
            @constraint(model, sqrt(aₖⱼ) * (μₖ * wⱼᶜ - lⱼ) + dₖⱼ - sqrt(1 / aₖⱼ) * (1 - μₖ + lₖ) ≤ ε)
            @constraint(model, sqrt(aₖⱼ) * (μₖ * wⱼᶜ - lⱼ) + dₖⱼ - sqrt(1 / aₖⱼ) * (1 - μₖ + lₖ) ≥ -ε)
        end

        # dᵢₖ = √aₖᵢ (μₖwᵢᶜ + lᵢ) - √aᵢₖ(1 - μₖ - lⱼ) 
        for i in filter(i -> i != k, 1:n)
            aᵢₖ = A[i, k]
            lᵢ = l[i]
            wᵢᶜ = Wᶜ[i, k]
            dᵢₖ = d[i, k]
            # @constraint(model, sqrt(aᵢₖ) * (1 - μₖ - lₖ) + dᵢₖ - sqrt(1 / aᵢₖ) * (μₖ * wᵢᶜ + lᵢ) == 0)
            @constraint(model, sqrt(aᵢₖ) * (1 - μₖ - lₖ) + dᵢₖ - sqrt(1 / aᵢₖ) * (μₖ * wᵢᶜ + lᵢ) ≤ ε)
            @constraint(model, sqrt(aᵢₖ) * (1 - μₖ - lₖ) + dᵢₖ - sqrt(1 / aᵢₖ) * (μₖ * wᵢᶜ + lᵢ) ≥ -ε)
        end

        # 正規性条件
        # Σᵢ≠ⱼlᵢ - lⱼ ≥ 0, j ∈ N 
        for j in 1:n
            @constraint(model, sum(l[i] for i in 1:n if i != j) - l[j] ≥ 0)
        end

        # μₖwᵢᶜ - lᵢ > 0, i ∈ N\{k}
        for i in filter(x -> x != k, 1:n)
            @constraint(model, μₖ * Wᶜ[i, k] - l[i] ≥ ε)
        end

        # (1 - μₖ) - lₖ > 0
        @constraint(model, (1 - μₖ) - lₖ ≥ ε)

        # phase2の最適性条件
        @constraint(model, sum(d[i, j] for i in filter(x -> x != k, 1:n)
                               for j in filter(x -> x != i && x != k, 1:n)) ≤ μₖ * d⃰ + ε)

        # 目的関数
        @objective(model, Min, sum(d))

        optimize!(model)

        return (
            μₖ⃰=value(μₖ),
            l⃰=value.(l)
        )

    finally
        empty!(model)
    end
end

# eAMRd/c法, gAMRd/c法 (step3の目的関数を変更したバージョン)
@inline function xAMRdc2(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T<:Real}
    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    Wᶜ = phase1_kai(A, method)

    # Phase 2
    d⃰ = Vector{T}(undef, n)
    for k = 1:n
        d⃰[k] = phase2_jump_amrd(A, Wᶜ, k, n)
    end

    # Phase 3
    wᴸ = Matrix{T}(undef, m, n)
    wᵁ = Matrix{T}(undef, m, n)
    centers = Matrix{T}(undef, m, n)
    l = Matrix{T}(undef, m, n)

    for k = 1:n
        (μₖ⃰, l⃰) = phase3_jump_amrd2(A, Wᶜ, d⃰[k], k, n)

        for i = 1:n
            lᵢ⃰ = l⃰[i]
            wᵢᶜ = Wᶜ[i, k]
            if i == k
                wᴸᵢ = (1 - μₖ⃰) - lᵢ⃰
                wᵁᵢ = (1 - μₖ⃰) + lᵢ⃰
                centers[i, k] = 1 - μₖ⃰
            else
                wᴸᵢ = μₖ⃰ * wᵢᶜ - lᵢ⃰
                wᵁᵢ = μₖ⃰ * wᵢᶜ + lᵢ⃰
                centers[i, k] = μₖ⃰ * wᵢᶜ
            end
            l[i, k] = lᵢ⃰
            wᴸ[i, k] = wᴸᵢ
            wᵁ[i, k] = wᵁᵢ
        end
    end

    # Phase 4
    w̄ᴸ = Vector{T}(undef, n)
    w̄ᵁ = Vector{T}(undef, n)
    W̄ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̄ᴸ[i] = mean(wᴸ[i, :])
        w̄ᵁ[i] = mean(wᵁ[i, :])
        if w̄ᴸ[i] > w̄ᵁ[i]
            w̄ᴸ[i] = w̄ᵁ[i]
        end

    end

    for i = 1:n
        W̄[i] = w̄ᴸ[i] .. w̄ᵁ[i]
    end

    return (
        # s=w_c,
        # centers=centers, l=l,
        wᴸ=w̄ᴸ, wᵁ=w̄ᵁ,
        W=W̄
    )
end

