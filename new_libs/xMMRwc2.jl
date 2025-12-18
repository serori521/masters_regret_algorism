"""
区間重要度推定法 eMMRw/c, gMMRw/c の関数
step3の目的関数を1~n全体を対象に変更
xMMRwc2(PCM, method)とすることで，区間重要度が求められる．
In: PCM:Matrix型, method:EV or GM
Out:LPResult_Individual型(下で定義)
"""

using IntervalArithmetic
using IntervalArithmetic.Symbols
using JuMP
import HiGHS

using Plots
include("./crisp-pcm.jl")
include("./nearly-equal.jl")
include("./solve-deterministic-ahp.jl")


# MMRE_Individual_kai = @NamedTuple{
#     # 区間重みベクトル
#     s::T,
#     centers::Matrix{T},
#     l::Matrix{T},
#     wᴸ::Vector{T}, wᵁ::Vector{T},
#     W::Vector{Interval{T}} # ([wᵢᴸ, wᵢᵁ])
# } where {T<:Real}

LPResult_Individual = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
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

# Phase2のループの中の部分
@inline function phase2_jump_kai(A::Matrix{T}, Wᶜ::Matrix{T}, k::Int, n::Int)::T where {T<:Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 目的関数をΣᵢ≠ₖlᵢ/μₖ に変更
    # t=1/μₖ とおいて, Lᵢ:=t*lᵢとしてLP化

    try
        @variable(model, L[i=1:n] ≥ 0)
        @variable(model, t ≥ 1 + ε)
        Lₖ = L[k]

        # aᵢⱼ(wⱼᶜ - Lⱼ) ≤ wᵢᶜ + Lᵢ, i,j ∈ N∖{k}, i≠j
        for j = filter(j -> j != k, 1:n)
            for i = filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᶜ = Wᶜ[i, k]
                wⱼᶜ = Wᶜ[j, k]
                Lᵢ = L[i]
                Lⱼ = L[j]
                @constraint(model, aᵢⱼ * (wⱼᶜ - Lⱼ) ≤ wᵢᶜ + Lᵢ)
            end
        end

        # aₖⱼ(wⱼᶜ - Lⱼ) ≤ (t - 1) + Lₖ, j∈N∖{k}
        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            Lⱼ = L[j]
            wⱼᶜ = Wᶜ[j, k]
            @constraint(model, aₖⱼ * (wⱼᶜ - Lⱼ) ≤ t - 1 + Lₖ)
        end

        # aᵢₖ * ((t - 1) - Lₖ) ≤ wᵢᶜ + Lᵢ, i∈N∖{k}
        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i, k]
            Lᵢ = L[i]
            wᵢᶜ = Wᶜ[i, k]
            @constraint(model, aᵢₖ * (t - 1 - Lₖ) ≤ wᵢᶜ + Lᵢ)
            # wᵢᶜ - Lᵢ ≥ ε, i∈N∖{k}
            @constraint(model, wᵢᶜ - Lᵢ ≥ t * ε)
        end

        # (t - 1) - Lₖ ≥ ε
        @constraint(model, (t - 1) - Lₖ ≥ ε)

        # 正規性条件
        # Σᵢ≠ⱼLᵢ - Lⱼ ≥ 0, j ∈ N 
        for j in 1:n
            @constraint(model, sum(L[i] for i in 1:n if i != j) - L[j] ≥ 0)
        end

        dₖ = sum(map(j -> L[j], filter(j -> j != k, 1:n)))
        @objective(model, Min, dₖ)

        optimize!(model)

        dₖ⃰ = sum(map(j -> value.(L[j]), filter(j -> j != k, 1:n)))

        return dₖ⃰

    finally
        empty!(model)
    end
end

# Phase3の戻り値
# 変更後
phase3_jump_result_kai2 = @NamedTuple{
    μₖ⃰::T, l⃰::Vector{T},
} where {T<:Real}

@inline function phase3_jump_kai2(A::Matrix{T}, Wᶜ::Matrix{T}, d⃰::T, k::Int, n::Int)::phase3_jump_result_kai2{T} where {T<:Real}
    ε = 1e-6
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 目的関数をΣₖlₖ/{(1-μₖ)+μₖ}, つまり,Σₖlₖに変更
    # Phase2とは異なり変形なしにLPとなる

    try
        @variable(model, l[i=1:n] ≥ ε)
        @variable(model, ε ≤ μₖ ≤ 1 - ε)
        lₖ = l[k]

        # 制約1: aᵢⱼ(μₖwⱼᶜ - lⱼ) ≤ μₖwᵢᶜ + lᵢ, i,j ∈ N∖{k}, i≠j
        for j = filter(j -> j != k, 1:n)
            for i = filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᶜ = Wᶜ[i, k]
                wⱼᶜ = Wᶜ[j, k]
                lᵢ = l[i]
                lⱼ = l[j]
                @constraint(model, aᵢⱼ * (μₖ * wⱼᶜ - lⱼ) ≤ μₖ * wᵢᶜ + lᵢ)
            end
        end

        # 制約2: aₖⱼ(μₖwⱼᶜ - lⱼ) ≤ 1-μₖ + lₖ, j∈N∖{k}
        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            lⱼ = l[j]
            wⱼᶜ = Wᶜ[j, k]
            @constraint(model, aₖⱼ * (μₖ * wⱼᶜ - lⱼ) ≤ 1 - μₖ + lₖ)
        end

        # 制約3: aᵢₖ((1-μₖ) - lₖ) ≤ μₖwᵢᶜ + lᵢ, i∈N∖{k}
        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i, k]
            lᵢ = l[i]
            wᵢᶜ = Wᶜ[i, k]
            @constraint(model, aᵢₖ * (1 - μₖ - lₖ) ≤ μₖ * wᵢᶜ + lᵢ)
            # μₖwᵢᶜ - lᵢ ≥ ε, i∈N∖{k}
            @constraint(model, μₖ * wᵢᶜ - lᵢ ≥ ε)
        end

        # (1 - μₖ) - lₖ ≥ ε
        @constraint(model, (1 - μₖ) - lₖ ≥ ε)

        # 正規性条件
        # Σᵢ≠ⱼlᵢ - lⱼ ≥ 0, j ∈ N 
        for j in 1:n
            @constraint(model, sum(l[i] for i in 1:n if i != j) - l[j] ≥ 0)
        end

        # 前のPhaseの最適性条件 Σⱼ≠ₖlⱼ ≤ μₖd⃰
        Σl = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
        @constraint(model, Σl ≤ μₖ * d⃰ + ε)

        @objective(model, Min, sum(l))

        optimize!(model)
        μₖ⃰ = value(μₖ)
        l⃰ = value.(l)
        return (μₖ⃰=μₖ⃰, l⃰=l⃰)

    finally
        empty!(model)
    end
end

# eMMRw/c, gMMRw/c (step3の目的関数を変更したバージョン)
@inline function xMMRwc2(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T<:Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    Wᶜ = phase1_kai(A, method)

    # Phase 2
    d⃰ = Vector{T}(undef, n)
    for k = 1:n
        d⃰[k] = phase2_jump_kai(A, Wᶜ, k, n)
    end

    # 必要な変数を初期化
    centers = Matrix{T}(undef, n, n)
    l = Matrix{T}(undef, n, n)
    wᴸ = Matrix{T}(undef, n, n)  # 一時的に行列として使用
    wᵁ = Matrix{T}(undef, n, n)  # 一時的に行列として使用

    # Phase 3
    for k = 1:n
        (μₖ⃰, l⃰) = phase3_jump_kai2(A, Wᶜ, d⃰[k], k, n)  # 戻り値の受け取り方を変更

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
    w̅̅ᴸ = Vector{T}(undef, n)
    w̅̅ᵁ = Vector{T}(undef, n)
    W̅̅ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̅̅ᴸ[i] = minimum(wᴸ[i, :])
        w̅̅ᵁ[i] = maximum(wᵁ[i, :])

        # precision error 対応
        if w̅̅ᴸ[i] > w̅̅ᵁ[i]
            w̅̅ᴸ[i] = w̅̅ᵁ[i]
        end
    end

    w_c = sum(w̅̅ᴸ .+ w̅̅ᵁ) / 2

    w̅̅ᴸ = w̅̅ᴸ ./ w_c
    w̅̅ᵁ = w̅̅ᵁ ./ w_c
    for i = 1:n
        W̅̅[i] = (w̅̅ᴸ[i]) .. (w̅̅ᵁ[i])
    end

    return (
        # s=w_c,
        # centers=centers, l=l,
        wᴸ=w̅̅ᴸ, wᵁ=w̅̅ᵁ,
        W=W̅̅
    )
end