"""
区間重要度推定法 eMMRd/c, gMMRd/c の関数
xMMRdc(PCM, method)とすることで，区間重要度が求められる．
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
phase3_jump_mmrd_result = @NamedTuple{
    tₖ⃰::T, L⃰::Vector{T}
} where {T<:Real}

# Phase2のループの中の部分
@inline function phase2_jump_mmrd(A::Matrix{T}, Wᶜ::Matrix{T}, k::Int, n::Int)::T where {T<:Real}
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
@inline function phase3_jump_mmrd(A::Matrix{T}, Wᶜ::Matrix{T}, d⃰::T, k::Int, n::Int)::phase3_jump_mmrd_result{T} where {T<:Real}
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
                Dᵢⱼ = D[i, j]
                #@constraint(model, sqrt(aᵢⱼ) * ((t - 1) * wⱼᶜ - Lⱼ) + Dᵢⱼ == sqrt(1 / aᵢⱼ) * ((t - 1) * wᵢᶜ + Lᵢ))
                # 等式制約の誤差許容範囲を10^(-6)に設定
                @constraint(model, sqrt(aᵢⱼ) * ((t - 1) * wⱼᶜ - Lⱼ) + Dᵢⱼ <= sqrt(1 / aᵢⱼ) * ((t - 1) * wᵢᶜ + Lᵢ) + ε)
                @constraint(model, sqrt(aᵢⱼ) * ((t - 1) * wⱼᶜ - Lⱼ) + Dᵢⱼ >= sqrt(1 / aᵢⱼ) * ((t - 1) * wᵢᶜ + Lᵢ) - ε)
            end
        end

        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            wⱼᶜ = Wᶜ[j, k]
            Lⱼ = L[j]
            Dₖⱼ = D[k, j]
            #@constraint(model, sqrt(aₖⱼ) * ((t - 1) * wⱼᶜ - Lⱼ) + Dₖⱼ == sqrt(1 / aₖⱼ) * (1 + Lₖ))
            # 等式制約の誤差許容範囲を10^(-6)に設定
            @constraint(model, sqrt(aₖⱼ) * ((t - 1) * wⱼᶜ - Lⱼ) + Dₖⱼ <= sqrt(1 / aₖⱼ) * (1 + Lₖ) + ε)
            @constraint(model, sqrt(aₖⱼ) * ((t - 1) * wⱼᶜ - Lⱼ) + Dₖⱼ >= sqrt(1 / aₖⱼ) * (1 + Lₖ) - ε)
        end

        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i, k]
            wᵢᶜ = Wᶜ[i, k]
            Lᵢ = L[i]
            Dᵢₖ = D[i, k]
            #@constraint(model, sqrt(aᵢₖ) * (1 - Lₖ) + Dᵢₖ == sqrt(1 / aᵢₖ) * ((t - 1) * wᵢᶜ + Lᵢ))
            # 等式制約の誤差許容範囲を10^(-6)に設定
            @constraint(model, sqrt(aᵢₖ) * (1 - Lₖ) + Dᵢₖ <= sqrt(1 / aᵢₖ) * ((t - 1) * wᵢᶜ + Lᵢ) + ε)
            @constraint(model, sqrt(aᵢₖ) * (1 - Lₖ) + Dᵢₖ >= sqrt(1 / aᵢₖ) * ((t - 1) * wᵢᶜ + Lᵢ) - ε)
        end

        # 正規性条件
        # Σᵢ≠ⱼLᵢ - Lⱼ ≥ 0, j ∈ N 
        for j in 1:n
            @constraint(model, sum(L[i] for i in 1:n if i != j) - L[j] ≥ 0)
        end

        for i = filter(i -> i != k, 1:n)
            @constraint(model, (t - 1) * Wᶜ[i, k] - L[i] >= ε)
        end
        @constraint(model, 1 - Lₖ ≥ ε)

        # 追加の制約条件
        @constraint(model, sum(D[i, j] for i in filter(x -> x != k, 1:n)
                               for j in filter(x -> x != i && x != k, 1:n)) ≤ (t - 1) * d⃰ + ε)

        # 目的関数
        @objective(model, Min, sum(D[k, j] for j in filter(j -> j != k, 1:n)) +
                               sum(D[i, k] for i in filter(i -> i != k, 1:n)))

        optimize!(model)
        return (
            tₖ⃰=value(t),
            L⃰=value.(L)
        )

    finally
        empty!(model)
    end
end

# eMMRd/c法, gMMRd/c法
#@inline function MMRd_kai(A::Matrix{T}, method::Function)::MMRD_Individual_kai{T} where {T<:Real}
@inline function xMMRdc(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T<:Real}
    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    Wᶜ = phase1_kai(A, method)

    # Phase 2
    d⃰ = Vector{T}(undef, n)
    for k = 1:n
        d⃰[k] = phase2_jump_mmrd(A, Wᶜ, k, n)
    end

    # Phase 3
    wᴸ = Matrix{T}(undef, m, n)
    wᵁ = Matrix{T}(undef, m, n)
    centers = Matrix{T}(undef, m, n)
    l = Matrix{T}(undef, m, n)

    for k = 1:n
        (tₖ⃰, L⃰) = phase3_jump_mmrd(A, Wᶜ, d⃰[k], k, n)

        for i = 1:n
            lᵢ⃰ = L⃰[i] / tₖ⃰
            wᵢᶜ = Wᶜ[i, k]
            if i == k
                wᴸᵢ = 1 / tₖ⃰ - lᵢ⃰
                wᵁᵢ = 1 / tₖ⃰ + lᵢ⃰
                centers[i, k] = 1 / tₖ⃰
                l[i, k] = lᵢ⃰
            else
                wᴸᵢ = (1 - 1 / tₖ⃰) * wᵢᶜ - lᵢ⃰
                wᵁᵢ = (1 - 1 / tₖ⃰) * wᵢᶜ + lᵢ⃰
                centers[i, k] = (1 - 1 / tₖ⃰) * wᵢᶜ
                l[i, k] = lᵢ⃰
            end
            wᴸ[i, k] = wᴸᵢ
            wᵁ[i, k] = wᵁᵢ
        end
    end

    # Phase 4 (eMMRd/c)
    w̄ᴸ = Vector{T}(undef, n)
    w̄ᵁ = Vector{T}(undef, n)
    W̄ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̄ᴸ[i] = minimum(wᴸ[i, :])
        w̄ᵁ[i] = maximum(wᵁ[i, :])
    end

    w_c = sum(w̄ᴸ .+ w̄ᵁ) / 2

    w̄ᴸ = w̄ᴸ ./ w_c
    w̄ᵁ = w̄ᵁ ./ w_c
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

