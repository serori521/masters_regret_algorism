"""
区間重要度推定法 eAMRw/c, gAMRw/c の関数
xAMRwc(PCM, method)とすることで，区間重要度が求められる．
In: PCM:Matrix型, method:EV or GM
Out:LPResult_Individual型(下で定義)
"""

using IntervalArithmetic
using IntervalArithmetic.Symbols
using JuMP
import HiGHS
using Statistics

include("./crisp-pcm.jl")
include("./nearly-equal.jl")
include("./solve-deterministic-ahp.jl")


LPResult_Individual = @NamedTuple{
    # 区間重要度ベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
} where {T<:Real}

# 指定した行と列を削除した行列を返す関数
@inline function remove_row_col(A::Matrix{T}, row::Int, col::Int)::Matrix{T} where {T<:Real}
    m, n = size(A)

    # 行を除外
    new_matrix = A[setdiff(1:m, row), :]
    # 列を除外
    result_matrix = new_matrix[:, setdiff(1:n, col)]

    return result_matrix
end

# Phase1
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
            @constraint(model, wᵢᶜ - Lᵢ ≥ ε)
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
phase3_jump_result_kai = @NamedTuple{
    # 区間重みベクトル
    tₖ⃰::T, L⃰::Vector{T},
} where {T<:Real}

# Phase3のループの中の部分
@inline function phase3_jump_kai(A::Matrix{T}, Wᶜ::Matrix{T}, d⃰::T, k::Int, n::Int)::phase3_jump_result_kai{T} where {T<:Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 目的関数をlₖ/(1-μₖ)に変更
    # t=1/(1-μₖ) とおいて, Lᵢ:=t*lᵢとしてLP化
    # μₖ/(1-μₖ)の箇所は 1/(1-μₖ) - 1 に変形でき, t-1になる

    try
        @variable(model, L[i=1:n] ≥ 0)
        @variable(model, t ≥ 1 + ε)
        Lₖ = L[k]

        # aᵢⱼ((t - 1)wⱼᶜ - Lⱼ) ≤ (t - 1)wᵢᶜ + Lᵢ, i,j ∈ N∖{k}, i≠j
        for j = filter(j -> j != k, 1:n)
            for i = filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᶜ = Wᶜ[i, k]
                wⱼᶜ = Wᶜ[j, k]
                Lᵢ = L[i]
                Lⱼ = L[j]
                @constraint(model, aᵢⱼ * ((t - 1) * wⱼᶜ - Lⱼ) ≤ (t - 1) * wᵢᶜ + Lᵢ)
            end
        end

        # aₖⱼ((t - 1)wⱼᶜ - Lⱼ) ≤ 1 + Lₖ, j∈N∖{k}
        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            Lⱼ = L[j]
            wⱼᶜ = Wᶜ[j, k]
            @constraint(model, aₖⱼ * ((t - 1) * wⱼᶜ - Lⱼ) ≤ 1 + Lₖ)
        end

        # aᵢₖ(1 - Lₖ) ≤ (t - 1)wᵢᶜ + Lᵢ, i∈N∖{k}
        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i, k]
            Lᵢ = L[i]
            wᵢᶜ = Wᶜ[i, k]
            @constraint(model, aᵢₖ * (1 - Lₖ) ≤ (t - 1) * wᵢᶜ + Lᵢ)
            # (t - 1)wᵢᶜ - Lᵢ ≥ ε, i∈N∖{k}
            @constraint(model, (t - 1) * wᵢᶜ - Lᵢ ≥ ε)
        end

        # 1 - Lₖ ≥ ε
        @constraint(model, 1 - Lₖ ≥ ε)

        # 正規性条件
        # Σᵢ≠ⱼLᵢ - Lⱼ ≥ 0, j ∈ N 
        for j in 1:n
            @constraint(model, sum(L[i] for i in 1:n if i != j) - L[j] ≥ 0)
        end

        # 前のPhaseの最適性条件
        Σl = sum(map(j -> L[j], filter(j -> j != k, 1:n)))
        @constraint(model, Σl ≤ (t - 1) * d⃰ + ε)

        # dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
        @objective(model, Min, L[k])

        optimize!(model)
        # println("k=$k, t = ", value.(t))
        tₖ⃰ = value.(t)
        L⃰ = value.(L)
        return (
            tₖ⃰=tₖ⃰,
            L⃰=L⃰
        )

    finally
        empty!(model)
    end
end

# eAMRw/c, gAMRw/c
@inline function xAMRwc(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T<:Real}

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

    # Phase 3
    wᴸ = Matrix{T}(undef, m, n)
    wᵁ = Matrix{T}(undef, m, n)
    centers = Matrix{T}(undef, m, n)
    l = Matrix{T}(undef, m, n)

    for k = 1:n
        (tₖ⃰, L⃰) = phase3_jump_kai(A, Wᶜ, d⃰[k], k, n)

        # t=1/(1-μₖ), Lᵢ=t*lᵢからμₖ,lᵢに戻す
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
            # 各kの時の推定値を縦ベクトルとして格納
            wᴸ[i, k] = wᴸᵢ
            wᵁ[i, k] = wᵁᵢ
        end
    end


    # Phase 4
    w̅̅ᴸ = Vector{T}(undef, n)
    w̅̅ᵁ = Vector{T}(undef, n)
    W̅̅ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̅̅ᴸ[i] = mean(wᴸ[i, :])
        w̅̅ᵁ[i] = mean(wᵁ[i, :])

        # precision error 対応
        if w̅̅ᴸ[i] > w̅̅ᵁ[i]
            w̅̅ᴸ[i] = w̅̅ᵁ[i]
        end

    end

    for i = 1:n
        W̅̅[i] = (w̅̅ᴸ[i]) .. (w̅̅ᵁ[i])
    end

    return (
        wᴸ=w̅̅ᴸ, wᵁ=w̅̅ᵁ,
        W=W̅̅
    )

end