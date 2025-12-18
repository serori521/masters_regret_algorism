"""
AMRw/c method for a crisp PCM
use: AMRwc(PCM)
"""

using IntervalArithmetic
using IntervalArithmetic.Symbols
using JuMP
import HiGHS
using Statistics

include("./crisp-pcm.jl")
include("./nearly-equal.jl")


LPResult_Individual = @NamedTuple{
    # interval weight vector
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
} where {T<:Real}

# Phase1
@inline function AMRwc_phase1(A::Matrix{T}, k::Int, n::Int)::T where {T<:Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 目的関数をΣᵢ≠ₖ(wᵢᵁ-wᵢᴸ)/Σᵢ≠ₖ(wᵢᵁ+wᵢᴸ)/2 に変更
    # t=1/Σᵢ≠ₖ(wᵢᵁ+wᵢᴸ)/2 とおいて, vᵢᵁ:=t*wᵢᵁ, vᵢᴸ:=t*wᵢᴸ としてLP化

    try
        @variable(model, vᵁ[i=1:n] ≥ ε)
        @variable(model, vᴸ[i=1:n] ≥ ε)
        @variable(model, t ≥ ε)

        # aᵢⱼ ∈ Wᵢ/Wⱼ
        for i = 1:n
            for j = filter(j -> j != i, 1:n)
                aᵢⱼ = A[i, j]
                vᵢᵁ = vᵁ[i]
                vⱼᴸ = vᴸ[j]
                @constraint(model, aᵢⱼ * vⱼᴸ ≤ vᵢᵁ)
            end
        end

        # normality condition
        for j = 1:n
            vⱼᴸ = vᴸ[j]
            vⱼᵁ = vᵁ[j]
            # Sᵁ = Σvᵢᵁ, (i≠j)
            # Sᴸ = Σvᵢᴸ, (i≠j)
            Sᵁ = sum(vᵁ[i] for i in 1:n if i != j)
            Sᴸ = sum(vᴸ[i] for i in 1:n if i != j)
            @constraint(model, Sᵁ + vⱼᴸ ≥ t)
            @constraint(model, Sᴸ + vⱼᵁ ≤ t)
        end

        for i = 1:n
            vᵢᵁ = vᵁ[i]
            vᵢᴸ = vᴸ[i]
            @constraint(model, vᵢᵁ - vᵢᴸ ≥ 0)
        end

        @constraint(model, sum(vᵁ) + sum(vᴸ) == 2 * t)

        cₖ = sum(map(i -> (vᵁ[i] + vᴸ[i]) / 2, filter(i -> i != k, 1:n)))
        @constraint(model, cₖ == 1)

        dₖ = sum(map(i -> (vᵁ[i] - vᴸ[i]), filter(i -> i != k, 1:n)))
        @objective(model, Min, dₖ)

        optimize!(model)
        d̂ₖ = sum(map(j -> (value.(vᵁ[j]) - value.(vᴸ[j])), filter(j -> j != k, 1:n)))

        return d̂ₖ

    finally
        empty!(model)
    end
end

# Phase2
@inline function AMRwc_phase2(A::Matrix{T}, d̂::T, k::Int, n::Int)::T where {T<:Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, wᵁ[i=1:n] ≥ ε)
        @variable(model, wᴸ[i=1:n] ≥ ε)

        # aᵢⱼ ∈ Wᵢ/Wⱼ
        for i = 1:n
            for j = filter(j -> j != i, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᵁ = wᵁ[i]
                wⱼᴸ = wᴸ[j]
                @constraint(model, aᵢⱼ * wⱼᴸ ≤ wᵢᵁ)
            end
        end

        # normality condition
        for j = 1:n
            wⱼᴸ = wᴸ[j]
            wⱼᵁ = wᵁ[j]
            # Sᵁ = Σwᵢᵁ, (i≠j)
            # Sᴸ = Σwᵢᴸ, (i≠j)
            Sᵁ = sum(wᵁ[i] for i in 1:n if i != j)
            Sᴸ = sum(wᴸ[i] for i in 1:n if i != j)
            @constraint(model, Sᵁ + wⱼᴸ ≥ 1)
            @constraint(model, Sᴸ + wⱼᵁ ≤ 1)
        end

        for i = 1:n
            wᵢᵁ = wᵁ[i]
            wᵢᴸ = wᴸ[i]
            @constraint(model, wᵢᵁ - wᵢᴸ ≥ 0)
        end

        @constraint(model, sum(wᵁ) + sum(wᴸ) == 2)

        dₖ = sum(map(i -> (wᵁ[i] - wᴸ[i]), filter(i -> i != k, 1:n)))
        #@constraint(model, dₖ ≤ d̂ + ε)
        cₖ = sum(map(i -> (wᵁ[i] + wᴸ[i]) / 2, filter(i -> i != k, 1:n)))
        @constraint(model, dₖ ≤ d̂ * cₖ + ε)

        @objective(model, Min, sum(wᵁ) - sum(wᴸ))

        optimize!(model)
        dd̂ₖ = objective_value(model)

        return dd̂ₖ

    finally
        empty!(model)
    end
end

phase3_AMRwc_result = @NamedTuple{
    # 区間重みベクトル
    kwᴸ::Vector{T}, kwᵁ::Vector{T},
} where {T<:Real}

# Phase3_max
@inline function AMRwc_phase3_max(A::Matrix{T}, d̂::T, dd̂::T, k::Int, n::Int)::phase3_AMRwc_result{T} where {T<:Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, wᵁ[i=1:n] ≥ ε)
        @variable(model, wᴸ[i=1:n] ≥ ε)

        # aᵢⱼ ∈ Wᵢ/Wⱼ
        for i = 1:n
            for j = filter(j -> j != i, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᵁ = wᵁ[i]
                wⱼᴸ = wᴸ[j]
                @constraint(model, aᵢⱼ * wⱼᴸ ≤ wᵢᵁ)
            end
        end

        # normality condition
        for j = 1:n
            wⱼᴸ = wᴸ[j]
            wⱼᵁ = wᵁ[j]
            # Sᵁ = Σwᵢᵁ, (i≠j)
            # Sᴸ = Σwᵢᴸ, (i≠j)
            Sᵁ = sum(wᵁ[i] for i in 1:n if i != j)
            Sᴸ = sum(wᴸ[i] for i in 1:n if i != j)
            @constraint(model, Sᵁ + wⱼᴸ ≥ 1)
            @constraint(model, Sᴸ + wⱼᵁ ≤ 1)
        end

        for i = 1:n
            wᵢᵁ = wᵁ[i]
            wᵢᴸ = wᴸ[i]
            @constraint(model, wᵢᵁ - wᵢᴸ ≥ 0)
        end

        @constraint(model, sum(wᵁ) + sum(wᴸ) == 2)

        dₖ = sum(map(i -> (wᵁ[i] - wᴸ[i]), filter(i -> i != k, 1:n)))
        #@constraint(model, dₖ ≤ d̂ + ε)
        cₖ = sum(map(i -> (wᵁ[i] + wᴸ[i]) / 2, filter(i -> i != k, 1:n)))
        @constraint(model, dₖ ≤ d̂ * cₖ + ε)

        @constraint(model, sum(wᵁ) - sum(wᴸ) ≤ dd̂ + ε)

        @objective(model, Max, wᵁ[k])

        optimize!(model)
        uwᴸ = value.(wᴸ)
        uwᵁ = value.(wᵁ)

        return uwᴸ, uwᵁ

    finally
        empty!(model)
    end
end

# Phase3_min
@inline function AMRwc_phase3_min(A::Matrix{T}, d̂::T, dd̂::T, k::Int, n::Int)::phase3_AMRwc_result{T} where {T<:Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, wᵁ[i=1:n] ≥ ε)
        @variable(model, wᴸ[i=1:n] ≥ ε)

        # aᵢⱼ ∈ Wᵢ/Wⱼ
        for i = 1:n
            for j = filter(j -> j != i, 1:n)
                aᵢⱼ = A[i, j]
                wᵢᵁ = wᵁ[i]
                wⱼᴸ = wᴸ[j]
                @constraint(model, aᵢⱼ * wⱼᴸ ≤ wᵢᵁ)
            end
        end

        # normality condition
        for j = 1:n
            wⱼᴸ = wᴸ[j]
            wⱼᵁ = wᵁ[j]
            # Sᵁ = Σwᵢᵁ, (i≠j)
            # Sᴸ = Σwᵢᴸ, (i≠j)
            Sᵁ = sum(wᵁ[i] for i in 1:n if i != j)
            Sᴸ = sum(wᴸ[i] for i in 1:n if i != j)
            @constraint(model, Sᵁ + wⱼᴸ ≥ 1)
            @constraint(model, Sᴸ + wⱼᵁ ≤ 1)
        end

        for i = 1:n
            wᵢᵁ = wᵁ[i]
            wᵢᴸ = wᴸ[i]
            @constraint(model, wᵢᵁ - wᵢᴸ ≥ 0)
        end

        @constraint(model, sum(wᵁ) + sum(wᴸ) == 2)

        dₖ = sum(map(i -> (wᵁ[i] - wᴸ[i]), filter(i -> i != k, 1:n)))
        #@constraint(model, dₖ ≤ d̂ + ε)
        cₖ = sum(map(i -> (wᵁ[i] + wᴸ[i]) / 2, filter(i -> i != k, 1:n)))
        @constraint(model, dₖ ≤ d̂ * cₖ + ε)

        @constraint(model, sum(wᵁ) - sum(wᴸ) ≤ dd̂ + ε)

        @objective(model, Min, wᴸ[k])

        optimize!(model)
        lwᴸ = value.(wᴸ)
        lwᵁ = value.(wᵁ)

        return lwᴸ, lwᵁ

    finally
        empty!(model)
    end
end

# 推定法の関数
@inline function AMRwc(A::Matrix{T})::LPResult_Individual{T} where {T<:Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    # k番目を除いた幅総和最小化
    d̂ = Vector{T}(undef, n)
    for k = 1:n
        d̂[k] = AMRwc_phase1(A, k, n)
    end



    # Phase 2
    # k番目の幅最小化
    dd̂ = Vector{T}(undef, n)
    for k = 1:n
        dd̂[k] = AMRwc_phase2(A, d̂[k], k, n)
    end

    # Phase 3
    kwᴸ = Matrix{T}(undef, m, 2 * n)
    kwᵁ = Matrix{T}(undef, m, 2 * n)
    for k = 1:n
        uwᴸ, uwᵁ = AMRwc_phase3_max(A, d̂[k], dd̂[k], k, n)
        for j = 1:n
            kwᴸ[j, k] = uwᴸ[j]
            kwᵁ[j, k] = uwᵁ[j]
        end
        lwᴸ, lwᵁ = AMRwc_phase3_min(A, d̂[k], dd̂[k], k, n)
        for j = 1:n
            kwᴸ[j, k+n] = lwᴸ[j]
            kwᵁ[j, k+n] = lwᵁ[j]
        end
    end

    # Phase 4
    w̅̅ᴸ = Vector{T}(undef, n)
    w̅̅ᵁ = Vector{T}(undef, n)
    W̅̅ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̅̅ᴸ[i] = mean(kwᴸ[i, :])
        w̅̅ᵁ[i] = mean(kwᵁ[i, :])

        # precision error 対応
        if w̅̅ᴸ[i] > w̅̅ᵁ[i]
            w̅̅ᴸ[i] = w̅̅ᵁ[i]
        end

        W̅̅[i] = (w̅̅ᴸ[i]) .. (w̅̅ᵁ[i])
    end

    return (
        wᴸ=w̅̅ᴸ, wᵁ=w̅̅ᵁ,
        W=W̅̅
    )

end