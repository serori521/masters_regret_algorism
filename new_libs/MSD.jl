"""
MSD method for a crisp PCM
use: MSD(PCM)
"""

using IntervalArithmetic
using IntervalArithmetic.Symbols
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")
include("./solve-deterministic-ahp.jl")


LPResult_Individual = @NamedTuple{
    # interval weight vector
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
} where {T<:Real}

@inline function MSD(A::Matrix{T})::LPResult_Individual{T} where {T<:Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, wᵁ[i=1:n] ≥ ε)
        @variable(model, wᴸ[i=1:n] ≥ ε)
        @variable(model, d[1:n, 1:n] ≥ 0)

        # dᵢⱼ = sqrt(aⱼᵢ)*wⱼᵁ - sqrt(aᵢⱼ)*wᵢᴸ, i,j=1,...,n,(i != j)
        for j = 1:n
            for i = filter(i -> i != j, 1:n)
                aᵢⱼ = A[i, j]
                aⱼᵢ = A[j, i]
                wⱼᵁ = wᵁ[j]
                wᵢᴸ = wᴸ[i]
                dᵢⱼ = d[i, j]
                @constraint(model, dᵢⱼ == sqrt(aᵢⱼ) * wⱼᵁ - sqrt(aⱼᵢ) * wᵢᴸ)
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

        dv = sum(d[i, j] for i in 1:n, j in 1:n if i != j)

        @objective(model, Min, dv)

        optimize!(model)

        Wᵁ = value.(wᵁ)
        Wᴸ = value.(wᴸ)
        W = Vector{Interval{T}}(undef, n)

        for i = 1:n
            # precision error correction
            if Wᴸ[i] > Wᵁ[i]
                Wᴸ[i] = Wᵁ[i]
            end
            W[i] = (Wᴸ[i]) .. (Wᵁ[i])
        end

        return (
            wᴸ=Wᴸ, wᵁ=Wᵁ, W=W
        )

    finally
        empty!(model)
    end

end