using IntervalArithmetic
using IntervalArithmetic.Symbols
using LinearAlgebra

include("./nearly-equal.jl")

@inline function isCrispPCM(A::Matrix{T})::Bool where {T <: Real}
    # Check if the matrix is square
    if size(A, 1) != size(A, 2)
        return false
    end

    # Check reciprocity
    for i in 1:size(A, 1)
        for j in (i+1):size(A, 2)
            if !nearlyEqual(A[i, j]*A[j, i], 1)
                return false
            end
        end
    end

    # Check if the diagonal elements are 1
    if any(diag(A) .!= 1)
        return false
    end

    return true
end

@inline function randomIndex(n::Integer)::Real
    if n < 3
        return 0
    end
    if n == 3
        return 0.58
    end
    if n == 4
        return 0.9
    end
    if n == 5
        return 1.12
    end
    if n == 6
        return 1.24
    end
    if n == 7
        return 1.32
    end
    if n == 8
        return 1.41
    end
    if n == 9
        return 1.45
    end
    if n >= 10
        return 1.49
    end
end

@inline function consistencyIndex(A::Matrix{T})::Real where {T <: Real}
    if (!isCrispPCM(A))
        throw(ArgumentError("A must be crisp PCM."))
    end

    n = size(A, 2)
    λₘₐₓ = maximum(real.(filter(λ  -> isreal(λ), eigvals(A))))
    return (λₘₐₓ - n) / (n - 1)
end


@inline function consistencyRatio(A::Matrix{T})::Real where {T <: Real}
    if (!isCrispPCM(A))
        throw(ArgumentError("A must be crisp PCM."))
    end

    n = size(A, 2)
    return consistencyIndex(A) / randomIndex(n)
end