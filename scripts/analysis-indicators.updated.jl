using IntervalArithmetic
using IntervalArithmetic.Symbols
using DataFrames
using Statistics
include("./crisp-pcm.jl")

# 区間の交差を計算する関数
@inline function interval_intersect(a::Interval{T}, b::Interval{T}) where T
    l = max(inf(a), inf(b))
    u = min(sup(a), sup(b))
    if l <= u
        return interval(l, u)
    else
        return interval(0.0, 0.0)  # 空集合の代わりに幅0の区間を返す
    end
end

# ベクトル化された交差演算
@inline function vector_intersect(T::Vector{Interval{Float64}}, E::Vector{Interval{Float64}})
    return [interval_intersect(t, e) for (t, e) in zip(T, E)]
end

# ベクトル化されたhull演算
@inline function vector_hull(T::Vector{Interval{Float64}}, E::Vector{Interval{Float64}})
    return [interval(min(inf(t), inf(e)), max(sup(t), sup(e))) for (t, e) in zip(T, E)]
end

# データフレームを分割する関数
@inline function split_dataframe(df, chunk_size)
    n = nrow(df)
    m = div(n, chunk_size)
    subdfs = []
    for i in 1:chunk_size:n
        push!(subdfs, df[i:min(i+chunk_size-1, n), :])
    end
    return subdfs
end

@inline function CI(A::Matrix{T})::T where {T <: Real}
    m, n = size(A)

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    λₘₐₓ = maximum(real(eigen(A).values))

    return CI = (λₘₐₓ - n) / (n - 1)

end

# Interval が空集合の場合は幅0を返す
@inline function c_diam(interval)
    if isempty_interval(interval)
        return 0.0
    else
        return diam(interval)
    end
end

# P値
@inline function calculate_P(T, E)
    TcapE = vector_intersect(T, E)
    TcupE = vector_hull(T, E)
    P = c_diam.(TcapE) ./ c_diam.(TcupE)
    return P
end

# Q値
@inline function calculate_Q(T, E)
    TcapE = vector_intersect(T, E)
    Q = c_diam.(TcapE) ./ c_diam.(T)
    return Q
end

# R値
@inline function calculate_R(T, E)
    TcapE = vector_intersect(T, E)
    R = Float64[]
    for i in eachindex(T)
        if c_diam(E[i])==0.0
            if issubset_interval(E[i],T[i])
                push!(R, 1.0)
            else
                push!(R, 0.0)
            end
        else
            push!(R, c_diam(TcapE[i]) / c_diam(E[i]))
        end
    end
    # R = c_diam.(TcapE) ./ c_diam.(E)
    return R
end

# F値
@inline function calculate_F(T, E)
    Qv = calculate_Q(T, E)
    Rv = calculate_R(T, E)
    denominator = Qv .+ Rv
    # 分母が 0 でない場合のみ計算を行う
    F = ifelse.(denominator .== 0, 0.0, 2 * (Qv .* Rv) ./ denominator)
    return F
end

# crispな推定値がTの範囲に含まれているか
@inline function est_in_range(T, E)
    n = length(E)
    cnt = 0
    for i in 1:n
        if E[i] in T[i]
            cnt += 1
        end
    end 
    return cnt
end

# intervalの中心を計算する
@inline function interval_centers(intervals::Vector{Interval{Float64}})::Vector{Float64}
    return [(inf(interval) + sup(interval)) / 2 for interval in intervals]
end

# 2つのベクトルのユークリッド距離を計算する
@inline function calculate_euclidean(v1, v2)
    dist = sqrt(sum((v1 .- v2).^2))
    return dist
end

@inline function calculate_manhattan(v1, v2)
    dist =  sum(abs.(v1.-v2))
    return dist
end

@inline function _interval_vector(w_std)
    if w_std isa Vector{Interval{Float64}}
        return w_std
    elseif w_std isa NamedTuple && haskey(w_std, :L) && haskey(w_std, :R)
        return interval.(w_std.L, w_std.R)
    elseif w_std isa Tuple && length(w_std) == 2
        return interval.(w_std[1], w_std[2])
    else
        throw(ArgumentError("w_std must be Vector{Interval} or provide L/R vectors"))
    end
end

@inline function _aggregate_score(v::Vector{Float64}, agg::Symbol)
    if agg == :mean
        return mean(v)
    elseif agg == :sum
        return sum(v)
    else
        throw(ArgumentError("agg must be :mean or :sum"))
    end
end

"""
    best_ts_grid(w_std_true, w_std_est, tL, tU, tsL, tsU; Nt=50, Nts=50, metric=:F, agg=:mean)

Grid-search ts* for each t to maximize PQRF-based score between T(t)=t*w_std_true and
E(ts)=ts*w_std_est. Returns (ts_star_list, r_list).
"""
function best_ts_grid(w_std_true, w_std_est, tL, tU, tsL, tsU; Nt=50, Nts=50, metric=:F, agg=:mean)
    w_true = _interval_vector(w_std_true)
    w_est  = _interval_vector(w_std_est)

    t_grid  = range(tL,  tU;  length=Nt)
    ts_grid = range(tsL, tsU; length=Nts)

    ts_star_list  = Vector{Float64}(undef, length(t_grid))
    r_list        = Vector{Float64}(undef, length(t_grid))

    # 追加で保存する統計（基準ごとの score_vec を t ごとに集約したもの）
    best_obj_list  = Vector{Float64}(undef, length(t_grid))   # agg(:mean/:sum) で使った目的関数値
    best_mean_list = Vector{Float64}(undef, length(t_grid))   # mean(score_vec)
    best_var_list  = Vector{Float64}(undef, length(t_grid))   # var(score_vec)   (corrected=false)
    best_sd_list   = Vector{Float64}(undef, length(t_grid))   # std(score_vec)   (corrected=false)
    best_min_list  = Vector{Float64}(undef, length(t_grid))   # minimum(score_vec)

    denom = tsU - tsL

    @inbounds for (ti, t) in enumerate(t_grid)
        T = t .* w_true

        best_ts   = ts_grid[1]
        best_obj  = -Inf
        best_mean = NaN
        best_var  = NaN
        best_sd   = NaN
        best_min  = NaN

        for ts in ts_grid
            E = ts .* w_est

            score_vec =
                if metric == :P
                    calculate_P(T, E)
                elseif metric == :Q
                    calculate_Q(T, E)
                elseif metric == :R
                    calculate_R(T, E)
                elseif metric == :F
                    calculate_F(T, E)
                else
                    throw(ArgumentError("metric must be one of :P, :Q, :R, :F"))
                end

            obj = _aggregate_score(score_vec, agg)

            # NaN を混ぜたくないので、目的関数が有限のときだけ更新
            if isfinite(obj) && obj > best_obj
                best_obj = obj
                best_ts  = ts

                # 統計は「基準（N個）」方向で計算（1つの t に対して N 個のスコアがある）
                finite_vec = filter(isfinite, score_vec)
                if isempty(finite_vec)
                    best_mean = NaN
                    best_var  = NaN
                    best_sd   = NaN
                    best_min  = NaN
                else
                    best_mean = mean(finite_vec)
                    best_var  = var(finite_vec; corrected=false)
                    best_sd   = std(finite_vec; corrected=false)
                    best_min  = minimum(finite_vec)
                end
            end
        end

        ts_star_list[ti]   = best_ts
        r_list[ti]         = denom == 0.0 ? 0.0 : (best_ts - tsL) / denom
        best_obj_list[ti]  = best_obj
        best_mean_list[ti] = best_mean
        best_var_list[ti]  = best_var
        best_sd_list[ti]   = best_sd
        best_min_list[ti]  = best_min
    end

    return ts_star_list, r_list, best_obj_list, best_mean_list, best_var_list, best_sd_list, best_min_list
end
