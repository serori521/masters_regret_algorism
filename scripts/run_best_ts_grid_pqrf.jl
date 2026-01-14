include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))
include(joinpath(@__DIR__, "..", "new_libs", "analysis-indicators.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using IntervalArithmetic
using Statistics
using Printf


# -------------------------
# Running statistics (Welford) for mean/var/sd without storing all values
# NOTE: var/sd here are "population" (denominator = n), consistent with corrected=false.
# -------------------------
mutable struct RunningStat
    n::Int
    mean::Float64
    m2::Float64
end

RunningStat() = RunningStat(0, 0.0, 0.0)

@inline function update!(rs::RunningStat, x::Float64)
    rs.n += 1
    δ = x - rs.mean
    rs.mean += δ / rs.n
    δ2 = x - rs.mean
    rs.m2 += δ * δ2
    return rs
end

@inline function var_pop(rs::RunningStat)
    rs.n == 0 ? NaN : rs.m2 / rs.n
end

@inline function sd_pop(rs::RunningStat)
    v = var_pop(rs)
    isfinite(v) ? sqrt(v) : NaN
end
# -------------------------
# Config
# -------------------------

const REPEAT_NUM = 1000

const NT = 50
const NTS = 50
const METRIC = :F
const AGG = :mean

function out_csv_path(paths,n,trueW,method)
    outdir = joinpath(paths.data, "metrics_julia/tsresults")
    mkpath(outdir)
    return joinpath(outdir, "best_ts_grid_pqrf_N$(n)_$(trueW)_$(method).csv")
end

function main(N,Tw,method)
    paths = Paths.project_paths()
    N = N
    TRUE_WEIGHT_TYPE = Tw
    METHOD = method
    outpath = out_csv_path(paths,N,TRUE_WEIGHT_TYPE,METHOD)

    trueW = LoadInstance.read_true_weights(paths, TRUE_WEIGHT_TYPE; N=N)
    w_std_true = interval.(trueW.L, trueW.R)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)
    t_grid = range(tL_true, tU_true; length=NT)

    
# -------------------------
# Accumulators for summary (across pcm_id) per t_idx
# -------------------------
stat_ts_by_t      = [RunningStat() for _ in 1:NT]
stat_obj_by_t     = [RunningStat() for _ in 1:NT]
stat_min_by_t     = [RunningStat() for _ in 1:NT]
stat_r_by_t       = [RunningStat() for _ in 1:NT]
count_r0_by_t     = zeros(Int, NT)
count_r1_by_t     = zeros(Int, NT)

# Store raw values for quantiles (50 t_idx × 1000 pcm => OK memory)
ts_vals_by_t  = [Float64[] for _ in 1:NT]
obj_vals_by_t = [Float64[] for _ in 1:NT]
min_vals_by_t = [Float64[] for _ in 1:NT]

# Overall accumulators across all (pcm_id, t_idx)
stat_ts_all  = RunningStat()
stat_obj_all = RunningStat()
stat_min_all = RunningStat()
stat_r_all   = RunningStat()
count_r0_all = 0
count_r1_all = 0
filename = joinpath(TRUE_WEIGHT_TYPE, METHOD)
    methodW = LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    repeat = min(REPEAT_NUM, length(methodW))

    open(outpath, "w") do io
        println(io, "pcm_id,t_idx,t,ts_star,r,best_obj,best_min")

        for pcm_id in 1:repeat
            wL = methodW[pcm_id].L
            wU = methodW[pcm_id].R
            w_std_est = interval.(wL, wU)
            tsL, tsU = SetRegretCore.find_optimal_trange(wL, wU)

            ts_star_list, r_list, best_obj_list, best_min_list = best_ts_grid(
                w_std_true,
                w_std_est,
                tL_true,
                tU_true,
                tsL,
                tsU;
                Nt=NT,
                Nts=NTS,
                metric=METRIC,
                agg=AGG
            )

            for (t_idx, t) in enumerate(t_grid)
                # update summary accumulators (per t_idx and overall)
                ts_val  = Float64(ts_star_list[t_idx])
                r_val   = Float64(r_list[t_idx])
                obj_val = Float64(best_obj_list[t_idx])
                min_val = Float64(best_min_list[t_idx])
                update!(stat_ts_by_t[t_idx], ts_val)
                update!(stat_r_by_t[t_idx],  r_val)
                update!(stat_obj_by_t[t_idx], obj_val)
                update!(stat_min_by_t[t_idx], min_val)
                if r_val == 0.0; count_r0_by_t[t_idx] += 1; end
                if r_val == 1.0; count_r1_by_t[t_idx] += 1; end
                update!(stat_ts_all, ts_val)
                update!(stat_r_all,  r_val)
                update!(stat_obj_all, obj_val)
                update!(stat_min_all, min_val)
                # keep raw values for quantiles
                push!(ts_vals_by_t[t_idx], ts_val)
                push!(obj_vals_by_t[t_idx], obj_val)
                push!(min_vals_by_t[t_idx], min_val)
                if r_val == 0.0; count_r0_all += 1; end
                if r_val == 1.0; count_r1_all += 1; end
                println(io, join([
                    pcm_id,
                    t_idx,
                    @sprintf("%.10f", t),
                    @sprintf("%.10f", ts_star_list[t_idx]),
                    @sprintf("%.10f", r_list[t_idx]),
                    @sprintf("%.10f", best_obj_list[t_idx]),
                    @sprintf("%.10f", best_min_list[t_idx])
                ], ','))
            end
        end
    end

    # -------------------------
# Write summary CSV (across pcm_id) per t_idx
# -------------------------
summary_path = replace(outpath, ".csv" => "_summary.csv")
open(summary_path, "w") do sio
    println(sio, "t_idx,t,count,ts_mean,ts_var,ts_sd,ts_q25,ts_q50,ts_q75,obj_mean,obj_var,obj_sd,obj_q25,obj_q50,obj_q75,min_mean,min_var,min_sd,min_q25,min_q50,min_q75,r_mean,Pr0,Pr1")
    for t_idx in 1:NT
        c = stat_ts_by_t[t_idx].n
        tval = t_grid[t_idx]

        ts_mean = stat_ts_by_t[t_idx].mean
        ts_var  = var_pop(stat_ts_by_t[t_idx])
        ts_sd   = sd_pop(stat_ts_by_t[t_idx])


        # quantiles (ignore non-finite; needs `using Statistics`)
        ts_vec  = filter(isfinite, ts_vals_by_t[t_idx])
        obj_vec = filter(isfinite, obj_vals_by_t[t_idx])
        min_vec = filter(isfinite, min_vals_by_t[t_idx])

        ts_q25  = isempty(ts_vec)  ? NaN : quantile(ts_vec, 0.25)
        ts_q50  = isempty(ts_vec)  ? NaN : quantile(ts_vec, 0.50)
        ts_q75  = isempty(ts_vec)  ? NaN : quantile(ts_vec, 0.75)

        obj_q25 = isempty(obj_vec) ? NaN : quantile(obj_vec, 0.25)
        obj_q50 = isempty(obj_vec) ? NaN : quantile(obj_vec, 0.50)
        obj_q75 = isempty(obj_vec) ? NaN : quantile(obj_vec, 0.75)

        min_q25 = isempty(min_vec) ? NaN : quantile(min_vec, 0.25)
        min_q50 = isempty(min_vec) ? NaN : quantile(min_vec, 0.50)
        min_q75 = isempty(min_vec) ? NaN : quantile(min_vec, 0.75)

        obj_mean = stat_obj_by_t[t_idx].mean
        obj_var  = var_pop(stat_obj_by_t[t_idx])
        obj_sd   = sd_pop(stat_obj_by_t[t_idx])

        min_mean = stat_min_by_t[t_idx].mean
        min_var  = var_pop(stat_min_by_t[t_idx])
        min_sd   = sd_pop(stat_min_by_t[t_idx])

        r_mean = stat_r_by_t[t_idx].mean
        pr0 = c == 0 ? NaN : count_r0_by_t[t_idx] / c
        pr1 = c == 0 ? NaN : count_r1_by_t[t_idx] / c

        println(sio, join([
            string(t_idx),
            @sprintf("%.10f", tval),
            string(c),
            @sprintf("%.10f", ts_mean),
            @sprintf("%.10f", ts_var),
            @sprintf("%.10f", ts_sd),
            @sprintf("%.10f", ts_q25),
            @sprintf("%.10f", ts_q50),
            @sprintf("%.10f", ts_q75),
            @sprintf("%.10f", obj_mean),
            @sprintf("%.10f", obj_var),
            @sprintf("%.10f", obj_sd),
            @sprintf("%.10f", obj_q25),
            @sprintf("%.10f", obj_q50),
            @sprintf("%.10f", obj_q75),
            @sprintf("%.10f", min_mean),
            @sprintf("%.10f", min_var),
            @sprintf("%.10f", min_sd),
            @sprintf("%.10f", min_q25),
            @sprintf("%.10f", min_q50),
            @sprintf("%.10f", min_q75),
            @sprintf("%.10f", r_mean),
            @sprintf("%.10f", pr0),
            @sprintf("%.10f", pr1)
        ], ","))
    end
end

# Overall summary (one-line CSV)
overall_path = replace(outpath, ".csv" => "_overall.csv")
open(overall_path, "w") do oio
    println(oio, "count,ts_mean,ts_var,ts_sd,obj_mean,obj_var,obj_sd,min_mean,min_var,min_sd,r_mean,Pr0,Pr1")
    c = stat_ts_all.n
    pr0 = c == 0 ? NaN : count_r0_all / c
    pr1 = c == 0 ? NaN : count_r1_all / c
    println(oio, join([
        string(c),
        @sprintf("%.10f", stat_ts_all.mean),
        @sprintf("%.10f", var_pop(stat_ts_all)),
        @sprintf("%.10f", sd_pop(stat_ts_all)),
        @sprintf("%.10f", stat_obj_all.mean),
        @sprintf("%.10f", var_pop(stat_obj_all)),
        @sprintf("%.10f", sd_pop(stat_obj_all)),
        @sprintf("%.10f", stat_min_all.mean),
        @sprintf("%.10f", var_pop(stat_min_all)),
        @sprintf("%.10f", sd_pop(stat_min_all)),
        @sprintf("%.10f", stat_r_all.mean),
        @sprintf("%.10f", pr0),
        @sprintf("%.10f", pr1)
    ], ","))
end

@info "saved" outpath
@info "saved" summary_path
@info "saved" overall_path

end

if abspath(PROGRAM_FILE) == @__FILE__
    method_list = ["eAMRw","eMMRw"]
    trueW = ["A","B","C","D","E"]
    Ns = 4:8
    for n in Ns,tw in trueW,md in method_list
        main(n,tw,md)
    end
end
