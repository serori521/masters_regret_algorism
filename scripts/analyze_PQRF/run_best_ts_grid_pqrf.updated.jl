include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "load_instance.jl"))
include(joinpath(@__DIR__, "..", "src", "SetRegretCore.jl"))
include(joinpath(@__DIR__, "..", "new_libs", "analysis-indicators.jl"))

using .Paths
using .LoadInstance
using .SetRegretCore
using IntervalArithmetic
using Printf

# -------------------------
# Config
# -------------------------
const N = 6
const TRUE_WEIGHT_TYPE = "A"
const METHOD = "eAMRwc"
const REPEAT_NUM = 1000

const NT = 50
const NTS = 50
const METRIC = :F
const AGG = :mean

function out_csv_path(paths)
    outdir = joinpath(paths.data, "metrics_julia")
    mkpath(outdir)
    return joinpath(outdir, "best_ts_grid_pqrf_N$(N)_$(TRUE_WEIGHT_TYPE)_$(METHOD).csv")
end

function main()
    paths = Paths.project_paths()
    outpath = out_csv_path(paths)

    trueW = LoadInstance.read_true_weights(paths, TRUE_WEIGHT_TYPE; N=N)
    w_std_true = interval.(trueW.L, trueW.R)
    tL_true, tU_true = SetRegretCore.find_optimal_trange(trueW.L, trueW.R)
    t_grid = range(tL_true, tU_true; length=NT)

    filename = joinpath(TRUE_WEIGHT_TYPE, METHOD)
    methodW = LoadInstance.read_method_weights(paths, filename, REPEAT_NUM, N; a3="a3")
    repeat = min(REPEAT_NUM, length(methodW))

    open(outpath, "w") do io
        println(io, "pcm_id,t_idx,t,ts_star,r,best_obj,best_mean,best_var,best_sd,best_min")

        for pcm_id in 1:repeat
            wL = methodW[pcm_id].L
            wU = methodW[pcm_id].R
            w_std_est = interval.(wL, wU)
            tsL, tsU = SetRegretCore.find_optimal_trange(wL, wU)

            ts_star_list, r_list, best_obj_list, best_mean_list, best_var_list, best_sd_list, best_min_list = best_ts_grid(
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
                println(io, join([
                    pcm_id,
                    t_idx,
                    @sprintf("%.10f", t),
                    @sprintf("%.10f", ts_star_list[t_idx]),
                    @sprintf("%.10f", r_list[t_idx]),
                    @sprintf("%.10f", best_obj_list[t_idx]),
                    @sprintf("%.10f", best_mean_list[t_idx]),
                    @sprintf("%.10f", best_var_list[t_idx]),
                    @sprintf("%.10f", best_sd_list[t_idx]),
                    @sprintf("%.10f", best_min_list[t_idx])
                ], ','))
            end
        end
    end

    @info "saved" outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
