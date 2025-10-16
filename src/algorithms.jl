module RankChangeInterval
using Statistics, LinearAlgebra, DataFrames, CSV
export find_rank_change_points_from_tR,
    analyze_all_alternatives_with_avail_space,
    save_rank_changes_and_rankings_to_csv
include("calc_IPW.jl")

# =============================================================
# 既存のデータ構造・関数（create_minimax_R_Matrix, calc_regret）を活用
#   - 既存：minimax_regret_tuple（difference_U, rank, regret, ...）
#   - 既存：create_minimax_R_Matrix(utility)
#   - 既存：calc_regret(matrix, Y_L, Y_R)
# =============================================================

# --- 共有ワークバッファ（deepcopy多発を避ける） ---
mutable struct RegretWorkspace
    base_matrix::Array{minimax_regret_tuple,2}   # difference_U, rank を保持
    work_matrix::Array{minimax_regret_tuple,2}   # regret, interm_index, Avail_space を都度上書き
end

function RegretWorkspace(utility::Matrix{Float64})
    base = create_minimax_R_Matrix(utility)
    # work は base の difference_U / rank をコピー（regret系は毎回上書き）
    work = [minimax_regret_tuple(copy(base[i, j].difference_U),
        copy(base[i, j].rank),
        0.0, 0, 0.0)
            for i in 1:size(base, 1), j in 1:size(base, 2)]
    return RegretWorkspace(base, work)
end

# --- 与えられた t で MR と argmax を一括計算（差分更新のベース） ---
function eval_MR_at_t!(ws::RegretWorkspace, methodW, t::Float64)
    Y_L = methodW.L .* t
    Y_R = methodW.R .* t
    # work_matrix に regret / interm / Avail_space を上書き
    regret_max, _ = calc_regret(ws.work_matrix, Y_L, Y_R)
    # 各 p の MR と相手 q を抽出
    n = size(ws.work_matrix, 1)
    MR_vals = zeros(n)
    MR_arg = fill(0, n)
    for p in 1:n
        # regret_max[p] は (max_val, argmax_index) を返す前提（既存コード準拠）
        maxv, argj = regret_max[p]
        MR_vals[p] = maxv
        MR_arg[p] = argj
    end
    return MR_vals, MR_arg
end

# --- 区間右端（傾き/能動集合が変わる最短点）を、既存の Avail_space 論理で決定 ---
function next_interval_right!(ws::RegretWorkspace, methodW, t::Float64, t_L::Float64)
    # 既存 compute_next_t のロジックをインライン実装
    Y_L = methodW.L .* t
    Y_R = methodW.R .* t
    # work_matrix を再利用（regret計算と同時に Avail_space 更新）
    calc_regret(ws.work_matrix, Y_L, Y_R)
    n = size(ws.work_matrix, 1)
    # i!=j の Avail_space の最小を取得
    min_avail = +Inf
    for i in 1:n, j in 1:n
        i == j && continue
        av = ws.work_matrix[i, j].Avail_space
        if av < min_avail
            min_avail = av
        end
    end
    # r = 1/(1+min_avail) による t 次ステップ（既存関数と同じ意味）
    r = 1 / (1 + min_avail)
    next_t = max(r * t, t_L) # 左端を下回らない
    return next_t, min_avail
end

# --- 直線近似の傾き（区間内で能動集合が不変のとき用） ---
#     小さすぎる微小よりも、[t, t_right] の2点で傾きを推定し安定化
function slope_on_interval!(ws::RegretWorkspace, methodW, t_left::Float64, t_right::Float64)
    MR_l, _ = eval_MR_at_t!(ws, methodW, t_left)
    MR_r, _ = eval_MR_at_t!(ws, methodW, t_right)
    return (MR_r .- MR_l) ./ (t_right - t_left), MR_l
end

# --- トップ1（チャンピオン）だけを追い、交点があればそこにジャンプ ---
function find_rank_change_points_from_tR(o_o::Int, t_L::Float64, t_R::Float64,
    matrix_unused, methodW)
    # 互換シグネチャ確保のために定義のみ（外側の総合関数で集約）
    return (
        change_points=Float64[],
        max_regret_alts=Int[],
        max_regret_values=Float64[],
        intervals=Tuple{Float64,Float64}[],
        interval_data=Any[],
        r_M=NaN,
        r_m=NaN,
    )
end

# --- 外側：全代替案についての総合的な順位変化列挙（トップ1駆動） ---
function analyze_all_alternatives_with_avail_space(utility::Matrix{Float64}, methodW,
    t_range::Tuple{Float64,Float64})
    n = size(utility, 1)
    # 既存 matrix を一度作る（difference_U / rank を持つベース）
    base_matrix = create_minimax_R_Matrix(utility)
    ws = RegretWorkspace(utility)

    tL, tR = t_range
    # 初期評価
    MR_vals, MR_arg = eval_MR_at_t!(ws, methodW, tR)  # 右端開始
    # 総合順位（MRが小さい順）
    ranking_at_t = Dict{Float64,Vector{Int}}()
    ranking_at_t[tR] = sortperm(MR_vals)

    # トップ1 を追跡
    change_points = Float64[tR]
    champion = argmin(MR_vals)  # Minimaxなので MR 最小

    t = tR
    while t > tL + 1e-12
        t_next, _ = next_interval_right!(ws, methodW, t, tL)
        if isapprox(t_next, t; atol=1e-12)
            break
        end
        # 区間 [t_next, t] での傾き（直線）を推定
        slopes, MR_at_left = slope_on_interval!(ws, methodW, t_next, t)
        # 現在の champion が区間で保持されるか、挑戦者が上回る t* があるか調べる
        # MR は "小さい方が良い" なので、他の候補 r が champion より小さくなる交点を探す
        a_c = slopes[champion]
        b_c = MR_at_left[champion]

        t_star = +Inf
        r_star = 0
        for r in 1:n
            r == champion && continue
            a_r = slopes[r]
            b_r = MR_at_left[r]
            da = a_r - a_c
            db = b_r - b_c
            abs(da) < 1e-15 && continue  # 平行
            tcross = t_next + (-db) / da  # b_c + a_c*(t-t_next) = b_r + a_r*(t-t_next)
            if t_next < tcross && tcross <= t + 1e-12
                MRc = b_c + a_c * (tcross - t_next)
                MRr = b_r + a_r * (tcross - t_next)
                if MRr < MRc - 1e-12 && tcross < t_star
                    t_star = tcross
                    r_star = r
                end
            end
        end

        if isfinite(t_star)
            t = t_star
            MR_vals, MR_arg = eval_MR_at_t!(ws, methodW, t)
            champion = argmin(MR_vals)
            ranking_at_t[t] = sortperm(MR_vals)
            push!(change_points, t)
        else
            t = t_next
            MR_vals, MR_arg = eval_MR_at_t!(ws, methodW, t)
            champion = argmin(MR_vals)
            ranking_at_t[t] = sortperm(MR_vals)
            push!(change_points, t)
        end
    end

    unique!(change_points)
    sort!(change_points)
    filtered = Float64[]
    prev_rank = nothing
    for τ in change_points
        rk = ranking_at_t[τ]
        if prev_rank === nothing || rk != prev_rank
            push!(filtered, τ)
            prev_rank = rk
        end
    end

    return (
        all_results_per_alternative=Dict{Int,Any}(),
        change_points=filtered,
        rank_change_data=[(t_value=τ, ranking=ranking_at_t[τ]) for τ in filtered],
        intervals=Tuple{Float64,Float64}[],
        all_points=collect(keys(ranking_at_t)) |> x -> (sort!(x)),
        rankings=ranking_at_t,
    )
end

function save_rank_changes_and_rankings_to_csv(results, filename="rank_change_details.csv")
    if !hasfield(typeof(results), :change_points) || !hasfield(typeof(results), :rankings)
        @warn "resultsに必要フィールドがありません"
        return DataFrame()
    end
    tlist = sort(unique(results.change_points))
    if isempty(tlist)
        @info "順位変化点が空です"
        return DataFrame()
    end
    n = length(first(values(results.rankings)))
    header = [:t_change_point; Symbol.("alt_" .* string.(1:n) .* "_rank")]
    rows = Vector{Vector{Any}}()
    for τ in tlist
        rk = results.rankings[τ]
        invrk = zeros(Int, n)
        for (pos, aid) in enumerate(rk)
            1 <= aid <= n && (invrk[aid] = pos)
        end
        push!(rows, Any[τ; invrk...])
    end
    df = DataFrame([name => [row[i] for row in rows] for (i, name) in enumerate(header)])
    CSV.write(abspath(filename), df)
    return df
end

end # module