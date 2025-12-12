module RegretRTR

using .SetRegretCore

# 公開するもの（上位API）
export RTRState,
       build_state!,
       step_RTR!,
       run_RTR!,
       get_current_MR_ranking,
       get_change_points

###########################
# 状態構造体
###########################

"""
RTRState は，右→左（t^U→t^L）走査の進行状態を保持する高レベル状態です。

- matrix:
    各ペア (p,q) の minimax_regret_tuple (SetRegretCore側で定義された線形モデルと状態)
- wL, wU:
    各基準 i の下限重み w_i^L と上限重み w_i^U
- tL, tR:
    走査範囲 [tL, tR]
- t:
    現在位置 t (右→左に単調減少させる)
- Tchg:
    Minimax Regret の外側順位が入れ替わった交点候補（これが“変化点”として最終的に出力される）
"""
mutable struct RTRState
    matrix::Array{SetRegretCore.minimax_regret_tuple,2}
    wL::Vector{Float64}
    wU::Vector{Float64}
    tL::Float64
    tR::Float64
    t::Float64
    Tchg::Vector{Float64}
end


###########################
# 初期化
###########################

"""
build_state!(utility, wL, wU)

- utility[p,i] = u_i(o_p)
- wL[i] = w_i^L
- wU[i] = w_i^U

手順:
1. tL,tR を決定
2. (p,q) ごとの差分・rankを作成
3. t = tR 時点で線形モデル (slope/intercept/tstar等) を初期化
4. 状態 RTRState を返す
"""
function build_state!(utility::Matrix{Float64},
                      wL::Vector{Float64},
                      wU::Vector{Float64})
    # 走査範囲
    tL, tR = SetRegretCore.find_optimal_trange(wL, wU)

    # (p,q)ごとのセルを生成
    matrix = SetRegretCore.create_minimax_R_Matrix(utility)

    # tR 時点で線形モデル等を初期化
    SetRegretCore.initialize_linear_models!(matrix, wL, wU, tR)

    # 状態構築
    st = RTRState(
        matrix,
        wL, wU,
        tL, tR,
        tR,                # 現在位置 t
        Float64[]          # 交点(順位変化点)ログ
    )
    return st
end


###########################
# 1ステップ前進 (t を左に進める)
###########################

"""
step_RTR!(st)

1ステップ分，現在の t から次のイベント時刻 t_next まで進める。
- 全セルの regret を Δt で差分更新
- event_type (:active_set or :inner) に応じてヒットしたペアだけ再線形化 (promote / rebuild)
- その区間 [t_next, t_cur] における Minimax Regret の外側順位入替時刻を計算し st.Tchg に追記
- 状態 st.t を更新

戻り値:
  (t_next, event_type, hit_pairs)

注意:
  既に st.t ≤ st.tL の場合は変化なしでそのまま返す。
"""
function step_RTR!(st::RTRState)
    t_cur = st.t
    t_L   = st.tL

    if t_cur <= t_L
        # もうこれ以上進めない
        return st.t, :none, Tuple{Int,Int}[]
    end

    # SetRegretCore.advance_TR_once! は
    #   - 次イベント時刻 t_next
    #   - そのイベントでヒットした (p,q) のリスト
    #   - イベント種 (:active_set or :inner or :none)
    #   - 区間で起きた外側順位の交点Tchg_step
    t_next, hit_pairs, evtype, Tchg_step =
        SetRegretCore.advance_TR_once!(
            st.matrix,
            st.wL, st.wU,
            t_cur, t_L
        )

    # 変化点を蓄積
    append!(st.Tchg, Tchg_step)

    # 現在位置を更新
    st.t = t_next

    return t_next, evtype, hit_pairs
end


###########################
# 全走査実行
###########################

"""
run_RTR!(st)

tR から tL までイベント駆動で走査を繰り返し，
Minimax Regret の順位変化点（外側順位が入れ替わった交点候補）を st.Tchg に貯める。

繰り返し停止条件:
- st.t ≤ st.tL
- または next_event が :none で，これ以上イベントが発生しない場合（区間が線形で固定）

戻り値:
  st.Tchg （いままでに検出された変化点 t の集まり）
"""
function run_RTR!(st::RTRState)
    # 安全装置: 無限ループ防止のため上限
    # （理屈的にはイベント数は有限だが，バグ検出の保険）
    max_steps = 10_000

    for _ in 1:max_steps
        t_prev = st.t
        t_new, evtype, _hit = step_RTR!(st)

        # これ以上進まない (t が変わらない or evtype=:none)
        if t_new == t_prev || evtype == :none || t_new <= st.tL
            break
        end
    end

    return st.Tchg
end


###########################
# 現時点のMR順位を取得（解析・可視化用）
###########################

"""
get_current_MR_ranking(st)

現在時刻 st.t における
- MRベクトル
- ランキング（MR小さい順）
を返す。
"""
function get_current_MR_ranking(st::RTRState)
    MR = SetRegretCore.max_regret_vector(st.matrix)
    rk = SetRegretCore.ranking_from_MR(MR)
    return MR, rk
end


###########################
# 変化点の取得（重複はまとめたい場合はユニーク取ればOK）
###########################

"""
get_change_points(st)

今までに記録された外側順位の交点候補（Minimax Regret の順位変化点） st.Tchg を返す。
必要に応じて sort / unique は呼び出し側で行ってください。
"""
get_change_points(st::RTRState) = st.Tchg

end # module RegretRTR
