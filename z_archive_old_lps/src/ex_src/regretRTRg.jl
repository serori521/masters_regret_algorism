module RegretRTR

# このモジュールがユーザーに提供する機能
export scan_events, collect_events, EventPoint

# --- ステップ1で作成したコアモジュールを読み込む ---
# (同じディレクトリにあるファイルを読み込む)
include("set_regret.jl")
using .SetRegretCore

# --- ユーザー向けのデータ構造 ---

"""
    EventPoint

一つのイベント（傾きが変化する境界点）で観測された状態を格納する。

# Fields
- `t::Float64`: イベントが発生したパラメータ `t` の値。
- `MR::Vector{Float64}`: `t` 時点での各代替案の最大リグレット（Maximum Regret）ベクトル。
- `ranking::Vector{Int}`: MR値に基づく順位。`ranking[1]` がMinimax解のインデックス。
- `hit_pairs::Vector{Tuple{Int,Int}}`: この `t` で傾きが変化したリグレット関数のペア `(p, q)` のリスト。
"""
struct EventPoint
    t::Float64
    MR::Vector{Float64}
    ranking::Vector{Int}
    hit_pairs::Vector{Tuple{Int,Int}}
end

"""
    RTRScanner

右から左へのパラメータ走査の状態を管理するイテレータ。
`for` ループで使うことで、イベント点を順次生成する。
"""
struct RTRScanner
    matrix::Array{minimax_regret_tuple, 2}
    t_L::Float64
    t_R::Float64
end

# --- ユーザー向けAPI関数 ---

"""
    scan_events(utility, L, R) -> RTRScanner

リグレットの順位変化点を走査するためのイテレータ (`RTRScanner`) を生成する。

`for event in scan_events(...)` のように使うことで、
`t` が大きい方から小さい方へ、イベントが発生する点ごとに処理を実行できる。
"""
function scan_events(utility::Matrix{Float64}, L::Vector{Float64}, R::Vector{Float64})
    # 1. パラメータ t の有効範囲を計算
    t_L, t_R = find_optimal_trange(L, R)

    # 2. コアエンジンを初期化
    matrix = create_minimax_R_Matrix(utility)
    initialize_linear_models!(matrix, L, R, t_R)

    # 3. 走査状態を持つイテレータオブジェクトを生成して返す
    return RTRScanner(matrix, t_L, t_R)
end

"""
    collect_events(utility, L, R) -> Vector{EventPoint}

すべてのイベント点を一度に計算し、`EventPoint` の配列として返す。
"""
function collect_events(utility::Matrix{Float64}, L::Vector{Float64}, R::Vector{Float64})
    # scan_events でイテレータを作り、`collect` で全要素を配列に変換する
    return collect(scan_events(utility, L, R))
end


# --- Juliaのイテレーション・プロトコルの実装 ---
import Base: iterate

# イテレータの初期化 (forループの1回目)
function Base.iterate(scanner::RTRScanner)
    # 開始点 (t = t_R) の状態を最初のイベントとして返す
    t = scanner.t_R
    MR = max_regret_vector(scanner.matrix)
    rk = ranking_from_MR(MR)
    
    # 最初のイベントポイントを作成 (hit_pairsは空)
    initial_event = EventPoint(t, MR, rk, [])
    
    # (今回返す値, 次のループに渡す状態) のタプルを返す
    # 次の状態として現在の t の値を渡す
    return (initial_event, t)
end

# ループの2回目以降
function Base.iterate(scanner::RTRScanner, t_current::Float64)
    # MinimaxRegretCoreの関数を呼び出し、状態を1ステップ進める
    t_next, hit_pairs = advance_TR_once!(scanner.matrix, t_current, scanner.t_L)
    
    # 終了条件: t_L に到達したら nothing を返してループを終了
    if t_next <= scanner.t_L + 1e-12
        return nothing
    end
    
    # 新しい t での MR と順位を計算
    MR = max_regret_vector(scanner.matrix)
    rk = ranking_from_MR(MR)
    
    # 今回のイベントポイントを作成
    next_event = EventPoint(t_next, MR, rk, hit_pairs)
    
    # (今回返す値, 次のループに渡す状態) を返す
    return (next_event, t_next)
end

end # module RegretRTR