# src/types.jl
module CoreTypes

# =========
# Core Types
# =========

"R(t) = A*t + B"
struct Line
    A::Float64
    B::Float64
end

@enum EventKind begin
    E1_COEF_SWITCH   # 係数切替
    E2_INNER_TOP     # 内側1位交代
    E3_OUTER_ORDER   # 外側順位変化
end

"イベント（次に起きる変化点）"
struct Event
    t::Float64
    kind::EventKind
    p::Int
    q::Int
end

"結果（ログや変化点）"
mutable struct LPSResult
    events::Vector{Event}
end

# =========
# Small utilities
# =========
eval_line(line::Line, t::Float64) = line.A * t + line.B

"入力データのまとめ（まずは必要最低限）"
struct LPSInstance
    utility::String
    N::Int
    M::Int
    U::Vector{Matrix{Float64}}
end

"初期状態（まずはinstanceと現在tだけ持つ）"
mutable struct LPSState
    t::Float64
    inst::LPSInstance
end

export Line, EventKind, Event, LPSResult, eval_line, LPSInstance, LPSState

end # module
