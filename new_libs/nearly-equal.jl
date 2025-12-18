"""
precision loss の修正を行う  
`abs(a - b)` が 1×10⁻⁸ より小さければ b を返す  
そうでなければ a を返す
"""
@inline function correctPrecisionLoss(a, b)
    if abs(a - b) < 1e-8 return b end
    return a
end

"""
`abs(a - b)` が 1×10⁻⁸ より小さいか
"""
@inline function nearlyEqual(a::Number, b::Number)
    if abs(a - b) < 1e-5 return true end # 1e-8 だと精度が高すぎ？-5に緩和した
    return false
end

"""
`abs(a - b)` が 1×10⁻³ より小さいか
"""
@inline function nearlyEqualLoose(a::Number, b::Number)
    if abs(a - b) < 1e-3 return true end
    return false
end