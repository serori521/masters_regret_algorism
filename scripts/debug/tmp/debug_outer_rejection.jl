"""
    debug_outer_rejection(matrix, qstar, x_p_max, t_min, t_max; x0=nothing, eps=1e-12,
                          near_tol=1e-6, print_all_in_interval=false, max_print=80)

collect_outer_changes の「候補が push されない理由」を特定するデバッグ。

- x0 を指定すると、交点 x が x0 近傍（|x-x0|<=near_tol）のペアだけ詳しく出す
- print_all_in_interval=true にすると、区間 [t_min, t_max] に入る交点を全部出す（多いので注意）
戻り値: 該当候補の NamedTuple 配列（後で手元でソート/分析できる）
"""
function debug_outer_rejection(matrix, qstar, x_p_max, t_min, t_max;
                              x0=nothing, eps=1e-12, near_tol=1e-6,
                              print_all_in_interval=false, max_print=80)

    A = length(qstar)
    out = NamedTuple[]
    nprinted = 0

    # minimax_regret_tuple から必要フィールドを取る（名前が違っても落ちにくいように）
    get_slope(l)     = getproperty(l, :slope)
    get_intercept(l) = getproperty(l, :intercept)
    get_tstar(l)     = getproperty(l, :tstar)

    for p1 in 1:A-1, p2 in p1+1:A
        q1 = qstar[p1]; q2 = qstar[p2]
        (q1 == 0 || q2 == 0) && continue

        l1 = matrix[p1, q1]
        l2 = matrix[p2, q2]

        A1 = get_slope(l1); B1 = get_intercept(l1); t1 = get_tstar(l1)
        A2 = get_slope(l2); B2 = get_intercept(l2); t2 = get_tstar(l2)

        denom = (A1 - A2)
        abs(denom) <= eps && continue  # 平行（交点なし）

        x = (B2 - B1) / denom

        in_interval = (x >= t_min - eps) && (x <= t_max + eps)
        near_x0 = (x0 === nothing) ? false : (abs(x - x0) <= near_tol)

        if !(near_x0 || (print_all_in_interval && in_interval))
            continue
        end

        # collect_outer_changes と同じ lower
        parts = (t_min=t_min, t1=t1, t2=t2, xp1=x_p_max[p1], xp2=x_p_max[p2])
        lower = max(parts.t_min, parts.t1, parts.t2, parts.xp1, parts.xp2)

        pass_lower = (lower <= x + eps)
        pass_upper = (x <= t_max - eps)

        verdict =
            if pass_lower && pass_upper
                :PUSHED
            elseif !pass_lower
                :REJECT_LOWER
            else
                :REJECT_UPPER
            end

        # どの要素が lower を支配してるか（複数同率あり）
        dom = String[]
        abs(parts.t_min - lower) <= 10eps && push!(dom, "t_min")
        abs(parts.t1    - lower) <= 10eps && push!(dom, "tstar(p1)")
        abs(parts.t2    - lower) <= 10eps && push!(dom, "tstar(p2)")
        abs(parts.xp1   - lower) <= 10eps && push!(dom, "x_p_max(p1)")
        abs(parts.xp2   - lower) <= 10eps && push!(dom, "x_p_max(p2)")

        rec = (
            p1=p1, p2=p2, q1=q1, q2=q2,
            x=x, lower=lower, t_min=t_min, t_max=t_max,
            dom=join(dom, "|"),
            parts=parts,
            verdict=verdict
        )
        push!(out, rec)

        if nprinted < max_print
            println("---- candidate ----")
            println("p1=$p1 q1=$q1   p2=$p2 q2=$q2")
            println("x = $x")
            println("lower = $lower   (dom: ", join(dom, ", "), ")")
            println("parts = ", parts)
            println("check: lower<=x? ", pass_lower, "   x<=t_max-eps? ", pass_upper, "   => ", verdict)
            nprinted += 1
        end
    end

    println("\n[debug_outer_rejection] total hits = ", length(out), " (printed ", min(length(out), max_print), ")")
    return out
end
