
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class IntervalResult:
    change_points: List[float]
    winners: List[int]
    winner_values: List[float]
    r_max_in_interval: float
    r_min_in_interval: float

def track_top1_in_interval(t_lower: float, t_upper: float, A: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> IntervalResult:
    n = len(A)
    assert t_lower < t_upper
    def eval_i(i, t): return A[i]*t + B[i]
    vals_left = A * t_lower + B
    p = int(np.argmax(vals_left)); t = float(t_lower)
    change_points, winners, winner_values = [], [p], [eval_i(p, t)]
    r_min, r_max = float("inf"), float("-inf")
    while t < t_upper - eps:
        ap = A[p]
        candidates = [j for j in range(n) if j != p and A[j] > ap + eps]
        tau, q_star = float("inf"), None
        for j in candidates:
            denom = (A[p] - A[j])
            if abs(denom) <= eps: continue
            t_star = (B[j] - B[p]) / denom
            if t_star > t + eps and t_star <= t_upper + eps and t_star < tau:
                tau, q_star = t_star, j
        if q_star is None:
            r_end = eval_i(p, t_upper)
            r_min = min(r_min, eval_i(p, t), r_end)
            r_max = max(r_max, eval_i(p, t), r_end)
            t = t_upper; break
        else:
            r_min = min(r_min, eval_i(p, t), eval_i(p, tau))
            r_max = max(r_max, eval_i(p, t), eval_i(p, tau))
            change_points.append(tau); p = q_star; t = tau
            winners.append(p); winner_values.append(eval_i(p, t))
    if not np.isfinite(r_min) or not np.isfinite(r_max):
        vL = eval_i(p, t_lower); vU = eval_i(p, t_upper)
        r_min, r_max = min(vL, vU), max(vL, vU)
    return IntervalResult(change_points, winners, winner_values, r_max, r_min)

def track_top1_envelope(T0: float, TL: float, TU: float, n: int,
                        update_coefficients: Callable[[float], Tuple[np.ndarray, np.ndarray]],
                        linear_horizon: Callable[[float, np.ndarray, np.ndarray], float],
                        eps: float = 1e-10):
    assert TL < TU
    T = T0
    intervals, interval_results = [], []
    while T > TL + eps:
        A, B = update_coefficients(T)
        dT = float(linear_horizon(T, A, B)); dT = max(dT, 0.0)
        next_T = max(T - dT, TL)
        if abs(next_T - T) <= 1e-12:
            if T > TL + eps:
                res = track_top1_in_interval(TL, T, A, B, eps=eps)
                interval_results.append(res); intervals.append((TL, T))
            break
        res = track_top1_in_interval(next_T, T, A, B, eps=eps)
        interval_results.append(res); intervals.append((next_T, T))
        T = next_T
    change_points, winners, winner_values = [], [], []
    rM, rmin = float("-inf"), float("inf")
    for res in interval_results:
        change_points += res.change_points; winners += res.winners; winner_values += res.winner_values
        rM = max(rM, res.r_max_in_interval); rmin = min(rmin, res.r_min_in_interval)
    return {"intervals": intervals, "change_points": change_points, "winners": winners, "winner_values": winner_values, "r_M": rM, "r_m": rmin}
