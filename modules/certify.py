"""Numerical certification of a candidate equilibrium -- report section 3.6.

v_low = min_j (A^T x)_j ,  v_up = max_i (A y)_i ,  gap g = v_up - v_low >= 0.
g = 0 (within tolerance) proves (x, y) optimal and v_low <= v <= v_up.
"""
import numpy as np

EPS_P = 1e-8


def certify(A, x, y):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eps_v = 1e-7 * max(1.0, float(np.abs(A).max()))
    ATx = A.T @ x
    Ay = A @ y
    v_low = float(ATx.min())
    v_up = float(Ay.max())
    gap = v_up - v_low
    prob_ok = (abs(x.sum() - 1) <= EPS_P and abs(y.sum() - 1) <= EPS_P
               and x.min() >= -EPS_P and y.min() >= -EPS_P)
    return {
        "ATx": ATx, "Ay": Ay, "v_low": v_low, "v_up": v_up, "gap": gap,
        "eps_v": eps_v, "prob_ok": bool(prob_ok),
        "optimal": bool(prob_ok and abs(gap) <= eps_v),
    }
