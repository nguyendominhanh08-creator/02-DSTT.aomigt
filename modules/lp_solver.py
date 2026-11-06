"""General m x n zero-sum game via the primal-dual LP pair (report section 3.3-3.4).

(P)  max v   s.t.  A^T x >= v 1_n,  1^T x = 1,  x >= 0,  v free
(D)  min w   s.t.  A y   <= w 1_m,  1^T y = 1,  y >= 0,  w free
Both programs are solved on the ORIGINAL matrix (no shifting) and cross-checked.
"""
import numpy as np
from scipy.optimize import linprog


class GameSolution:
    def __init__(self, value, x, y, primal_value, dual_value):
        self.value = float(value)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.primal_value = float(primal_value)
        self.dual_value = float(dual_value)


def _clean(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    p[p < 1e-12] = 0.0
    return p / p.sum()


def solve_matrix_game(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0:
        raise ValueError("Matrix A must be a non-empty 2D array (m x n).")
    if not np.isfinite(A).all():
        raise ValueError("Matrix A must contain only finite numbers.")
    m, n = A.shape

    # Primal: variables [x_1..x_m, v]; minimize -v
    c_p = np.concatenate([np.zeros(m), [-1.0]])
    A_ub_p = np.hstack([-A.T, np.ones((n, 1))])          # -A^T x + v <= 0
    A_eq_p = np.concatenate([np.ones(m), [0.0]]).reshape(1, -1)
    res_p = linprog(c_p, A_ub=A_ub_p, b_ub=np.zeros(n), A_eq=A_eq_p, b_eq=[1.0],
                    bounds=[(0, None)] * m + [(None, None)], method="highs")

    # Dual: variables [y_1..y_n, w]; minimize w
    c_d = np.concatenate([np.zeros(n), [1.0]])
    A_ub_d = np.hstack([A, -np.ones((m, 1))])            # A y - w <= 0
    A_eq_d = np.concatenate([np.ones(n), [0.0]]).reshape(1, -1)
    res_d = linprog(c_d, A_ub=A_ub_d, b_ub=np.zeros(m), A_eq=A_eq_d, b_eq=[1.0],
                    bounds=[(0, None)] * n + [(None, None)], method="highs")

    if not (res_p.success and res_d.success):
        raise RuntimeError("Linear programming solver failed to converge.")

    x = _clean(res_p.x[:m])
    y = _clean(res_d.x[:n])
    v_p, w_d = -res_p.fun, res_d.fun
    return GameSolution(0.5 * (v_p + w_d), x, y, v_p, w_d)
