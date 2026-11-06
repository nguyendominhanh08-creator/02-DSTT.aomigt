"""Closed-form solution of a 2x2 zero-sum game (indifference principle) -- report section 3.2."""
import numpy as np


def solve_game_2x2(A):
    """Return (x, y, v) for A = [[a, b], [c, d]] when there is NO pure saddle point.

    p* = (d - c)/D,  q* = (d - b)/D,  v = (ad - bc)/D,  D = a + d - b - c.
    Raises ValueError if the formula is not applicable (saddle point exists, D = 0,
    or the probabilities fall outside [0, 1]); the caller should then use the LP.
    """
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("solve_game_2x2 needs a 2x2 matrix.")
    a, b = A[0]
    c, d = A[1]
    D = (a + d) - (b + c)
    if np.isclose(D, 0.0):
        raise ValueError("D = a + d - b - c = 0: closed form not applicable.")
    p = (d - c) / D
    q = (d - b) / D
    v = (a * d - b * c) / D
    eps = 1e-12
    if not (-eps <= p <= 1 + eps and -eps <= q <= 1 + eps):
        raise ValueError("Probabilities outside [0, 1]: a pure saddle point exists; closed form not applicable.")
    p = min(max(p, 0.0), 1.0)
    q = min(max(q, 0.0), 1.0)
    return np.array([p, 1 - p]), np.array([q, 1 - q]), float(v)
