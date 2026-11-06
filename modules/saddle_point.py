"""Pure saddle-point detection (Maximin / Minimax) -- report section 3.1."""
import numpy as np


def maximin_minimax(A):
    """Return (row_mins, col_maxs, L, U) with L = max_i min_j a_ij, U = min_j max_i a_ij."""
    A = np.asarray(A, dtype=float)
    row_mins = A.min(axis=1)
    col_maxs = A.max(axis=0)
    return row_mins, col_maxs, float(row_mins.max()), float(col_maxs.min())


def find_saddle_points(A, tol=1e-9):
    """All pure saddle points (i, j): minimum of its row AND maximum of its column.

    Returns a list of (row_index, col_index); empty list when L < U.
    """
    A = np.asarray(A, dtype=float)
    row_mins, col_maxs, L, U = maximin_minimax(A)
    scale = max(1.0, float(np.abs(A).max()))
    if abs(L - U) > tol * scale:
        return []
    pts = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i, j] - row_mins[i]) <= tol * scale and abs(A[i, j] - col_maxs[j]) <= tol * scale:
                pts.append((i, j))
    return pts
