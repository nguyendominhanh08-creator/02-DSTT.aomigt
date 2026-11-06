import numpy as np


def validate_matrix(matrix_input):
    """Return (ok, message, matrix). The matrix must be a non-empty 2D array of finite reals."""
    try:
        M = np.array(matrix_input, dtype=float)
    except Exception as e:  # non-numeric content
        return False, f"Data format error: {e}", None
    if M.ndim != 2 or M.size == 0:
        return False, "Matrix must be a non-empty 2-dimensional array.", None
    if np.isnan(M).any():
        return False, "The matrix has empty cells. Please fill in every entry.", None
    if not np.isfinite(M).all():
        return False, "The matrix contains infinite values.", None
    return True, "Data is valid.", M
