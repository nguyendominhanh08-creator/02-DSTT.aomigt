"""Run with:  python -m pytest -q   (or simply: python tests/test_solvers.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.certify import certify
from modules.lp_solver import solve_matrix_game
from modules.saddle_point import find_saddle_points
from modules.scenarios import SCENARIOS
from modules.solve_2x2 import solve_game_2x2


def S(key):
    return next(v["A"] for k, v in SCENARIOS.items() if k.startswith(key))


def test_case1_saddle():
    A = S("Case Study 1")
    assert find_saddle_points(A) == [(1, 0)]
    s = solve_matrix_game(A)
    assert abs(s.value - 22) < 1e-9 and s.x.tolist() == [0, 1] and s.y.tolist() == [1, 0]


def test_case2_saddle():
    A = S("Case Study 2")
    assert find_saddle_points(A) == [(1, 1)]
    assert abs(solve_matrix_game(A).value - 3) < 1e-9


def test_case3_mixed_closed_form_matches_lp():
    A = S("Case Study 3")
    assert find_saddle_points(A) == []
    x, y, v = solve_game_2x2(A)
    assert np.allclose(x, [50.54 / 55.04, 1 - 50.54 / 55.04])
    assert np.allclose(y, [54.04 / 55.04, 1 - 54.04 / 55.04])
    assert abs(v + 50.54 / 55.04) < 1e-12
    s = solve_matrix_game(A)
    assert abs(s.value - v) < 1e-9 and certify(A, x, y)["optimal"]


def test_case3_sensitivity_cp8():
    A = np.array([[-1, 7], [0, -50.54]])
    x, _, v = solve_game_2x2(A)
    assert abs(x[0] - 50.54 / 58.54) < 1e-12 and abs(v + 50.54 / 58.54) < 1e-12


def test_3x3_section_3_5():
    A = S("Example A")
    s = solve_matrix_game(A)
    assert np.allclose(s.x, np.array([43, 34, 32]) / 109, atol=1e-8)
    assert np.allclose(s.y, np.array([35, 9, 65]) / 109, atol=1e-8)
    assert abs(s.value + 99 / 109) < 1e-9


def test_3x4_rectangular():
    A = S("Example B")
    s = solve_matrix_game(A)
    assert abs(s.value - 19 / 27) < 1e-9
    assert certify(A, s.x, s.y)["optimal"]
    assert s.y[0] < 1e-9  # Y1 unused (complementary slackness)


def test_saddle_3x3_section_2_2():
    A = np.array([[3, 2, 4], [1, 0, 5], [2, -1, 3]], float)
    assert find_saddle_points(A) == [(0, 1)]


def test_2x2_rejects_saddle_and_random_lp_certified():
    try:
        solve_game_2x2(S("Case Study 1"))
        raise AssertionError("should have raised")
    except ValueError:
        pass
    rng = np.random.default_rng(1)
    for _ in range(200):
        m, n = rng.integers(2, 7, 2)
        A = rng.integers(-9, 10, (m, n)).astype(float)
        s = solve_matrix_game(A)
        c = certify(A, s.x, s.y)
        assert c["optimal"] and abs(s.value - c["v_low"]) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
