"""Built-in case studies and m x n examples (data only)."""
import numpy as np

CUSTOM = "Custom Matrix Input (m x n)"

SCENARIOS = {
    "Case Study 1: Smartphone Advertising (2x2)": {
        "A": np.array([[19.0, 30.0], [22.0, 33.0]]),
        "rows": ["A1: Camera", "A2: Battery life"],
        "cols": ["B1: Speed", "B2: Design"],
    },
    "Case Study 2: Core Banking Bidding (2x2)": {
        "A": np.array([[3.0, -4.0], [10.0, 3.0]]),
        "rows": ["A1: TCO focus", "A2: Security & Architecture"],
        "cols": ["B1: Risk minimisation", "B2: Innovation ecosystem"],
    },
    "Case Study 3: Traffic Enforcement (2x2)": {
        "A": np.array([[-1.0, 3.5], [0.0, -50.54]]),
        "rows": ["A1: Deploy enforcement", "A2: No enforcement"],
        "cols": ["B1: Stop at red light", "B2: Run red light"],
    },
    "Example A: 3x3 Game": {
        "A": np.array([[3.0, -1.0, -3.0], [-2.0, 4.0, -1.0], [-5.0, -6.0, 2.0]]),
        "rows": None, "cols": None,
    },
    "Example B: 3x4 Game": {
        "A": np.array([[4.0, 0.0, -2.0, 3.0], [-1.0, 3.0, 2.0, -2.0], [0.0, -3.0, 5.0, 1.0]]),
        "rows": None, "cols": None,
    },
}
