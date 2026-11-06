import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from modules.saddle_point import find_saddle_points, maximin_minimax
from modules.solve_2x2 import solve_game_2x2
from modules.lp_solver import solve_matrix_game
from modules.scenarios import SCENARIOS, CUSTOM
from utils.data_validator import validate_matrix

# ---------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EquiMatrix - Game Theory Solver",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, button, input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main-title { font-size: 2.6rem; font-weight: 800; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #4B5563; text-align: center; font-style: italic; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">EQUIMATRIX OPTIMIZATION SOLVER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Player Zero-Sum Game Theory Engine • Linear Algebra Project</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar Setup & Scenario Presets
# ---------------------------------------------------------
with st.sidebar:
    st.title("Control Panel")
    preset = st.selectbox("Choose Test Scenario:", [CUSTOM] + list(SCENARIOS.keys()))

    if preset == CUSTOM:
        m = int(st.number_input("Number of rows m (Player 1 strategies):", min_value=2, max_value=8, value=3, step=1))
        n = int(st.number_input("Number of columns n (Player 2 strategies):", min_value=2, max_value=8, value=4, step=1))
    st.caption("Row player maximises the payoff, column player minimises it.")

# Preset initial data
if preset == CUSTOM:
    init_data = np.zeros((m, n), dtype=float)
    row_labels = [f"Strategy X{i+1}" for i in range(m)]
    col_labels = [f"Strategy Y{j+1}" for j in range(n)]
    info = None
else:
    info = SCENARIOS[preset]
    init_data = info["A"].copy()
    m, n = init_data.shape
    row_labels = info["rows"] or [f"Strategy X{i+1}" for i in range(m)]
    col_labels = info["cols"] or [f"Strategy Y{j+1}" for j in range(n)]

default_df = pd.DataFrame(init_data, columns=col_labels, index=row_labels)

st.subheader("Payoff Matrix Input A")
edited_df = st.data_editor(
    default_df,
    key=f"editor_{preset}_{m}x{n}",
    num_rows="fixed",
    width="stretch",
)

solve_btn = st.button("Calculate Equilibrium Solution", type="primary", width="stretch")


# ---------------------------------------------------------
# 3. Plotly Chart Helper Function
# ---------------------------------------------------------
def plot_strategy_distribution(probs, player_name, color_scale, labels):
    probs = np.asarray(probs, dtype=float)
    labels = list(labels)
    if len(probs) != len(labels):
        raise ValueError(f"{player_name}: probability count mismatch.")

    df = pd.DataFrame({
        "Strategy": labels,
        "Probability": probs,
        "Label": [f"{value:.4f}" for value in probs],
    })
    fig = px.bar(
        df, x="Strategy", y="Probability", text="Label",
        title=f"{player_name} Strategy Distribution",
        color="Probability", color_continuous_scale=color_scale,
        category_orders={"Strategy": labels},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        font_family="Times New Roman",
        title_font_size=16,
        showlegend=False,
        yaxis=dict(range=[0, 1.25], title="Probability", tickformat=".2f"),
        xaxis=dict(type="category", categoryorder="array", categoryarray=labels),
        coloraxis_showscale=False,
        margin=dict(t=65, b=50),
    )
    return fig


# ---------------------------------------------------------
# 4. Solver Execution
# ---------------------------------------------------------
if solve_btn:
    ok, msg, A = validate_matrix(edited_df.to_numpy())
    if not ok:
        st.error(msg)
        st.stop()

    st.markdown("---")
    st.header("Equilibrium Results")
    m, n = A.shape
    labels_x = [f"X{i+1}" for i in range(m)]
    labels_y = [f"Y{j+1}" for j in range(n)]
    plot_x = list(row_labels) if preset != CUSTOM and info["rows"] else [f"Strategy {s}" for s in labels_x]
    plot_y = list(col_labels) if preset != CUSTOM and info["cols"] else [f"Strategy {s}" for s in labels_y]

    # Step 1: Maximin / Minimax
    row_mins, col_maxs, L, U = maximin_minimax(A)
    c1, c2 = st.columns(2)
    c1.metric("Maximin  L = max_i min_j a_ij", f"{L:.4f}")
    c2.metric("Minimax  U = min_j max_i a_ij", f"{U:.4f}")

    # Step 2: Saddle point check
    saddle_pts = find_saddle_points(A)
    solver_used = ""
    if saddle_pts:
        r_idx, c_idx = saddle_pts[0]
        game_val = float(A[r_idx, c_idx])
        x_opt = np.eye(m)[r_idx]
        y_opt = np.eye(n)[c_idx]

        st.success("Pure Strategy Equilibrium (Saddle Point) Identified!  (L = U)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Game Value (v)", f"{game_val:.4f}")
        m2.metric("Player 1 Choice", plot_x[r_idx])
        m3.metric("Player 2 Choice", plot_y[c_idx])
        where = ", ".join(f"({i+1}, {j+1})" for i, j in saddle_pts)
        st.info(f"**Equilibrium Location:** Cell {where} | Payoff = {game_val:.4f}")
    else:
        x_opt = y_opt = game_val = None
        # Step 3: closed form for 2x2, otherwise the general LP
        if (m, n) == (2, 2):
            try:
                x_opt, y_opt, game_val = solve_game_2x2(A)
                solver_used = "2x2 Mixed Strategy Analytical Engine (indifference principle)"
            except ValueError:
                x_opt = None
        if x_opt is None:
            sol = solve_matrix_game(A)
            x_opt, y_opt, game_val = sol.x, sol.y, sol.value
            solver_used = f"General {m} × {n} Linear Programming Engine (HiGHS, primal-dual)"

        st.warning(f"No Pure Saddle Point found (L < U). Optimal Mixed Equilibrium calculated via {solver_used}.")
        st.metric("Expected Game Value (v)", f"{game_val:.4f}")

    col1, col2 = st.columns(2)
    col1.plotly_chart(plot_strategy_distribution(x_opt, "Player 1", "Blues", plot_x), width="stretch")
    col2.plotly_chart(plot_strategy_distribution(y_opt, "Player 2", "Tealgrn", plot_y), width="stretch")
    st.caption(f"x* = {np.round(x_opt, 4).tolist()}   |   y* = {np.round(y_opt, 4).tolist()}")
