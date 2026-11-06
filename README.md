# EquiMatrix - Two-Player Zero-Sum Game Solver

    pip install -r requirements.txt
    streamlit run app.py

Select a built-in scenario (3 case studies, 3x3 and 3x4 examples) or enter any m x n payoff matrix, then press Calculate.
Pure saddle points are detected first; otherwise 2x2 games use the closed form and all other games use linear programming.
