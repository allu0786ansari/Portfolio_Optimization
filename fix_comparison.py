"""Fix dashboard/pages/3_Comparison.py — corrects the highlight_best function.
Run: python fix_comparison.py
"""
from pathlib import Path

content = '''\
"""Page 3: Strategy Comparison — RL vs Markowitz vs Equal-Weight."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_utils import load_backtest_results, get_metrics_df

st.set_page_config(page_title="Comparison", page_icon="\\u2696\\ufe0f", layout="wide")
st.title("\\u2696\\ufe0f Strategy Comparison")
st.caption("RL Agent vs Markowitz vs Equal-Weight — walk-forward backtest")

results = load_backtest_results()
if results is None:
    st.error("Backtest results not found. Run: python -m backtesting.walk_forward")
    st.stop()

algo = results.get("algo", "ppo").upper()

# ── Metrics table ────────────────────────────────────────────────
st.subheader("Full Metrics Table")
metrics_df = get_metrics_df(results)

# Metrics where HIGHER is better
HIGHER_IS_BETTER = {"Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "CAGR"}
# Metrics where LESS NEGATIVE (closer to 0) is better
LOWER_IS_BETTER  = {"Max Drawdown", "VaR 95%", "CVaR 95%"}


def highlight_best_row(row):
    """Highlight the best value per row with a green background."""
    styles = [""] * len(row)
    metric_name = row.name

    # Parse numeric values from formatted strings
    numeric = []
    for v in row:
        try:
            numeric.append(float(str(v).replace("%", "").strip()))
        except Exception:
            numeric.append(None)

    valid = [(i, v) for i, v in enumerate(numeric) if v is not None]
    if not valid:
        return styles

    if metric_name in HIGHER_IS_BETTER:
        best_idx = max(valid, key=lambda x: x[1])[0]
    elif metric_name in LOWER_IS_BETTER:
        # For drawdown/VaR/CVaR — least negative (closest to 0) is best
        best_idx = max(valid, key=lambda x: x[1])[0]
    else:
        best_idx = max(valid, key=lambda x: x[1])[0]

    styles[best_idx] = "background-color: #E1F5EE; font-weight: 600; color: #085041"
    return styles


st.dataframe(
    metrics_df.style.apply(highlight_best_row, axis=1),
    use_container_width=True,
)

st.markdown("---")

# ── Per-window bar chart ─────────────────────────────────────────
st.subheader("Sharpe Ratio — Per Walk-Forward Window")
windows = results.get("windows", [])

if windows:
    w_labels = [w["test_start"][:7] for w in windows]
    fig = go.Figure()
    for key, label, color in [
        ("rl_sharpe",  f"RL-{algo}", "#185FA5"),
        ("mvo_sharpe", "Markowitz",  "#0F6E56"),
        ("ew_sharpe",  "Equal-Wt",   "#854F0B"),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=w_labels,
            y=[w[key] for w in windows],
            marker_color=color,
        ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.5)
    fig.update_layout(
        barmode="group", height=380,
        xaxis_title="Test Window Start",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Statistical significance ─────────────────────────────────────
st.markdown("---")
st.subheader("Statistical Significance")
rl_m   = results["strategies"].get("rl", {})
t_stat = rl_m.get("t_stat_vs_ew", 0)
p_val  = rl_m.get("p_value_vs_ew", 1)

c1, c2, c3 = st.columns(3)
c1.metric("t-statistic", f"{t_stat:.3f}")
c2.metric("p-value",     f"{p_val:.4f}")
c3.metric("Significant?", "YES" if p_val < 0.05 else "NO",
          delta="p < 0.05 threshold" if p_val < 0.05 else "Need p < 0.05")

if p_val < 0.05:
    st.success(
        "The RL agent alpha over equal-weight is statistically significant "
        "(p < 0.05). The excess returns are unlikely to be due to chance."
    )
else:
    st.warning(
        "Alpha is not statistically significant yet. "
        "Consider retraining with more timesteps."
    )
'''

path = Path("dashboard/pages/3_Comparison.py")
path.write_text(content, encoding="utf-8")
print(f"Written: {path.resolve()}")
print("Streamlit will hot-reload automatically — just refresh the browser.")