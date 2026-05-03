"""Page 1: Performance Overview — equity curves, drawdown, rolling Sharpe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.data_utils import (
    load_backtest_results, get_equity_df,
    get_rolling_sharpe_df,
)

st.set_page_config(page_title="Performance", page_icon="📊", layout="wide")
st.title("📊 Performance Overview")
st.caption("Walk-forward backtest results across 5 years and 11 market windows")

results = load_backtest_results()
if results is None:
    st.error("Backtest results not found. Run: python -m backtesting.walk_forward")
    st.stop()

strategies = results["strategies"]
rl  = strategies.get("rl", {})
algo = results.get("algo", "ppo").upper()

# ── KPI Cards ───────────────────────────────────────────────────
st.subheader("Key Metrics — RL Agent")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Sharpe Ratio",   f"{rl.get('sharpe',0):.3f}",   delta="vs 0.6 benchmark")
c2.metric("Sortino Ratio",  f"{rl.get('sortino',0):.3f}")
c3.metric("CAGR",           f"{rl.get('cagr',0):.1%}",     delta=f"+{rl.get('cagr',0)-0.08:.1%} vs 8% passive")
c4.metric("Max Drawdown",   f"{rl.get('max_drawdown',0):.1%}")
c5.metric("CVaR 95%",       f"{rl.get('cvar_95',0):.2%}")

p_val  = rl.get("p_value_vs_ew", 1.0)
sig    = "✅ p<0.05" if p_val < 0.05 else "❌ p≥0.05"
c6.metric("Alpha Sig.",     sig, delta=f"p={p_val:.4f}")

st.markdown("---")

# ── Equity Curves ───────────────────────────────────────────────
eq_df = get_equity_df(results)
st.subheader("Equity Curves")

fig1 = go.Figure()
colors = {"RL Agent": "#185FA5", "Markowitz": "#0F6E56", "Equal-Weight": "#854F0B"}
for col in eq_df.columns:
    fig1.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df[col],
        name=col, line=dict(color=colors.get(col, "#888"), width=2),
    ))
fig1.update_layout(
    height=380, hovermode="x unified",
    yaxis_title="Portfolio Value (start=1.0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=20),
)
st.plotly_chart(fig1, use_container_width=True)

# ── Drawdown + Rolling Sharpe side by side ──────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Drawdown")
    dates  = results.get("dates", [])
    dds    = results.get("drawdowns", {})
    fig2   = go.Figure()
    for key, label, color in [("rl","RL Agent","#185FA5"), ("equal_weight","Equal-Weight","#854F0B")]:
        if key in dds:
            fig2.add_trace(go.Scatter(
                x=pd.to_datetime(dates),
                y=[v*100 for v in dds[key]],
                name=label, fill="tozeroy",
                line=dict(color=color, width=1.5),
            ))
    fig2.update_layout(height=300, yaxis_title="Drawdown (%)",
                       hovermode="x unified", margin=dict(t=20,b=20))
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("Rolling 63-Day Sharpe")
    rs_df  = get_rolling_sharpe_df(results)
    fig3   = go.Figure()
    for col, color in [("RL Agent","#185FA5"), ("Equal-Weight","#854F0B")]:
        if col in rs_df.columns:
            fig3.add_trace(go.Scatter(
                x=rs_df.index, y=rs_df[col],
                name=col, line=dict(color=color, width=2),
            ))
    fig3.add_hline(y=1, line_dash="dot", line_color="#0F6E56",
                   annotation_text="Target Sharpe=1.0")
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig3.update_layout(height=300, yaxis_title="Sharpe",
                       hovermode="x unified", margin=dict(t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

# ── Per-window table ────────────────────────────────────────────
with st.expander("Per-Window Sharpe Ratios (all 11 windows)"):
    windows = results.get("windows", [])
    if windows:
        wdf = pd.DataFrame(windows)
        wdf = wdf.rename(columns={
            "test_start": "Window Start", "test_end": "Window End",
            "rl_sharpe": "RL Agent", "mvo_sharpe": "Markowitz",
            "ew_sharpe": "Equal-Weight",
        })
        wdf["RL Wins"] = wdf["RL Agent"] > wdf["Equal-Weight"]
        st.dataframe(wdf.drop(columns=["window"]).style.format({
            "RL Agent": "{:.3f}", "Markowitz": "{:.3f}", "Equal-Weight": "{:.3f}",
        }), use_container_width=True)