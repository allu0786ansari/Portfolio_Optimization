"""Page 4: Efficient Frontier — Markowitz risk-return visualisation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_utils import load_backtest_results
from models.rl_agent.data_loader import load_aligned_features, build_return_matrix
from models.classical.markowitz import ledoit_wolf_cov
from data.config import ALL_TICKERS

st.set_page_config(page_title="Frontier", page_icon="🔵", layout="wide")
st.title("🔵 Markowitz Efficient Frontier")
st.caption("Risk-return tradeoff for random portfolios vs optimal frontier — with RL agent plotted")

results = load_backtest_results()

# ── Controls ─────────────────────────────────────────────────────
n_portfolios = st.slider("Number of random portfolios to simulate", 500, 5000, 2000, step=500)

with st.spinner("Computing efficient frontier..."):
    try:
        features_dict, tickers, dates = load_aligned_features(ALL_TICKERS)
        return_matrix = build_return_matrix(features_dict, tickers)

        # Use last 2 years for frontier computation
        lookback = min(504, len(dates))
        ret = return_matrix[-lookback:]
        mu  = ret.mean(axis=0) * 252           # annualised mean returns
        cov = ledoit_wolf_cov(ret) * 252       # annualised covariance
        n   = len(tickers)

        # Simulate random portfolios
        np.random.seed(42)
        port_returns = []
        port_vols    = []
        port_sharpes = []

        for _ in range(n_portfolios):
            w  = np.random.dirichlet(np.ones(n))
            pr = float(w @ mu)
            pv = float(np.sqrt(w @ cov @ w))
            port_returns.append(pr)
            port_vols.append(pv)
            port_sharpes.append(pr / pv if pv > 0 else 0)

        port_returns = np.array(port_returns)
        port_vols    = np.array(port_vols)
        port_sharpes = np.array(port_sharpes)

        # RL agent point from backtest results
        rl_ret = rl_vol = None
        if results:
            rl_m   = results["strategies"].get("rl", {})
            rl_ret = rl_m.get("cagr", None)
            rl_std = rl_m.get("vol_daily", None)
            if rl_std:
                rl_vol = rl_std * np.sqrt(252)

        # ── Plot ────────────────────────────────────────────────
        fig = go.Figure()

        # Random portfolios coloured by Sharpe
        fig.add_trace(go.Scatter(
            x=port_vols * 100,
            y=port_returns * 100,
            mode="markers",
            marker=dict(
                size=4,
                color=port_sharpes,
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                opacity=0.6,
            ),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))

        # Max-Sharpe portfolio
        best_idx = np.argmax(port_sharpes)
        fig.add_trace(go.Scatter(
            x=[port_vols[best_idx] * 100],
            y=[port_returns[best_idx] * 100],
            mode="markers",
            marker=dict(size=16, color="#FFD700", symbol="star"),
            name=f"Max Sharpe ({port_sharpes[best_idx]:.2f})",
        ))

        # RL agent point
        if rl_ret is not None and rl_vol is not None:
            fig.add_trace(go.Scatter(
                x=[rl_vol * 100],
                y=[rl_ret * 100],
                mode="markers",
                marker=dict(size=16, color="#185FA5", symbol="diamond"),
                name=f"RL Agent (Sharpe={rl_ret/rl_vol:.2f})" if rl_vol > 0 else "RL Agent",
            ))

        fig.update_layout(
            height=520,
            xaxis_title="Annualised Volatility (%)",
            yaxis_title="Annualised Return (%)",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Stats ────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Sharpe Portfolio", f"{port_sharpes[best_idx]:.3f}")
        c2.metric("Best Portfolio Return", f"{port_returns[best_idx]*100:.1f}%")
        c3.metric("Best Portfolio Vol",   f"{port_vols[best_idx]*100:.1f}%")

    except Exception as e:
        st.error(f"Error computing frontier: {e}")
        st.info("Make sure feature engineering has been run: python -m data.ingestion.feature_engineering")