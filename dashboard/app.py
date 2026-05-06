"""Streamlit multi-page app entry point.

Run with: streamlit run dashboard/app.py

Pages are in dashboard/pages/ — Streamlit auto-discovers them
by filename prefix (1_, 2_, 3_, 4_) and shows them in the sidebar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title = "Portfolio Optimization Engine",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("📈 Portfolio Optimizer")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**AI-Powered Portfolio Allocation**

Trained with Reinforcement Learning (PPO/SAC) on Nifty50 + S&P500.
Walk-forward backtested over 5 years.

---
**Stack**
- RL Agent: Stable-Baselines3 (PPO)
- Tracking: MLflow
- API: FastAPI
- Backtest: Walk-forward (11 windows)
""")
st.sidebar.markdown("---")
st.sidebar.caption("Navigate using the pages above ↑")

# ── Home page ─────────────────────────────────────────────────────
st.title("Portfolio Optimization Engine")
st.markdown("""
An end-to-end AI engineering project demonstrating:
- **Reinforcement Learning** for portfolio allocation (PPO/SAC agents)
- **MLOps** with MLflow experiment tracking and model registry
- **Walk-forward backtesting** across 11 market windows (5 years)
- **Production serving** via FastAPI with auto model hot-reload
- **CI/CD** with Docker and GitHub Actions

---
**Navigate using the sidebar** to explore:
- 📊 **Performance** — equity curves, Sharpe, drawdown, rolling metrics
- 🥧 **Allocation** — live portfolio weights from the API
- ⚖️ **Comparison** — RL vs Markowitz vs Equal-Weight
- 🔵 **Frontier** — Markowitz efficient frontier visualisation
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Assets", "40", "Nifty50 + S&P500")
with col2:
    st.metric("Backtest Period", "5 Years", "11 walk-forward windows")
with col3:
    st.metric("Strategy", "RL-PPO", "Champion model")