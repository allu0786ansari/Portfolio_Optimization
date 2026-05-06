"""Shared data loading utilities for all dashboard pages.

Uses st.cache_data to avoid reloading backtest results on every
page interaction — critical for dashboard responsiveness.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

RESULTS_PATH = Path("backtesting/backtest_results.json")
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "portfolio-secret-key-change-in-production")


@st.cache_data(ttl=300)   # cache for 5 minutes
def load_backtest_results() -> dict | None:
    """Load walk-forward backtest results from JSON file."""
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text())


@st.cache_data(ttl=300)
def get_equity_df(results: dict) -> pd.DataFrame:
    """Build a tidy DataFrame of equity curves for all strategies."""
    dates = results.get("dates", [])
    eq    = results.get("equity_curves", {})
    df = pd.DataFrame({
        "date":         pd.to_datetime(dates),
        "RL Agent":     eq.get("rl", []),
        "Markowitz":    eq.get("markowitz", []),
        "Equal-Weight": eq.get("equal_weight", []),
    })
    return df.set_index("date")


@st.cache_data(ttl=300)
def get_metrics_df(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame of all strategy metrics."""
    strategies = results.get("strategies", {})
    rows = []
    labels = {
        "rl":           f"RL-{results.get('algo','PPO').upper()} Agent",
        "markowitz":    "Markowitz (MVO)",
        "equal_weight": "Equal-Weight (1/N)",
    }
    metric_labels = {
        "sharpe":       "Sharpe Ratio",
        "sortino":      "Sortino Ratio",
        "calmar":       "Calmar Ratio",
        "cagr":         "CAGR",
        "max_drawdown": "Max Drawdown",
        "var_95":       "VaR 95%",
        "cvar_95":      "CVaR 95%",
    }
    for key, label in labels.items():
        s = strategies.get(key, {})
        row = {"Strategy": label}
        for mkey, mlabel in metric_labels.items():
            v = s.get(mkey, 0.0)
            if mkey in ("cagr", "max_drawdown", "var_95", "cvar_95"):
                row[mlabel] = f"{v:.1%}"
            else:
                row[mlabel] = f"{v:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


def call_predict_api(tickers: list[str]) -> dict | None:
    """Call the FastAPI /predict endpoint and return result dict."""
    try:
        import httpx
        r = httpx.post(
            f"{API_BASE_URL}/predict",
            json={"tickers": tickers},
            headers={"X-API-Key": API_KEY},
            timeout=10.0,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"API error {r.status_code}: {r.text}"}
    except Exception as e:
        return {"error": f"Could not reach API: {e}"}


def get_rolling_sharpe_df(results: dict) -> pd.DataFrame:
    """Build rolling Sharpe DataFrame."""
    dates = results.get("dates", [])
    rs    = results.get("rolling_sharpe", {})
    df = pd.DataFrame({
        "date":         pd.to_datetime(dates),
        "RL Agent":     [v if v == v else None for v in rs.get("rl", [])],
        "Equal-Weight": [v if v == v else None for v in rs.get("equal_weight", [])],
    })
    return df.set_index("date")