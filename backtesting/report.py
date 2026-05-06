"""Generate a self-contained HTML backtest report with interactive charts."""
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

INPUT_PATH  = Path("backtesting/backtest_results.json")
OUTPUT_PATH = Path("backtesting/backtest_report.html")

COLORS = {
    "rl":           "#185FA5",
    "markowitz":    "#0F6E56",
    "equal_weight": "#854F0B",
}


def build_report(results: dict) -> str:
    """Build full HTML report string from backtest results dict."""
    dates       = results["dates"]
    strategies  = results["strategies"]
    eq_curves   = results["equity_curves"]
    roll_sharpe = results["rolling_sharpe"]
    drawdowns   = results["drawdowns"]
    windows     = results["windows"]
    algo        = results["algo"].upper()

    # --- Figure 1: Equity curves ---
    fig1 = go.Figure()
    labels = {
        "rl":           f"RL-{algo} Agent",
        "markowitz":    "Markowitz (MVO)",
        "equal_weight": "Equal-Weight (1/N)",
    }
    for key, label in labels.items():
        if key in eq_curves:
            fig1.add_trace(go.Scatter(
                x=dates, y=eq_curves[key], name=label,
                line=dict(color=COLORS[key], width=2),
            ))

    # --- Figure 2: Drawdown ---
    fig2 = go.Figure()
    for key in ["rl", "equal_weight"]:
        if key in drawdowns:
            fig2.add_trace(go.Scatter(
                x=dates, y=[v * 100 for v in drawdowns[key]],
                name=labels[key], fill="tozeroy",
                line=dict(color=COLORS[key], width=1.5),
            ))

    # --- Figure 3: Rolling Sharpe ---
    fig3 = go.Figure()
    for key in ["rl", "equal_weight"]:
        if key in roll_sharpe:
            clean = [v if not np.isnan(v) else None for v in roll_sharpe[key]]
            fig3.add_trace(go.Scatter(
                x=dates, y=clean, name=labels[key],
                line=dict(color=COLORS[key], width=2),
            ))

    # --- Figure 4: Per-window Sharpe ---
    w_labels = [f"{w['test_start'][:7]}" for w in windows]
    fig4 = go.Figure()
    fig4.add_bar(name=f"RL-{algo}", x=w_labels, y=[w["rl_sharpe"] for w in windows])
    fig4.add_bar(name="Markowitz", x=w_labels, y=[w["mvo_sharpe"] for w in windows])
    fig4.add_bar(name="Equal-Wt", x=w_labels, y=[w["ew_sharpe"] for w in windows])

    # --- Metrics ---
    rl  = strategies["rl"]
    mvo = strategies.get("markowitz", {})
    ew  = strategies.get("equal_weight", {})

    def fmt_pct(v): return f"{v:.1%}" if v else "—"
    def fmt_f(v): return f"{v:.3f}" if v else "—"

    rows_html = ""
    metrics_rows = [
        ("Sharpe Ratio", fmt_f(rl.get("sharpe")), fmt_f(mvo.get("sharpe")), fmt_f(ew.get("sharpe"))),
        ("CAGR", fmt_pct(rl.get("cagr")), fmt_pct(mvo.get("cagr")), fmt_pct(ew.get("cagr"))),
    ]

    # ✅ FIXED: properly indented inside function
    for label, r1, r2, r3 in metrics_rows:
        rows_html += f"<tr><td>{label}</td><td><strong>{r1}</strong></td><td>{r2}</td><td>{r3}</td></tr>\n"

    # ✅ FIXED: html also inside function
    html = f"""
    <html>
    <body>
    <h1>Backtest Report</h1>
    {pio.to_html(fig1, full_html=False)}
    {pio.to_html(fig2, full_html=False)}
    {pio.to_html(fig3, full_html=False)}
    {pio.to_html(fig4, full_html=False)}
    <table>{rows_html}</table>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        logger.error("Run walk_forward first")
        raise SystemExit(1)

    results = json.loads(INPUT_PATH.read_text())
    html = build_report(results)
    OUTPUT_PATH.write_text(html, encoding="utf-8" )
    webbrowser.open(OUTPUT_PATH.resolve().as_uri())