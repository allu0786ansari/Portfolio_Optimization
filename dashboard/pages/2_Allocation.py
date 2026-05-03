"""Page 2: Live Allocation — calls FastAPI /predict in real time."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from dashboard.data_utils import call_predict_api
from data.config import NIFTY50_TICKERS, SP500_TICKERS

st.set_page_config(page_title="Allocation", page_icon="🥧", layout="wide")
st.title("🥧 Live Portfolio Allocation")
st.caption("Calls the FastAPI /predict endpoint in real time and displays weights")

# ── Ticker selection ─────────────────────────────────────────────
st.subheader("Select Assets")
col_left, col_right = st.columns(2)

with col_left:
    nifty_selected = st.multiselect(
        "Nifty50 stocks",
        options=NIFTY50_TICKERS,
        default=NIFTY50_TICKERS[:5],
        help="Indian market stocks (NSE)"
    )

with col_right:
    sp500_selected = st.multiselect(
        "S&P500 stocks",
        options=SP500_TICKERS,
        default=SP500_TICKERS[:5],
        help="US market stocks"
    )

selected = nifty_selected + sp500_selected

st.info(f"Selected: **{len(selected)} assets**. Minimum 2 required.")

if len(selected) < 2:
    st.warning("Please select at least 2 assets.")
    st.stop()

# ── Predict button ───────────────────────────────────────────────
if st.button("🚀 Get Portfolio Allocation", type="primary", use_container_width=True):
    with st.spinner("Calling API..."):
        result = call_predict_api(selected)

    if result is None or "error" in result:
        err = result.get("error", "Unknown error") if result else "No response"
        st.error(f"API Error: {err}")
        st.info("Make sure the FastAPI server is running: uvicorn serving.main:app --port 8000")
    else:
        weights = result["weights"]
        sorted_w = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

        # ── Metadata row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model",       result.get("model_version", "—"))
        m2.metric("Algorithm",   result.get("algo", "—"))
        m3.metric("Latency",     f"{result.get('latency_ms',0):.1f} ms")
        m4.metric("Weights Sum", f"{result.get('weights_sum',0):.6f}")

        st.markdown("---")

        # ── Charts side by side
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Allocation Donut")
            fig_donut = go.Figure(go.Pie(
                labels = list(sorted_w.keys()),
                values = list(sorted_w.values()),
                hole   = 0.45,
                textinfo = "label+percent",
                hovertemplate = "%{label}: %{value:.4f}<extra></extra>",
            ))
            fig_donut.update_layout(
                height=380, margin=dict(t=20,b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("Weight Ranking")
            fig_bar = go.Figure(go.Bar(
                x = list(sorted_w.values()),
                y = list(sorted_w.keys()),
                orientation = "h",
                marker_color = "#185FA5",
                text = [f"{v:.3f}" for v in sorted_w.values()],
                textposition = "outside",
            ))
            fig_bar.update_layout(
                height=380, xaxis_title="Weight",
                margin=dict(t=20, b=20, r=60),
                xaxis=dict(range=[0, max(sorted_w.values()) * 1.2]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Raw weights table
        with st.expander("Raw weights table"):
            import pandas as pd
            df = pd.DataFrame(sorted_w.items(), columns=["Ticker", "Weight"])
            df["Weight %"] = (df["Weight"] * 100).round(2).astype(str) + "%"
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Skipped tickers
        skipped = result.get("skipped_tickers", [])
        if skipped:
            st.warning(f"Tickers not in feature store (skipped): {skipped}")