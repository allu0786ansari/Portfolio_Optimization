"""Asset universes (v1 constituents; point-in-time membership arrives in Phase 1)."""

from typing import Any

NIFTY_V1: tuple[str, ...] = (
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "NESTLEIND.NS", "WIPRO.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS",
)
SP500_V1: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "LLY", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA",
    "PG", "COST", "HD", "JNJ", "ABBV",
)
UNIVERSES: dict[str, dict[str, Any]] = {
    "nifty": {"tickers": NIFTY_V1, "benchmark": "nifty50", "benchmark_ticker": "^NSEI", "currency": "INR"},
    "sp500": {"tickers": SP500_V1, "benchmark": "sp500", "benchmark_ticker": "^GSPC", "currency": "USD"},
}
