"""Central configuration loaded from .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_DATE: str = os.getenv("START_DATE", "2019-01-01")
END_DATE: str = os.getenv("END_DATE", "2024-12-31")

NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "NESTLEIND.NS", "WIPRO.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS",
]

SP500_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "LLY", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA",
    "PG", "COST", "HD", "JNJ", "ABBV",
]

ALL_TICKERS: list[str] = NIFTY50_TICKERS + SP500_TICKERS

BENCHMARKS: dict[str, str] = {
    "nifty50": "^NSEI",
    "sp500": "^GSPC",
}

MOMENTUM_WINDOWS: list[int] = [21, 63, 126]
VOLATILITY_WINDOW: int = 21
BETA_WINDOW: int = 63
MIN_HISTORY_DAYS: int = 252
