"""Downloads OHLCV price data for all tickers and saves as Parquet files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yfinance as yf
import pandas as pd
from loguru import logger

from data.config import (
    RAW_DIR, START_DATE, END_DATE,
    ALL_TICKERS, BENCHMARKS
)


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker. Returns None on failure."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # adjusted close — handles splits + dividends
            progress=False,
        )
        if df.empty or len(df) < 50:
            logger.warning(f"Skipping {ticker}: insufficient data ({len(df)} rows)")
            return None

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        logger.info(f"Downloaded {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        return df

    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}")
        return None


def fetch_all_tickers(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Fetch all tickers and save each as a Parquet file. Returns dict of successful downloads."""
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        out_path = output_dir / f"{ticker.replace('/', '_')}.parquet"

        if out_path.exists():
            logger.info(f"Cache hit: {ticker} — loading from disk")
            results[ticker] = pd.read_parquet(out_path)
            continue

        df = fetch_ticker(ticker, start, end)
        if df is not None:
            df.to_parquet(out_path, index=True)
            results[ticker] = df

    logger.info(f"Fetched {len(results)}/{len(tickers)} tickers successfully")
    return results


def fetch_benchmarks(
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Fetch benchmark index data (Nifty50, S&P500)."""
    results = {}
    for name, ticker in BENCHMARKS.items():
        out_path = output_dir / f"benchmark_{name}.parquet"
        if out_path.exists():
            results[name] = pd.read_parquet(out_path)
            continue
        df = fetch_ticker(ticker, start, end)
        if df is not None:
            df.to_parquet(out_path, index=True)
            results[name] = df
            logger.info(f"Saved benchmark {name} ({ticker})")
    return results


if __name__ == "__main__":
    logger.info("=== Starting data fetch ===")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Total tickers: {len(ALL_TICKERS)}")

    fetch_all_tickers(ALL_TICKERS)
    fetch_benchmarks()

    logger.info("=== Data fetch complete ===")