"""Downloads macroeconomic indicators: VIX, US 10Y yield, sector ETFs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger

from data.config import RAW_DIR, PROCESSED_DIR, START_DATE, END_DATE


# Macro tickers available via yfinance
MACRO_TICKERS: dict[str, str] = {
    "vix": "^VIX",              # CBOE Volatility Index
    "us_10y_yield": "^TNX",     # US 10-year treasury yield
    "india_vix": "^INDIAVIX",   # India VIX (may be unavailable — handled gracefully)
    "gold": "GC=F",             # Gold futures (safe-haven signal)
    "dxy": "DX-Y.NYB",          # US Dollar Index
}

# US sector ETFs (useful regime signals)
SECTOR_ETFS: dict[str, str] = {
    "sector_tech": "XLK",
    "sector_finance": "XLF",
    "sector_energy": "XLE",
    "sector_health": "XLV",
}


def fetch_series(ticker: str, name: str, start: str, end: str) -> pd.Series | None:
    """Download a single time series and return as a named Series."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            logger.warning(f"No data for macro ticker {ticker} ({name})")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].squeeze()
        close.name = name
        close.index = pd.to_datetime(close.index)
        return close
    except Exception as e:
        logger.warning(f"Could not fetch {ticker} ({name}): {e}")
        return None


def build_macro_features(
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download all macro series, compute derived features, return aligned DataFrame."""
    series_list = []

    # Download all macro tickers
    for name, ticker in {**MACRO_TICKERS, **SECTOR_ETFS}.items():
        s = fetch_series(ticker, name, start, end)
        if s is not None:
            series_list.append(s)

    if not series_list:
        raise RuntimeError("No macro data could be downloaded")

    macro = pd.concat(series_list, axis=1)
    macro = macro.sort_index()
    macro.index.name = "date"

    # Derived features
    if "vix" in macro.columns:
        macro["vix_change_1d"] = macro["vix"].pct_change()
        macro["vix_ma10"] = macro["vix"].rolling(10).mean()
        macro["vix_regime"] = (macro["vix"] > macro["vix"].rolling(63).mean()).astype(int)

    if "us_10y_yield" in macro.columns:
        macro["yield_change_1d"] = macro["us_10y_yield"].diff()

    # Sector ETF returns
    for col in SECTOR_ETFS:
        if col in macro.columns:
            macro[f"{col}_return_5d"] = np.log(
                macro[col] / macro[col].shift(5)
            )

    # Forward-fill weekends and holidays, then drop NaN rows
    macro = macro.ffill().dropna(how="all")

    logger.info(f"Macro features: {macro.shape} — columns: {macro.columns.tolist()}")
    return macro


if __name__ == "__main__":
    logger.info("=== Fetching macro features ===")
    macro = build_macro_features()

    out_path = PROCESSED_DIR / "macro_features.parquet"
    macro.to_parquet(out_path, index=True)
    logger.info(f"Saved macro features to {out_path}")
    logger.info("=== Macro features complete ===")