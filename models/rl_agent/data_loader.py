"""Loads and aligns processed feature files for the RL environment.

The environment needs all assets aligned to the same trading dates.
This module handles the alignment and returns a clean dict of DataFrames.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from data.config import ALL_TICKERS, PROCESSED_DIR
from models.forecasting.dataset import FEATURE_COLS


def load_aligned_features(
    tickers: list[str] = ALL_TICKERS,
    feature_cols: list[str] = FEATURE_COLS,
    min_overlap_days: int = 500,
) -> tuple[dict[str, pd.DataFrame], list[str], pd.DatetimeIndex]:
    """Load feature files for all tickers, align to common trading dates.

    Returns:
        features_dict : {ticker -> DataFrame(dates, features)}
        valid_tickers : list of tickers that had sufficient data
        common_dates  : DatetimeIndex of overlapping trading days
    """
    raw: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        path = PROCESSED_DIR / f"features_{ticker.replace('/', '_')}.parquet"
        if not path.exists():
            logger.warning(f"Missing feature file for {ticker}")
            continue
        df = pd.read_parquet(path)[feature_cols]
        if df.isnull().any().any():
            df = df.dropna()
        raw[ticker] = df

    if not raw:
        raise RuntimeError("No feature files found. Run feature_engineering.py first.")

    # Find common date index across all tickers
    common_dates = None
    for df in raw.values():
        idx = df.index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)

    if len(common_dates) < min_overlap_days:
        raise RuntimeError(
            f"Only {len(common_dates)} overlapping days — need {min_overlap_days}. "
            "Check your date range and feature files."
        )

    # Align all DataFrames to common dates
    aligned: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        aligned[ticker] = df.loc[common_dates].copy()

    valid_tickers = list(aligned.keys())
    logger.info(
        f"Loaded {len(valid_tickers)} tickers with "
        f"{len(common_dates)} overlapping trading days "
        f"({common_dates[0].date()} to {common_dates[-1].date()})"
    )
    return aligned, valid_tickers, common_dates


def build_return_matrix(
    features_dict: dict[str, pd.DataFrame],
    tickers: list[str],
) -> np.ndarray:
    """Stack log_return series into (T, N) matrix for portfolio return calc."""
    returns = [features_dict[t]["log_return"].values for t in tickers]
    return np.stack(returns, axis=1)   # shape (T, N)