"""Unit tests for the Week 1 data pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.config import ALL_TICKERS, MOMENTUM_WINDOWS, NIFTY50_TICKERS, SP500_TICKERS
from data.ingestion.feature_engineering import (
    compute_beta,
    compute_momentum,
    compute_returns,
    compute_rsi,
    compute_volatility,
)

# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_prices() -> pd.Series:
    """300 days of synthetic price data following a random walk."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 300)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


@pytest.fixture
def sample_returns(sample_prices) -> pd.Series:
    return compute_returns(sample_prices)


# ── Config tests ──────────────────────────────────────────────────

def test_ticker_lists_non_empty():
    assert len(NIFTY50_TICKERS) == 20, "Expected 20 Nifty50 tickers"
    assert len(SP500_TICKERS) == 20, "Expected 20 S&P500 tickers"
    assert len(ALL_TICKERS) == 40


def test_no_duplicate_tickers():
    assert len(ALL_TICKERS) == len(set(ALL_TICKERS)), "Duplicate tickers found"


def test_nifty_tickers_have_ns_suffix():
    for t in NIFTY50_TICKERS:
        assert t.endswith(".NS"), f"{t} should end with .NS for NSE"


# ── Feature computation tests ─────────────────────────────────────

def test_compute_returns_shape(sample_prices, sample_returns):
    assert len(sample_returns) == len(sample_prices)


def test_compute_returns_first_is_nan(sample_returns):
    assert pd.isna(sample_returns.iloc[0]), "First log return must be NaN"


def test_compute_returns_reasonable_range(sample_returns):
    valid = sample_returns.dropna()
    assert valid.abs().max() < 0.5, "Daily returns > 50% are suspicious"


def test_momentum_columns(sample_returns):
    mom = compute_momentum(sample_returns, MOMENTUM_WINDOWS)
    assert set(mom.columns) == {f"momentum_{w}d" for w in MOMENTUM_WINDOWS}


def test_momentum_shape(sample_returns):
    mom = compute_momentum(sample_returns, MOMENTUM_WINDOWS)
    assert len(mom) == len(sample_returns)


def test_volatility_non_negative(sample_returns):
    vol = compute_volatility(sample_returns, 21)
    assert (vol.dropna() >= 0).all(), "Volatility must be non-negative"


def test_beta_calculation(sample_returns):
    # Use same series as benchmark — beta should converge to 1.0
    beta = compute_beta(sample_returns, sample_returns, window=63)
    valid = beta.dropna()
    assert len(valid) > 0, "No valid beta values computed"
    # Beta of an asset against itself should be ~1.0
    assert abs(valid.mean() - 1.0) < 0.05, f"Self-beta should be ~1.0, got {valid.mean():.3f}"


def test_rsi_bounded(sample_returns):
    rsi = compute_rsi(sample_returns, 14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]"


# ── File existence tests (run after pipeline executes) ────────────

def test_raw_data_files_exist():
    from data.config import RAW_DIR
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    # Lenient check — at least some files should exist after running fetch_data.py
    # Skip this test if data hasn't been downloaded yet
    if len(parquet_files) == 0:
        pytest.skip("Raw data not downloaded yet — run fetch_data.py first")
    assert len(parquet_files) >= 5, f"Expected at least 5 Parquet files, found {len(parquet_files)}"


def test_processed_features_schema():
    from data.config import PROCESSED_DIR
    feature_files = list(PROCESSED_DIR.glob("features_*.parquet"))
    if len(feature_files) == 0:
        pytest.skip("Processed features not generated yet — run feature_engineering.py first")

    df = pd.read_parquet(feature_files[0])
    required_cols = ["close", "log_return", "momentum_21d", "volatility_21d", "beta_63d", "rsi_14"]
    for col in required_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    # No NaNs in the final processed file (we dropped them in feature engineering)
    assert df.isnull().sum().sum() == 0, "Processed features should have no NaN values"