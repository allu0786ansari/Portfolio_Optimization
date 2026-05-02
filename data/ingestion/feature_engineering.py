"""Computes technical and price-based features for all downloaded tickers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from data.config import (
    RAW_DIR, PROCESSED_DIR, ALL_TICKERS,
    MOMENTUM_WINDOWS, VOLATILITY_WINDOW, BETA_WINDOW, MIN_HISTORY_DAYS, BENCHMARKS
)


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Log returns: ln(P_t / P_{t-1})."""
    return np.log(df["Close"] / df["Close"].shift(1))


def compute_momentum(returns: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Cumulative log return over each lookback window (momentum signal)."""
    feats = {}
    for w in windows:
        feats[f"momentum_{w}d"] = returns.rolling(w).sum()
    return pd.DataFrame(feats, index=returns.index)


def compute_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Annualised rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def compute_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling market beta against a benchmark index."""
    def _rolling_beta(idx: int) -> float:
        if idx < window:
            return np.nan
        a = asset_returns.iloc[idx - window : idx].values
        b = benchmark_returns.iloc[idx - window : idx].values
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 20:
            return np.nan
        cov = np.cov(a[mask], b[mask])
        var_b = cov[1, 1]
        return cov[0, 1] / var_b if var_b > 1e-10 else np.nan

    betas = [_rolling_beta(i) for i in range(len(asset_returns))]
    return pd.Series(betas, index=asset_returns.index, name=f"beta_{window}d")


def compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    gain = returns.clip(lower=0).rolling(window).mean()
    loss = (-returns.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_features_for_ticker(
    ticker: str,
    benchmark_returns: pd.Series,
) -> pd.DataFrame | None:
    """Load raw data for one ticker and compute all features."""
    raw_path = RAW_DIR / f"{ticker.replace('/', '_')}.parquet"
    if not raw_path.exists():
        logger.warning(f"No raw data for {ticker}, skipping feature engineering")
        return None

    df = pd.read_parquet(raw_path)
    if len(df) < MIN_HISTORY_DAYS:
        logger.warning(f"{ticker}: only {len(df)} rows — skipping (min {MIN_HISTORY_DAYS})")
        return None

    # Align benchmark to same dates
    bench = benchmark_returns.reindex(df.index)

    returns = compute_returns(df)
    features = pd.DataFrame(index=df.index)

    # Price and volume
    features["close"] = df["Close"]
    features["volume"] = df.get("Volume", pd.Series(np.nan, index=df.index))
    features["log_return"] = returns

    # Momentum features
    mom = compute_momentum(returns, MOMENTUM_WINDOWS)
    features = pd.concat([features, mom], axis=1)

    # Volatility
    features["volatility_21d"] = compute_volatility(returns, VOLATILITY_WINDOW)
    features["volatility_63d"] = compute_volatility(returns, 63)

    # Beta
    features["beta_63d"] = compute_beta(returns, bench, BETA_WINDOW)

    # RSI
    features["rsi_14"] = compute_rsi(returns, 14)

    # Price vs moving averages
    features["price_vs_ma50"] = df["Close"] / df["Close"].rolling(50).mean() - 1
    features["price_vs_ma200"] = df["Close"] / df["Close"].rolling(200).mean() - 1

    # Drop rows where features cannot be computed (beginning of series)
    features = features.dropna(subset=["momentum_126d", "beta_63d", "price_vs_ma200"])

    logger.info(f"Features for {ticker}: {features.shape} — {features.index[0].date()} to {features.index[-1].date()}")
    return features


def engineer_all_features(
    tickers: list[str] = ALL_TICKERS,
    output_dir: Path = PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    """Run feature engineering for all tickers. Returns dict of feature DataFrames."""
    # Load benchmark returns for beta calculation
    nifty_path = RAW_DIR / "benchmark_nifty50.parquet"
    sp500_path = RAW_DIR / "benchmark_sp500.parquet"

    if not nifty_path.exists() or not sp500_path.exists():
        raise FileNotFoundError(
            "Benchmark files missing. Run fetch_data.py first."
        )

    nifty_returns = compute_returns(pd.read_parquet(nifty_path))
    sp500_returns = compute_returns(pd.read_parquet(sp500_path))

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        # Use appropriate benchmark
        bench = nifty_returns if ticker.endswith(".NS") else sp500_returns
        feats = compute_features_for_ticker(ticker, bench)

        if feats is not None:
            out_path = output_dir / f"features_{ticker.replace('/', '_')}.parquet"
            feats.to_parquet(out_path, index=True)
            results[ticker] = feats

    logger.info(f"Feature engineering complete: {len(results)}/{len(tickers)} tickers")
    return results


if __name__ == "__main__":
    logger.info("=== Starting feature engineering ===")
    engineer_all_features()
    logger.info("=== Feature engineering complete ===")