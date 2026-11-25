"""Causal per-asset features (the v1 feature set, recomputed safely).

Input is a wide price panel (rows = trading days of ONE exchange calendar,
columns = assets). Every feature at row ``t`` uses prices at rows ``<= t``
only; this is verified by the future-shuffle test in ``tests/leakage``.

Differences from v1:
* RSI is computed on price changes (not log returns) and rescaled to [-1, 1].
* Beta uses the same window including day ``t`` (still causal).
* No clipping: features are standardized by a fold-fitted scaler instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "volatility_21d",
    "volatility_63d",
    "beta_63d",
    "rsi_14",
    "price_vs_ma50",
    "price_vs_ma200",
)
WARMUP_DAYS = 200  # longest look-back (200-day moving average)


def compute_features(prices: pd.DataFrame, market: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """Return a dict ``name -> DataFrame (T x N)`` of causal features.

    ``market`` is the benchmark price series used for beta. If omitted, an
    equal-weight index of the panel is used.
    """
    prices = prices.sort_index()
    logp = np.log(prices)
    lr = logp.diff()
    if market is None:
        mkt_r = prices.pct_change(fill_method=None).mean(axis=1)
    else:
        mkt_r = market.reindex(prices.index).pct_change(fill_method=None)
    mkt_lr = np.log1p(mkt_r)

    feats: dict[str, pd.DataFrame] = {"log_return": lr}
    for w in (21, 63, 126):
        feats[f"momentum_{w}d"] = lr.rolling(w).sum()
    for w in (21, 63):
        feats[f"volatility_{w}d"] = lr.rolling(w).std() * np.sqrt(252)

    cov = lr.rolling(63).cov(mkt_lr)
    var = mkt_lr.rolling(63).var()
    feats["beta_63d"] = cov.div(var.replace(0.0, np.nan), axis=0)

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0.0, np.nan))
    feats["rsi_14"] = (rsi - 50.0) / 50.0

    feats["price_vs_ma50"] = prices / prices.rolling(50).mean() - 1.0
    feats["price_vs_ma200"] = prices / prices.rolling(200).mean() - 1.0
    return {k: feats[k] for k in FEATURE_NAMES}


def to_tensor(features: dict[str, pd.DataFrame], names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    """Stack into an array of shape (T, N, F)."""
    return np.stack([features[n].to_numpy(dtype=float) for n in names], axis=-1)


@dataclass
class FoldScaler:
    """Per-feature standardization fitted on a training window only.

    Statistics are pooled over time and assets of the training rows; the
    same parameters are then applied to any later rows (validation/test).
    """

    clip: float = 5.0
    mean_: np.ndarray | None = field(default=None, init=False)
    std_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray, rows: range | slice) -> FoldScaler:
        block = x[rows] if isinstance(rows, slice) else x[rows.start : rows.stop]
        flat = block.reshape(-1, block.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FoldScaler.transform called before fit")
        z = (x - self.mean_) / self.std_
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(z, -self.clip, self.clip)
