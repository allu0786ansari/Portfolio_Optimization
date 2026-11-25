"""Market data containers and loaders (Phase 0: single-calendar universes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MarketData:
    """Prices for one universe on ONE exchange calendar (no cross-market mixing)."""

    name: str
    prices: pd.DataFrame  # T x N close prices (adjusted)
    benchmark: pd.Series | None = None  # benchmark index level, same calendar
    currency: str = "USD"

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change(fill_method=None)

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)


NIFTY_SUFFIX = ".NS"


def _read_close(path: Path) -> pd.Series | None:
    df = pd.read_parquet(path)
    for col in ("Close", "close", "Adj Close"):
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.index = pd.to_datetime(df.index)
            return s.astype(float)
    return None


def load_v1_universe(
    universe: str,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    min_coverage: float = 0.95,
) -> MarketData:
    """Load v1-downloaded prices for ``universe`` in {"nifty", "sp500"}.

    Prefers ``data/raw/<ticker>.parquet`` (Close); falls back to the v1
    processed feature files (``close`` column). Each universe stays on its own
    exchange calendar: dates are those on which the benchmark traded, and
    tickers with less than ``min_coverage`` of those dates are dropped.
    """
    from pfo.data.universes import UNIVERSES

    if universe not in UNIVERSES:
        raise ValueError("Phase 0 supports universe 'nifty' or 'sp500' (cross-market needs FX: Phase 1)")
    spec = UNIVERSES[universe]
    tickers, bench_name, ccy = tuple(spec["tickers"]), str(spec["benchmark"]), str(spec["currency"])

    series: dict[str, pd.Series] = {}
    for t in tickers:
        fname = t.replace("/", "_")
        s = None
        if (raw_dir / f"{fname}.parquet").exists():
            s = _read_close(raw_dir / f"{fname}.parquet")
        elif (processed_dir / f"features_{fname}.parquet").exists():
            s = _read_close(processed_dir / f"features_{fname}.parquet")
        if s is not None:
            series[t] = s
    if not series:
        raise FileNotFoundError(
            f"No price files for {universe} under {raw_dir} or {processed_dir}. "
            "Run `python -m data.ingestion.fetch_data` first."
        )
    bench = None
    bpath = raw_dir / f"benchmark_{bench_name}.parquet"
    if bpath.exists():
        bench = _read_close(bpath)
    panel = pd.DataFrame(series).sort_index()
    calendar = bench.dropna().index if bench is not None else panel.dropna(how="all").index
    panel = panel.reindex(calendar)
    coverage = panel.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = sorted(set(panel.columns) - set(keep))
    if dropped:
        logger.warning(f"{universe}: dropping {dropped} (coverage < {min_coverage:.0%})")
    # Forward-fill single missing quotes within the asset's own history only (no back-fill).
    panel = panel[keep].ffill(limit=3)
    first_full = panel.dropna().index.min()
    panel = panel.loc[first_full:].dropna(axis=0, how="any")
    if bench is not None:
        bench = bench.reindex(panel.index).ffill(limit=3)
    logger.info(f"{universe}: {panel.shape[1]} assets, {len(panel)} days "
                f"({panel.index[0].date()} -> {panel.index[-1].date()})")
    return MarketData(name=universe, prices=panel, benchmark=bench, currency=ccy)


def make_synthetic_market(
    n_assets: int = 20,
    n_days: int = 1800,
    n_factors: int = 3,
    seed: int = 0,
    start: str = "2016-01-01",
    name: str = "synthetic",
) -> MarketData:
    """Factor-model prices with volatility regimes and fat tails.

    Contains no exploitable predictability by construction (other than a
    tiny momentum component), so any strategy that shows a large edge on this
    data is suspect — a useful sanity check for leakage.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors)) * np.array([0.9, 0.4, 0.3][:n_factors])
    regime = np.where(rng.random(n_days) < 0.02, 1, 0).cumsum() % 2  # alternating calm / stressed
    vol_scale = np.where(regime == 1, 1.8, 1.0)
    f = rng.standard_t(df=5, size=(n_days, n_factors)) * 0.008 * vol_scale[:, None]
    eps = rng.standard_t(df=4, size=(n_days, n_assets)) * 0.012 * vol_scale[:, None]
    drift = rng.normal(0.0003, 0.0002, size=n_assets)
    r = drift + f @ loadings.T + eps
    r = np.clip(r, -0.25, 0.25)
    prices = 100.0 * np.cumprod(1.0 + r, axis=0)
    tickers = [f"SYN{i:02d}" for i in range(n_assets)]
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    bench = panel.pct_change().mean(axis=1).fillna(0.0).add(1.0).cumprod() * 100.0
    return MarketData(name=name, prices=panel, benchmark=bench, currency="USD")
