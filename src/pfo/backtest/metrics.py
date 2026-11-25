"""Performance statistics with honest inference.

All inputs are *daily simple returns* on a monotonic, non-duplicated index.
Every function refuses duplicated dates — the v1 backtest concatenated
overlapping windows, which silently inflated the sample size six-fold.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def _validate(r: pd.Series) -> np.ndarray:
    if isinstance(r, pd.Series):
        if r.index.has_duplicates:
            raise ValueError("return series has duplicated dates (overlapping windows?)")
        if not r.index.is_monotonic_increasing:
            raise ValueError("return series dates are not monotonic")
        return r.to_numpy(dtype=float)
    return np.asarray(r, dtype=float)


def sharpe(r: pd.Series | np.ndarray, rf_daily: float = 0.0) -> float:
    x = _validate(r) - rf_daily
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sortino(r: pd.Series | np.ndarray, rf_daily: float = 0.0) -> float:
    x = _validate(r) - rf_daily
    if len(x) < 2:
        return float("nan")
    dd = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    return float(x.mean() / dd * math.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


def cagr(r: pd.Series | np.ndarray) -> float:
    x = _validate(r)
    if len(x) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + x))
    years = len(x) / TRADING_DAYS
    return growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0


def annual_vol(r: pd.Series | np.ndarray) -> float:
    x = _validate(r)
    return float(x.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(x) > 1 else float("nan")


def max_drawdown(r: pd.Series | np.ndarray) -> float:
    x = _validate(r)
    if len(x) == 0:
        return 0.0
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.min(eq / peak - 1.0))


def var_cvar(r: pd.Series | np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    x = _validate(r)
    var = float(np.quantile(x, 1.0 - alpha))
    tail = x[x <= var]
    return var, float(tail.mean()) if len(tail) else var


def summary(r: pd.Series, turnover: pd.Series | None = None, costs: pd.Series | None = None) -> dict:
    var95, cvar95 = var_cvar(r)
    out = {
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "n_days": int(len(r)),
        "cagr": cagr(r),
        "ann_vol": annual_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
        "calmar": (cagr(r) / abs(max_drawdown(r))) if max_drawdown(r) < 0 else float("nan"),
        "var_95": var95,
        "cvar_95": cvar95,
    }
    years = len(r) / TRADING_DAYS
    if turnover is not None:
        out["annual_turnover"] = float(turnover.sum() / years) if years > 0 else float("nan")
    if costs is not None:
        out["annual_cost_drag"] = float(costs.sum() / years) if years > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
    return idx


def sharpe_confidence_interval(
    r: pd.Series | np.ndarray,
    level: float = 0.95,
    n_boot: int = 1000,
    mean_block: float = 21.0,
    seed: int = 0,
) -> tuple[float, float]:
    x = _validate(r)
    rng = np.random.default_rng(seed)
    boots = np.array([sharpe(x[_stationary_bootstrap_indices(len(x), mean_block, rng)]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """t-statistic of the mean of ``x`` with Newey-West (Bartlett) HAC variance."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), 0
    if lags is None:
        lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    s = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 1e-30) / n)
    t = float(x.mean() / se)
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return t, p, lags


def paired_monthly_test(a: pd.Series, b: pd.Series) -> dict:
    """Compare two strategies on *non-overlapping calendar months*.

    Daily returns are compounded within each month; the monthly excess series
    is tested with a Newey-West t-test.
    """
    _validate(a)
    _validate(b)
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    monthly = (1.0 + df).groupby(df.index.to_period("M")).prod() - 1.0
    excess = (monthly["a"] - monthly["b"]).to_numpy()
    t, p, lags = newey_west_tstat(excess)
    return {
        "n_months": int(len(excess)),
        "mean_monthly_excess": float(excess.mean()) if len(excess) else float("nan"),
        "t_stat_nw": t,
        "p_value": p,
        "nw_lags": lags,
    }


def sharpe_difference_bootstrap(
    a: pd.Series, b: pd.Series, n_boot: int = 1000, mean_block: float = 21.0, seed: int = 0
) -> dict:
    """Paired stationary-bootstrap test of Sharpe(a) - Sharpe(b)."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    x, y = df["a"].to_numpy(), df["b"].to_numpy()
    rng = np.random.default_rng(seed)
    obs = sharpe(x) - sharpe(y)
    diffs = []
    for _ in range(n_boot):
        ii = _stationary_bootstrap_indices(len(x), mean_block, rng)
        diffs.append(sharpe(x[ii]) - sharpe(y[ii]))
    d = np.array(diffs)
    d = d[np.isfinite(d)]
    # two-sided p-value of H0: diff == 0 using the centred bootstrap distribution
    p = float(np.mean(np.abs(d - d.mean()) >= abs(obs)))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"sharpe_diff": float(obs), "ci95": [float(lo), float(hi)], "p_value": p}
