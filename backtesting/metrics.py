"""Risk and return metrics for portfolio backtesting.

All metrics are computed from a list/array of daily log returns.
Annualisation assumes 252 trading days per year.
"""
import numpy as np
from scipy import stats


TRADING_DAYS = 252
RISK_FREE_DAILY = 0.0   # simplified: 0% daily


def sharpe_ratio(returns: np.ndarray, rf: float = RISK_FREE_DAILY) -> float:
    """Annualised Sharpe ratio."""
    r = np.asarray(returns)
    if len(r) < 2:
        return 0.0

    excess = r - rf
    mean = excess.mean()
    std = r.std()

    # 🔥 FIX: handle zero volatility correctly
    if std < 1e-10:
        if mean > 0:
            return 1e6
        elif mean < 0:
            return -1e6
        else:
            return 0.0

    return float(mean / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, rf: float = RISK_FREE_DAILY) -> float:
    """Annualised Sortino ratio — penalises only downside deviation."""
    r = np.asarray(returns)
    if len(r) < 2:
        return 0.0

    excess = r - rf
    downside = excess[excess < 0]

    if len(downside) == 0:
        return float(excess.mean() * np.sqrt(TRADING_DAYS) / 1e-10)

    down_std = float(np.sqrt(np.mean(downside ** 2)))

    return float(excess.mean() / down_std * np.sqrt(TRADING_DAYS)) if down_std > 1e-10 else 0.0


def calmar_ratio(returns: np.ndarray) -> float:
    """Calmar ratio = CAGR / abs(max drawdown)."""
    cagr = annualised_return(returns)
    mdd  = max_drawdown(returns)
    return float(cagr / abs(mdd)) if abs(mdd) > 1e-10 else 0.0


def annualised_return(returns: np.ndarray) -> float:
    """Compound Annual Growth Rate from log returns."""
    r = np.asarray(returns)
    if len(r) == 0:
        return 0.0

    total_log_return = float(r.sum())
    n_years = len(r) / TRADING_DAYS

    return float(np.exp(total_log_return / max(n_years, 1e-8)) - 1)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (negative value)."""
    r = np.asarray(returns)
    if len(r) == 0:
        return 0.0

    equity = np.exp(np.cumsum(r))
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak

    return float(dd.min())


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (negative value)."""
    r = np.asarray(returns)
    if len(r) == 0:
        return 0.0

    return float(np.percentile(r, (1 - confidence) * 100))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) — mean of returns below VaR (negative)."""
    r = np.asarray(returns)
    if len(r) == 0:
        return 0.0

    var   = value_at_risk(r, confidence)
    below = r[r <= var]

    return float(below.mean()) if len(below) > 0 else var


def equity_curve(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """Cumulative portfolio value from log returns."""
    return initial * np.exp(np.cumsum(np.asarray(returns)))


def rolling_sharpe(
    returns: np.ndarray,
    window: int = 63,
) -> np.ndarray:
    """Rolling annualised Sharpe ratio with given lookback window."""
    r = np.asarray(returns, dtype=float)
    out = np.full(len(r), np.nan)

    for i in range(window, len(r)):
        w = r[i - window : i]
        mean = w.mean()
        std = w.std()

        # 🔥 FIX: same zero-volatility handling
        if std < 1e-10:
            if mean > 0:
                out[i] = 1e6
            elif mean < 0:
                out[i] = -1e6
            else:
                out[i] = 0.0
        else:
            out[i] = float(mean / std * np.sqrt(TRADING_DAYS))

    return out


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Per-day drawdown from peak (non-positive values)."""
    equity = np.exp(np.cumsum(np.asarray(returns)))
    peak   = np.maximum.accumulate(equity)
    return (equity - peak) / peak


def compute_all_metrics(returns: np.ndarray, label: str = "") -> dict:
    """Compute the full metrics suite for one return series."""
    r = np.asarray(returns)

    return {
        "label":        label,
        "n_days":       len(r),
        "sharpe":       sharpe_ratio(r),
        "sortino":      sortino_ratio(r),
        "calmar":       calmar_ratio(r),
        "cagr":         annualised_return(r),
        "max_drawdown": max_drawdown(r),
        "var_95":       value_at_risk(r, 0.95),
        "cvar_95":      conditional_var(r, 0.95),
        "mean_daily":   float(r.mean()) if len(r) > 0 else 0.0,
        "vol_daily":    float(r.std()) if len(r) > 0 else 0.0,
    }


def ttest_excess_returns(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    frequency: str = "monthly",
) -> tuple[float, float]:
    """Two-sided t-test on excess returns (strategy - benchmark)."""

    s = np.asarray(strategy_returns)
    b = np.asarray(benchmark_returns)

    min_len = min(len(s), len(b))
    s, b = s[:min_len], b[:min_len]

    excess = s - b

    # 🔥 FIX: identical series → no difference
    if np.allclose(excess, 0):
        return 0.0, 1.0

    if frequency == "monthly":
        step = 21
        monthly = [excess[i:i+step].sum() for i in range(0, len(excess) - step, step)]
        excess = np.array(monthly)

    if len(excess) < 5:
        return 0.0, 1.0

    t_stat, p_value = stats.ttest_1samp(excess, 0.0)

    # 🔥 FIX: handle NaN from scipy
    if np.isnan(t_stat) or np.isnan(p_value):
        return 0.0, 1.0

    return float(t_stat), float(p_value)