"""Unit tests for Week 5 — risk metrics and backtesting utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from backtesting.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio,
    annualised_return, max_drawdown,
    value_at_risk, conditional_var,
    equity_curve, rolling_sharpe,
    drawdown_series, compute_all_metrics,
    ttest_excess_returns,
)


@pytest.fixture
def flat_positive():
    return np.array([0.001] * 252)

@pytest.fixture
def flat_negative():
    return np.array([-0.001] * 252)

@pytest.fixture
def random_returns():
    np.random.seed(42)
    return np.random.normal(0.0005, 0.012, 504)


# ── Sharpe ───────────────────────────────────────────────────────

def test_sharpe_positive_returns(flat_positive):
    assert sharpe_ratio(flat_positive) > 0

def test_sharpe_negative_returns(flat_negative):
    assert sharpe_ratio(flat_negative) < 0

def test_sharpe_empty():
    assert sharpe_ratio(np.array([])) == 0.0

def test_sharpe_reasonable_range(random_returns):
    s = sharpe_ratio(random_returns)
    assert -5 < s < 10


# ── Sortino ──────────────────────────────────────────────────────

def test_sortino_no_downside(flat_positive):
    s = sortino_ratio(flat_positive)
    assert s > 0

def test_sortino_gte_sharpe_for_positive(flat_positive):
    # With no downside, Sortino >= Sharpe always
    assert sortino_ratio(flat_positive) >= sharpe_ratio(flat_positive)

def test_sortino_negative(flat_negative):
    assert sortino_ratio(flat_negative) < 0


# ── Drawdown ─────────────────────────────────────────────────────

def test_max_drawdown_non_positive(random_returns):
    assert max_drawdown(random_returns) <= 0

def test_max_drawdown_flat_positive(flat_positive):
    assert max_drawdown(flat_positive) == 0.0

def test_drawdown_series_non_positive(random_returns):
    dd = drawdown_series(random_returns)
    assert (dd <= 1e-8).all()

def test_drawdown_series_length(random_returns):
    dd = drawdown_series(random_returns)
    assert len(dd) == len(random_returns)


# ── Equity curve ─────────────────────────────────────────────────

def test_equity_curve_starts_at_one(random_returns):
    eq = equity_curve(random_returns)
    assert abs(eq[0] - np.exp(random_returns[0])) < 1e-8

def test_equity_curve_positive(random_returns):
    eq = equity_curve(random_returns)
    assert (eq > 0).all()

def test_equity_curve_length(random_returns):
    eq = equity_curve(random_returns)
    assert len(eq) == len(random_returns)


# ── VaR / CVaR ───────────────────────────────────────────────────

def test_var_negative(random_returns):
    assert value_at_risk(random_returns, 0.95) < 0

def test_cvar_lte_var(random_returns):
    var  = value_at_risk(random_returns, 0.95)
    cvar = conditional_var(random_returns, 0.95)
    assert cvar <= var   # CVaR is worse (more negative) than VaR


# ── CAGR ─────────────────────────────────────────────────────────

def test_cagr_positive(flat_positive):
    assert annualised_return(flat_positive) > 0

def test_cagr_negative(flat_negative):
    assert annualised_return(flat_negative) < 0


# ── Calmar ───────────────────────────────────────────────────────

def test_calmar_reasonable(random_returns):
    c = calmar_ratio(random_returns)
    assert -10 < c < 20


# ── Rolling Sharpe ───────────────────────────────────────────────

def test_rolling_sharpe_length(random_returns):
    rs = rolling_sharpe(random_returns, window=63)
    assert len(rs) == len(random_returns)

def test_rolling_sharpe_nan_prefix(random_returns):
    rs = rolling_sharpe(random_returns, window=63)
    assert np.isnan(rs[:63]).all()
    assert not np.isnan(rs[63:]).any()


# ── compute_all_metrics ──────────────────────────────────────────

def test_compute_all_metrics_keys(random_returns):
    m = compute_all_metrics(random_returns, label="test")
    required = ["sharpe","sortino","calmar","cagr","max_drawdown","var_95","cvar_95"]
    for k in required:
        assert k in m, f"Missing key: {k}"

def test_compute_all_metrics_empty():
    m = compute_all_metrics(np.array([]))
    assert m["sharpe"] == 0.0


# ── t-test ───────────────────────────────────────────────────────

def test_ttest_identical_series(random_returns):
    t, p = ttest_excess_returns(random_returns, random_returns)
    assert p > 0.05   # no difference -> not significant

def test_ttest_returns_floats(random_returns):
    t, p = ttest_excess_returns(random_returns, random_returns * 0.5)
    assert isinstance(t, float)
    assert isinstance(p, float)
    assert 0 <= p <= 1