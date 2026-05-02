"""Unit tests for Week 4 — RL training utilities and baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from models.rl_agent.evaluate_agent import compute_metrics
from models.classical.markowitz import markowitz_weights, ledoit_wolf_cov, MarkowitzOptimiser


# ── compute_metrics tests ────────────────────────────────────────

def test_metrics_perfect_positive():
    returns = [0.001] * 252
    m = compute_metrics(returns)
    assert m["sharpe"]  > 0
    assert m["cagr"]    > 0
    assert m["max_dd"]  == 0.0   # no drawdown on flat positive returns


def test_metrics_all_negative():
    returns = [-0.001] * 252
    m = compute_metrics(returns)
    assert m["sharpe"]  < 0
    assert m["cagr"]    < 0
    assert m["max_dd"]  < 0


def test_metrics_empty():
    m = compute_metrics([])
    assert m["sharpe"]  == 0.0
    assert m["sortino"] == 0.0


def test_metrics_sharpe_scale():
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252).tolist()
    m = compute_metrics(returns)
    # Annualised Sharpe should be in a reasonable range
    assert -5 < m["sharpe"] < 10


def test_metrics_max_drawdown_negative_or_zero():
    returns = np.random.randn(252).tolist()
    m = compute_metrics(returns)
    assert m["max_dd"] <= 0.0


# ── Markowitz tests ───────────────────────────────────────────────

@pytest.fixture
def synthetic_returns():
    np.random.seed(7)
    T, N = 300, 10
    return np.random.randn(T, N) * 0.01


def test_ledoit_wolf_symmetric(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)


def test_ledoit_wolf_positive_definite(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues > 0).all(), "Covariance must be positive definite"


def test_markowitz_weights_sum_to_one(synthetic_returns):
    expected_ret = synthetic_returns.mean(axis=0)
    cov          = ledoit_wolf_cov(synthetic_returns)
    w = markowitz_weights(expected_ret, cov)
    assert abs(w.sum() - 1.0) < 1e-4


def test_markowitz_weights_non_negative(synthetic_returns):
    expected_ret = synthetic_returns.mean(axis=0)
    cov          = ledoit_wolf_cov(synthetic_returns)
    w = markowitz_weights(expected_ret, cov)
    assert (w >= -1e-6).all()


def test_markowitz_weights_max_concentration(synthetic_returns):
    expected_ret = synthetic_returns.mean(axis=0)
    cov          = ledoit_wolf_cov(synthetic_returns)
    max_w        = 0.40
    w = markowitz_weights(expected_ret, cov, max_weight=max_w)
    assert (w <= max_w + 1e-4).all()


def test_markowitz_optimiser_simulate(synthetic_returns):
    opt = MarkowitzOptimiser(lookback_days=100)
    port_returns, weights = opt.simulate(synthetic_returns, start_idx=100, end_idx=250)
    assert len(port_returns) == 150
    assert weights.shape == (150, 10)
    # Each row of weights should sum to ~1
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-4)


def test_markowitz_fallback_to_equal_weight():
    # Too few observations should fall back to equal weight
    tiny_returns = np.random.randn(5, 4) * 0.01
    opt = MarkowitzOptimiser(lookback_days=252)
    w = opt.compute_weights(tiny_returns)
    np.testing.assert_allclose(w, np.ones(4) / 4, atol=1e-6)