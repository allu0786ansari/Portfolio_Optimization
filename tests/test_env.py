"""Unit tests for Week 3 — PortfolioEnv Gym environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from models.rl_agent.portfolio_env import PortfolioEnv, softmax
from models.rl_agent.reward import step_reward, sortino_reward


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    """Shared env instance — loading data once is slow, reuse across tests."""
    e = PortfolioEnv(seed=0)
    yield e
    e.close()


# ── Softmax tests ────────────────────────────────────────────────

def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, -1.0, 0.5])
    w = softmax(x)
    assert abs(w.sum() - 1.0) < 1e-6


def test_softmax_all_positive():
    x = np.random.randn(20)
    w = softmax(x)
    assert (w > 0).all()


def test_softmax_numerical_stability():
    # Large values should not produce inf or nan
    x = np.array([1000.0, -1000.0, 500.0])
    w = softmax(x)
    assert not np.isnan(w).any()
    assert not np.isinf(w).any()


# ── Space tests ──────────────────────────────────────────────────

def test_observation_space_shape(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_action_space_shape(env):
    assert env.action_space.shape == (env.n_assets,)


def test_observation_no_nan(env):
    obs, _ = env.reset()
    assert not np.isnan(obs).any(), "Observation contains NaN after reset"


def test_observation_no_inf(env):
    obs, _ = env.reset()
    assert not np.isinf(obs).any(), "Observation contains Inf after reset"


# ── Step tests ───────────────────────────────────────────────────

def test_step_returns_correct_types(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs,        np.ndarray)
    assert isinstance(reward,     float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated,  bool)
    assert isinstance(info,       dict)


def test_weights_sum_to_one(env):
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert abs(info["weights"].sum() - 1.0) < 1e-5,             f"Weights sum to {info['weights'].sum()}, expected 1.0"


def test_weights_all_non_negative(env):
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert (info["weights"] >= 0).all()


def test_episode_terminates_at_252_steps(env):
    env.reset()
    steps = 0
    while True:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert steps == 252, f"Episode length should be 252, got {steps}"


def test_portfolio_value_positive(env):
    env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["portfolio_value"] > 0


def test_reward_finite(env):
    env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward), f"Reward is not finite: {reward}"


# ── Reset tests ──────────────────────────────────────────────────

def test_reset_resets_step_counter(env):
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    env.reset()
    assert env._current_step == 0


def test_reset_resets_weights_to_equal(env):
    env.reset()
    expected = np.ones(env.n_assets) / env.n_assets
    np.testing.assert_allclose(env._weights, expected, atol=1e-6)


def test_multiple_resets_stable(env):
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        assert not np.isnan(obs).any()
        assert "start_date" in info


# ── Reward function tests ────────────────────────────────────────

def test_step_reward_no_transaction_cost():
    n = 5
    w = np.ones(n) / n   # equal weights — same before and after
    r, info = step_reward(0.01, w, w)
    # No turnover means no transaction cost
    assert info["tc_penalty"] == 0.0
    assert abs(r - 0.01) < 1e-8


def test_step_reward_penalises_turnover():
    n = 4
    w_prev = np.array([0.25, 0.25, 0.25, 0.25])
    w_curr = np.array([1.00, 0.00, 0.00, 0.00])
    r_with_cost, info = step_reward(0.01, w_prev, w_curr)
    r_no_cost = 0.01
    assert r_with_cost < r_no_cost
    assert info["tc_penalty"] > 0


def test_sortino_positive_returns():
    returns = np.abs(np.random.randn(252)) * 0.01  # all positive returns
    n = 5
    w = np.ones(n) / n
    reward, info = sortino_reward(returns, w, w)
    # No downside -> high Sortino -> positive reward
    assert reward > 0


def test_sortino_all_negative():
    returns = -np.abs(np.random.randn(252)) * 0.01  # all negative
    n = 5
    w = np.ones(n) / n
    reward, info = sortino_reward(returns, w, w)
    assert reward < 0