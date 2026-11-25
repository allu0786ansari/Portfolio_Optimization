"""Unit tests for the Phase 0 core (engine, costs, metrics, env, strategies)."""

import numpy as np
import pandas as pd
import pytest

from pfo.backtest import metrics as M
from pfo.backtest.costs import LinearCostModel
from pfo.backtest.engine import simulate
from pfo.backtest.splits import walk_forward_splits
from pfo.backtest.walkforward import WalkForwardConfig, run_walk_forward
from pfo.rl.env import PortfolioEnvV2, max_weight_bound, project_action
from pfo.strategies.base import History
from pfo.strategies.classical import EqualWeight, MarkowitzV1, MinVariance


def _returns(n=300, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, k)), index=idx, columns=list("ABCD")[:k])


# engine -----------------------------------------------------------------------

def test_engine_buy_and_hold_matches_manual_compounding():
    r = _returns()
    tw = pd.DataFrame([[0.25] * 4], index=[r.index[0]], columns=r.columns)
    res = simulate(tw, r, LinearCostModel(0.0), execution_lag=0)
    # buy-and-hold: NAV = mean of each asset's cumulative growth
    nav = (1 + r.iloc[1:]).cumprod().mean(axis=1)
    np.testing.assert_allclose(res.equity.iloc[-1], nav.iloc[-1], rtol=1e-10)


def test_engine_charges_costs_on_drifted_weights():
    r = _returns()
    dates = [r.index[0], r.index[50]]
    tw = pd.DataFrame([[0.25] * 4, [0.25] * 4], index=dates, columns=r.columns)
    res = simulate(tw, r, LinearCostModel(10.0), execution_lag=0)
    assert res.turnover.iloc[0] == pytest.approx(1.0)  # from cash
    rebalance_turnover = res.turnover.loc[r.index[50]]
    assert 0.0 < rebalance_turnover < 0.5  # only the drift is traded back
    assert res.costs.loc[r.index[50]] == pytest.approx(rebalance_turnover * 10 / 1e4)


def test_engine_rejects_leverage_and_shorts():
    r = _returns()
    with pytest.raises(ValueError):
        simulate(pd.DataFrame([[0.6, 0.6, 0, 0]], index=[r.index[0]], columns=r.columns), r)
    with pytest.raises(ValueError):
        simulate(pd.DataFrame([[1.2, -0.2, 0, 0]], index=[r.index[0]], columns=r.columns), r)


def test_engine_keeps_cash_remainder():
    r = _returns()
    tw = pd.DataFrame([[0.25, 0.25, 0, 0]], index=[r.index[0]], columns=r.columns)
    res = simulate(tw, r, LinearCostModel(0.0), execution_lag=0)
    assert res.net_returns.abs().max() < r.abs().max().max()


# metrics -----------------------------------------------------------------------

def test_metrics_reject_duplicated_dates():
    r = _returns()["A"]
    dup = pd.concat([r.iloc[:50], r.iloc[25:75]])
    with pytest.raises(ValueError, match="duplicated"):
        M.sharpe(dup)


def test_metrics_basic_values():
    idx = pd.bdate_range("2020-01-01", periods=252)
    flat = pd.Series(0.001, index=idx)
    assert M.cagr(flat) == pytest.approx(1.001**252 - 1)
    assert M.max_drawdown(flat) == 0.0
    down = pd.Series([0.1, -0.5, 0.2], index=idx[:3])
    assert M.max_drawdown(down) == pytest.approx(-0.5)


def test_newey_west_detects_a_real_mean_and_not_noise():
    rng = np.random.default_rng(0)
    assert M.newey_west_tstat(rng.normal(0.5, 1, 200))[1] < 1e-4
    assert M.newey_west_tstat(rng.normal(0.0, 1, 200))[1] > 0.01


def test_paired_monthly_test_uses_calendar_months():
    r = _returns(n=504)
    out = M.paired_monthly_test(r["A"], r["B"])
    assert 22 <= out["n_months"] <= 25


def test_sharpe_ci_contains_point_estimate():
    r = _returns(n=756)["A"]
    lo, hi = M.sharpe_confidence_interval(r, n_boot=200)
    assert lo < M.sharpe(r) < hi


# splits -------------------------------------------------------------------------

def test_expanding_and_embargo():
    f = walk_forward_splits(1000, 300, 100, expanding=True, embargo=5)
    assert all(x.train_start == 0 for x in f)
    assert all(x.test_start - x.train_end == 5 for x in f)


# env ------------------------------------------------------------------------------

def test_project_action_and_bounds():
    w = project_action(np.array([1.0, -1.0, 0.0]), 5.0)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    # v1 scale (1.0) with 40 assets cannot exceed ~16% in one asset; v2 default can concentrate
    assert max_weight_bound(40, 1.0) == pytest.approx(0.1592, abs=1e-3)
    assert max_weight_bound(40, 5.0) > 0.9


def test_env_passes_sb3_checker_and_stays_in_window():
    from stable_baselines3.common.env_checker import check_env

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5, 3)).astype(np.float32)
    r = rng.normal(0, 0.01, size=(400, 5))
    env = PortfolioEnvV2(x, r, start=100, end=300, episode_length=60, seed=0)
    check_env(env, warn=True)
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            assert 100 <= info["reward_row"] < 300
            done = term or trunc


# strategies -------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [EqualWeight, MinVariance, MarkowitzV1])
def test_classical_strategies_return_valid_weights(cls):
    r = _returns(n=400).to_numpy()
    w = cls().decide(History(t=399, returns=r, features_now=None, prev_weights=np.full(4, 0.25),
                             date=pd.Timestamp("2021-07-01")))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()


def test_history_enforces_information_set():
    r = _returns(n=100).to_numpy()
    with pytest.raises(ValueError):
        History(t=50, returns=r, features_now=None, prev_weights=np.full(4, 0.25), date=pd.Timestamp("2020-03-01"))


def test_walk_forward_smoke(synth_market):
    res = run_walk_forward(synth_market, {"ew": EqualWeight, "minvar": MinVariance},
                           WalkForwardConfig(train_size=252, test_size=126))
    s = res.summaries()
    assert set(s) == {"ew", "minvar"}
    assert s["ew"]["n_days"] > 300
    assert not res.fold_metrics.empty


@pytest.mark.slow
def test_ppo_allocator_trains_and_decides(synth_market):
    from pfo.rl.strategy import PPOAllocator

    res = run_walk_forward(
        synth_market,
        {"rl": lambda: PPOAllocator(timesteps=1024, ppo_kwargs={"n_steps": 256, "batch_size": 64})},
        WalkForwardConfig(train_size=252, test_size=126),
    )
    d = res.results["rl"].decisions
    np.testing.assert_allclose(d.sum(axis=1), 1.0, atol=1e-6)
