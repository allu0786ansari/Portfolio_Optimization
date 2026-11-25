"""Phase 0 exit-gate leakage suite (proposal v2, section 12.4).

Every test here must pass before any result is reported. The suite runs on
synthetic data in CI; ``pfo-leakage-scan --source v1`` runs the same checks on
downloaded market data.
"""

import numpy as np
import pandas as pd
import pytest

from pfo.backtest.engine import simulate
from pfo.backtest.metrics import sharpe
from pfo.backtest.splits import assert_non_overlapping, walk_forward_splits
from pfo.backtest.walkforward import WalkForwardConfig, run_walk_forward
from pfo.data.market import MarketData
from pfo.features.basic import FoldScaler, compute_features, to_tensor
from pfo.rl.env import PortfolioEnvV2, project_action
from pfo.strategies.classical import EqualWeight, MarkowitzV1, MinVariance
from pfo.timing import check_causality, forward_returns, leakage_canary_scan

pytestmark = pytest.mark.leakage


# 1. Future-shuffle test --------------------------------------------------------

def test_feature_pipeline_is_causal(synth_market):
    prices = synth_market.prices
    cuts = [250, 400, 600, len(prices) - 50]
    problems = check_causality(lambda p: compute_features(p), prices, cuts)
    assert problems == [], problems


def test_future_shuffle_detects_a_leaky_transform(synth_market):
    leaky = lambda p: p.pct_change().shift(-1)  # noqa: E731 - uses tomorrow's price
    assert check_causality(leaky, synth_market.prices, [300]) != []


def test_walk_forward_decisions_ignore_future_data(synth_market):
    cfg = WalkForwardConfig(train_size=252, test_size=126, rebalance_every=21)
    strategies = {"ew": EqualWeight, "minvar": MinVariance, "mvo": MarkowitzV1}
    base = run_walk_forward(synth_market, strategies, cfg)
    cut_date = base.dates[base.folds[1].test_start + 30]

    tampered = synth_market.prices.copy()
    after = tampered.index > cut_date
    rng = np.random.default_rng(1)
    tampered.loc[after] = tampered.loc[after].to_numpy() * rng.uniform(0.5, 1.5, size=tampered.loc[after].shape)
    market2 = MarketData("tampered", tampered, synth_market.benchmark, "USD")
    alt = run_walk_forward(market2, strategies, cfg)

    for name in strategies:
        d1 = base.results[name].decisions
        d2 = alt.results[name].decisions
        upto = d1.index <= cut_date
        assert upto.sum() > 3
        pd.testing.assert_frame_equal(d1.loc[upto], d2.loc[upto], check_exact=False, atol=1e-8)


# 2. Oracle canary ---------------------------------------------------------------

def test_oracle_canary_is_flagged_and_real_features_are_not(synth_market):
    feats = compute_features(synth_market.prices, synth_market.benchmark)
    feats["ORACLE_tomorrow_return"] = forward_returns(synth_market.returns, 1)
    findings = {f.feature: f for f in leakage_canary_scan(feats, synth_market.returns)}
    assert findings["ORACLE_tomorrow_return"].flagged
    assert not any(f.flagged for k, f in findings.items() if k != "ORACLE_tomorrow_return")


# 3. Fold isolation -------------------------------------------------------------

def test_fold_scaler_uses_training_rows_only(synth_market):
    x = to_tensor(compute_features(synth_market.prices))
    rows = range(250, 500)
    a = FoldScaler().fit(x, rows)
    x2 = x.copy()
    x2[500:] = x2[500:] * 100.0 + 7.0  # alter everything after the training rows
    b = FoldScaler().fit(x2[:500], rows)
    np.testing.assert_array_equal(a.mean_, b.mean_)
    np.testing.assert_array_equal(a.std_, b.std_)


# 4. Non-overlapping out-of-sample windows (v1 defect #2) -----------------------

def test_default_splits_never_overlap():
    folds = walk_forward_splits(2000, 504, 126)
    assert_non_overlapping(folds)
    assert all(b.test_start == a.test_end for a, b in zip(folds, folds[1:], strict=False))


def test_v1_style_overlapping_splits_are_refused():
    with pytest.raises(ValueError, match="overlapping"):
        walk_forward_splits(2000, 504, 126, step=21)


def test_stitched_oos_dates_are_unique_and_monotonic(synth_market):
    res = run_walk_forward(synth_market, {"ew": EqualWeight}, WalkForwardConfig(train_size=252, test_size=126))
    idx = res.results["ew"].net_returns.index
    assert not idx.has_duplicates and idx.is_monotonic_increasing


# 5. Execution timing (v1 defect #1) ---------------------------------------------

def test_decision_does_not_earn_the_next_days_return():
    dates = pd.bdate_range("2024-01-01", periods=10)
    r = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    r.iloc[4, 0] = 0.50  # a jump the day after the decision
    tw = pd.DataFrame([[1.0, 0.0]], index=[dates[3]], columns=["A", "B"])
    res = simulate(tw, r, execution_lag=1)
    assert res.net_returns.max() < 0.01  # trade executes at close of day 4: the jump is missed
    res0 = simulate(tw, r, execution_lag=0)
    assert res0.net_returns.max() > 0.49  # market-on-close at day 3 captures it


def test_env_observation_is_independent_of_future_returns(synth_market):
    x = to_tensor(compute_features(synth_market.prices))[200:]
    r = synth_market.returns.to_numpy()[200:]
    x = np.nan_to_num(x)
    env = PortfolioEnvV2(x, r, start=0, end=600, episode_length=50, seed=0)
    obs, _ = env.reset(options={"start_row": 100})
    r2 = r.copy()
    r2[101:] *= -3.0
    env2 = PortfolioEnvV2(x, r2, start=0, end=600, episode_length=50, seed=0)
    obs2, _ = env2.reset(options={"start_row": 100})
    np.testing.assert_array_equal(obs, obs2)
    a = np.zeros(env.n_assets, dtype=np.float32)
    for _ in range(5):
        o1, _, _, _, i1 = env.step(a)
        o2, _, _, _, i2 = env2.step(a)
        np.testing.assert_array_equal(o1, o2)  # observations never see future returns
        assert i1["reward_row"] == env._t - 1 + env.lag + 1


def test_v1_timing_bug_is_exploitable_and_v2_timing_is_not(synth_market):
    """Regression test for the v1 RL result.

    A policy that tilts toward assets with a high *same-day* return (which v1
    put in the observation) earns an absurd Sharpe when rewarded with the same
    day's return (v1), and nothing special under the corrected timing (v2).
    """
    r = synth_market.returns.to_numpy()[1:]
    observed = r  # feature log_return[t] ~ r[t], visible at decision t
    w = np.array([project_action(np.sign(o), 5.0) for o in observed])
    v1_pnl = (w[:-2] * r[:-2]).sum(axis=1)  # v1: reward uses r[t]
    v2_pnl = (w[:-2] * r[2:]).sum(axis=1)  # v2: reward uses r[t + lag + 1], lag = 1
    assert sharpe(v1_pnl) > 8.0
    assert abs(sharpe(v2_pnl)) < 2.0
