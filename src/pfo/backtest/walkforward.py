"""Walk-forward evaluation: fit per fold, decide on the test window, stitch, simulate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from pfo.backtest.costs import LinearCostModel
from pfo.backtest.engine import BacktestResult, simulate
from pfo.backtest.metrics import summary
from pfo.backtest.splits import Fold, assert_non_overlapping, walk_forward_splits
from pfo.data.market import MarketData
from pfo.features.basic import WARMUP_DAYS, compute_features, to_tensor
from pfo.strategies.base import History, Strategy, TrainData
from pfo.timing import DEFAULT_EXECUTION_LAG

StrategyFactory = Callable[[], Strategy]


@dataclass
class WalkForwardConfig:
    train_size: int = 504
    test_size: int = 126
    rebalance_every: int = 21
    execution_lag: int = DEFAULT_EXECUTION_LAG
    cost_bps: float = 10.0
    expanding: bool = False
    warmup: int = WARMUP_DAYS


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: list[Fold]
    results: dict[str, BacktestResult]
    fold_metrics: pd.DataFrame
    dates: pd.DatetimeIndex
    extra: dict = field(default_factory=dict)

    def summaries(self) -> dict[str, dict]:
        return {k: summary(v.net_returns, v.turnover, v.costs) for k, v in self.results.items()}


def run_walk_forward(
    market: MarketData,
    strategies: Mapping[str, StrategyFactory],
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    cfg = config or WalkForwardConfig()
    feats = compute_features(market.prices, market.benchmark)
    # Drop the warm-up rows where the longest look-back is undefined.
    prices = market.prices.iloc[cfg.warmup :]
    returns_df = market.prices.pct_change(fill_method=None).iloc[cfg.warmup :]
    x = to_tensor({k: v.iloc[cfg.warmup :] for k, v in feats.items()})
    r = returns_df.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(prices.index)

    folds = walk_forward_splits(len(dates), cfg.train_size, cfg.test_size, expanding=cfg.expanding)
    assert_non_overlapping(folds)
    if not folds:
        raise ValueError(f"not enough data: {len(dates)} rows < train {cfg.train_size} + test {cfg.test_size}")
    logger.info(f"{market.name}: {len(folds)} non-overlapping folds, "
                f"OOS {dates[folds[0].test_start].date()} -> {dates[folds[-1].test_end - 1].date()}")

    decisions: dict[str, dict[pd.Timestamp, np.ndarray]] = {k: {} for k in strategies}
    for fold in folds:
        tr = slice(fold.train_start, fold.train_end)
        train = TrainData(returns=r[tr], features=x[tr], dates=dates[tr], tickers=tuple(prices.columns))
        for name, factory in strategies.items():
            strat = factory()
            strat.fit(train)
            prev = np.full(r.shape[1], 1.0 / r.shape[1])
            for t in range(fold.test_start, fold.test_end, cfg.rebalance_every):
                hist = History(t=t, returns=r[: t + 1], features_now=x[t], prev_weights=prev, date=dates[t])
                w = np.asarray(strat.decide(hist), dtype=float)
                decisions[name][dates[t]] = w
                prev = w

    oos_end = dates[folds[-1].test_end - 1]
    cost = LinearCostModel(cfg.cost_bps)
    results: dict[str, BacktestResult] = {}
    for name, dec in decisions.items():
        tw = pd.DataFrame.from_dict(dec, orient="index", columns=prices.columns).sort_index()
        results[name] = simulate(tw, returns_df, cost, execution_lag=cfg.execution_lag, end=oos_end)

    rows = []
    for fold in folds:
        lo, hi = dates[fold.test_start], dates[fold.test_end - 1]
        for name, res in results.items():
            seg = res.net_returns.loc[lo:hi]
            if len(seg) > 5:
                s = summary(seg)
                rows.append({"fold": fold.fold_id, "strategy": name, "start": str(lo.date()),
                             "end": str(hi.date()), "sharpe": s["sharpe"], "return": float((1 + seg).prod() - 1),
                             "max_drawdown": s["max_drawdown"]})
    return WalkForwardResult(cfg, folds, results, pd.DataFrame(rows), dates)
