"""A small, exact daily portfolio simulator that obeys the timing contract.

Every strategy (equal-weight, Markowitz, RL, and later quantum/router outputs)
goes through this one function, with the same cost model and execution lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfo.backtest.costs import LinearCostModel
from pfo.timing import DEFAULT_EXECUTION_LAG


@dataclass
class BacktestResult:
    net_returns: pd.Series  # daily simple returns after costs, monotonic dates
    gross_returns: pd.Series
    turnover: pd.Series  # one-way turnover executed on each day (0 if no trade)
    costs: pd.Series  # cost charged on each day, as a fraction of NAV
    weights: pd.DataFrame  # post-trade weights on execution days
    decisions: pd.DataFrame  # target weights indexed by decision date

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_returns).cumprod()


def simulate(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_model: LinearCostModel | None = None,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Simulate a long-only portfolio.

    Args:
        target_weights: rows = decision dates (must be in ``returns.index``),
            columns = assets. Row sums may be < 1 (the rest is cash at 0%).
        returns: daily simple returns, rows = trading days, columns = assets.
        execution_lag: trading days between decision and execution (>= 0).
        end: last day to simulate (inclusive). Defaults to the last return date.

    The output starts on the first day that a position earns a return, i.e.
    ``first decision + execution_lag + 1``. Weights drift with prices between
    trades and turnover is measured against the drifted weights.
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    cost_model = cost_model or LinearCostModel()
    if target_weights.empty:
        raise ValueError("no decisions to simulate")
    target_weights = target_weights.reindex(columns=returns.columns).fillna(0.0).sort_index()
    if (target_weights.to_numpy() < -1e-12).any():
        raise ValueError("long-only simulator received negative weights")
    sums = target_weights.sum(axis=1)
    if (sums > 1.0 + 1e-6).any():
        raise ValueError("target weights sum to more than 1 (leverage is not supported)")

    idx = returns.index
    pos = idx.get_indexer(target_weights.index)
    if (pos < 0).any():
        missing = target_weights.index[pos < 0][:3].tolist()
        raise KeyError(f"decision dates not in returns index: {missing}")

    exec_pos = pos + execution_lag
    schedule: dict[int, np.ndarray] = {}
    for p, w in zip(exec_pos, target_weights.to_numpy(), strict=True):
        if p < len(idx):
            schedule[int(p)] = w  # a later decision for the same execution day wins

    if not schedule:
        raise ValueError("all executions fall after the end of the return series")
    first_exec = min(schedule)
    last = len(idx) - 1 if end is None else int(idx.get_indexer([end])[0])
    r = returns.to_numpy(dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    n_assets = r.shape[1]

    w = np.zeros(n_assets)
    out_days, gross, net, tov, cst = [], [], [], [], []
    post_trade = {}
    # Day ``first_exec``: the trade happens at its close; returns start the next day.
    for s in range(first_exec, last + 1):
        if s > first_exec:
            day_r = float(w @ r[s])
            g = day_r
            # drift weights to the close of day s
            nav_growth = 1.0 + day_r
            w = w * (1.0 + r[s]) / nav_growth if nav_growth > 0 else w * 0.0
        else:
            g = 0.0
        t_o, c = 0.0, 0.0
        if s in schedule:
            target = schedule[s]
            t_o, c = cost_model.cost(w, target)
            w = target.copy()
            post_trade[idx[s]] = w.copy()
        if s > first_exec or c > 0:
            out_days.append(idx[s])
            gross.append(g)
            net.append((1.0 + g) * (1.0 - c) - 1.0)
            tov.append(t_o)
            cst.append(c)

    dates = pd.DatetimeIndex(out_days)
    return BacktestResult(
        net_returns=pd.Series(net, index=dates, name="net"),
        gross_returns=pd.Series(gross, index=dates, name="gross"),
        turnover=pd.Series(tov, index=dates, name="turnover"),
        costs=pd.Series(cst, index=dates, name="cost"),
        weights=pd.DataFrame.from_dict(post_trade, orient="index", columns=returns.columns),
        decisions=target_weights,
    )
