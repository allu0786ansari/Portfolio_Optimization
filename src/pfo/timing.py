"""The anti-leakage timing contract.

Conventions used everywhere in ``pfo`` (see proposal v2, section 6.2):

* ``prices.loc[t]`` is the close of trading day ``t``.
* ``returns.loc[t]`` is the *simple* return from close ``t-1`` to close ``t``.
  It is therefore only knowable at (or after) the close of day ``t``.
* A **decision** taken at the close of day ``t`` may use rows ``<= t`` only.
* The trade is **executed** ``execution_lag`` trading days later, at the close
  of day ``t + execution_lag``. The new position earns returns from day
  ``t + execution_lag + 1`` onward.

With ``execution_lag = 0`` the trade happens at the same close the decision is
made (market-on-close). The default of ``1`` is the conservative choice used
for all reported results.

v1 violated this contract: its RL observation at ``t`` contained ``r[t]`` and
the agent was rewarded with ``r[t]`` (an execution lag of -1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_EXECUTION_LAG = 1


def first_return_index(decision_idx: int, execution_lag: int = DEFAULT_EXECUTION_LAG) -> int:
    """Index of the first return earned by a decision taken at ``decision_idx``."""
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0 (a negative lag is look-ahead)")
    return decision_idx + execution_lag + 1


def forward_returns(returns: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return earned over the next ``horizon`` days, aligned to the decision day.

    ``forward_returns(r, 1).loc[t] == r.loc[t+1]``. This is a *label*; it must
    never be used as a feature.
    """
    growth = (1.0 + returns).rolling(horizon).apply(np.prod, raw=True) - 1.0
    return growth.shift(-horizon)


# ---------------------------------------------------------------------------
# Leakage detectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryFinding:
    feature: str
    ic_forward: float  # pooled correlation with next-day return
    ic_same_day: float  # pooled correlation with same-day return (informational)
    flagged: bool


def _pooled_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    x = a.to_numpy(dtype=float).ravel()
    y = b.reindex_like(a).to_numpy(dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return float("nan")
    x, y = x[mask], y[mask]
    if x.std() < 1e-15 or y.std() < 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def leakage_canary_scan(
    features: Mapping[str, pd.DataFrame],
    returns: pd.DataFrame,
    threshold: float = 0.3,
) -> list[CanaryFinding]:
    """Flag features that are implausibly correlated with *future* returns.

    Real daily predictive signals have |IC| well below 0.1. A feature with a
    pooled correlation above ``threshold`` against next-day returns is almost
    certainly leaking the label (e.g. it was built from ``r[t+1]``).
    """
    fwd = forward_returns(returns, 1)
    out = []
    for name, f in features.items():
        ic_f = _pooled_corr(f, fwd)
        ic_s = _pooled_corr(f, returns)
        out.append(CanaryFinding(name, ic_f, ic_s, bool(np.isfinite(ic_f) and abs(ic_f) > threshold)))
    return out


def check_causality(
    transform: Callable[[pd.DataFrame], pd.DataFrame | Mapping[str, pd.DataFrame]],
    data: pd.DataFrame,
    cut_positions: list[int],
    seed: int = 0,
    atol: float = 1e-10,
) -> list[str]:
    """Future-shuffle test for any feature/decision transform.

    For each cut position ``c`` the rows after ``c`` are replaced by a random
    permutation of themselves scaled by noise. A causal transform must produce
    identical outputs for all rows ``<= c``. Returns a list of violation
    messages (empty list == causal).
    """
    rng = np.random.default_rng(seed)
    base = transform(data)
    base_map = base if isinstance(base, Mapping) else {"output": base}
    problems: list[str] = []
    for c in cut_positions:
        tampered = data.copy()
        tail = tampered.iloc[c + 1 :].to_numpy(copy=True)
        if len(tail) == 0:
            continue
        tail = tail[rng.permutation(len(tail))] * rng.uniform(0.5, 1.5, size=tail.shape)
        tampered.iloc[c + 1 :] = tail
        out = transform(tampered)
        out_map = out if isinstance(out, Mapping) else {"output": out}
        for key, ref in base_map.items():
            a = np.asarray(ref.iloc[: c + 1], dtype=float)
            b = np.asarray(out_map[key].iloc[: c + 1], dtype=float)
            same = np.isclose(a, b, atol=atol, equal_nan=True)
            if not same.all():
                bad_row = int(np.argwhere(~same)[0][0])
                problems.append(
                    f"{key}: row {bad_row} (<= cut {c}) changed when data after the cut was altered"
                )
    return problems
