"""Run the Phase 0 leakage checks on real (or synthetic) data.

Exit code 0 only if every check passes — this is the Phase 0 exit gate
"leakage suite passes on v1 data".

    pfo-leakage-scan --source v1 --universe both
    pfo-leakage-scan --source synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pfo.backtest.splits import assert_non_overlapping, walk_forward_splits
from pfo.backtest.walkforward import StrategyFactory, WalkForwardConfig, run_walk_forward
from pfo.data.market import MarketData, load_v1_universe, make_synthetic_market
from pfo.features.basic import FoldScaler, compute_features, to_tensor
from pfo.strategies.classical import EqualWeight, MinVariance
from pfo.timing import check_causality, forward_returns, leakage_canary_scan


def scan(market: MarketData) -> dict:
    checks: dict[str, dict] = {}
    prices = market.prices
    n = len(prices)

    cuts = sorted({int(n * q) for q in (0.3, 0.5, 0.7, 0.9)})
    problems = check_causality(lambda p: compute_features(p), prices, cuts)
    checks["feature_causality"] = {"passed": not problems, "violations": problems[:10]}

    leaky = check_causality(lambda p: p.pct_change(fill_method=None).shift(-1), prices, cuts[:1])
    checks["causality_positive_control"] = {"passed": bool(leaky), "note": "a deliberately leaky transform must be caught"}

    feats = compute_features(prices, market.benchmark)
    feats["CANARY_tomorrow_return"] = forward_returns(market.returns, 1)
    findings = leakage_canary_scan(feats, market.returns)
    table = [{"feature": f.feature, "ic_forward": round(f.ic_forward, 4), "ic_same_day": round(f.ic_same_day, 4),
              "flagged": f.flagged} for f in findings]
    canary_ok = any(f.flagged for f in findings if f.feature.startswith("CANARY"))
    real_flagged = [f.feature for f in findings if f.flagged and not f.feature.startswith("CANARY")]
    checks["oracle_canary"] = {"passed": canary_ok and not real_flagged, "flagged_real_features": real_flagged,
                               "table": table}

    x = to_tensor(compute_features(prices))
    split = n // 2
    a = FoldScaler().fit(x, range(200, split))
    b = FoldScaler().fit(x[:split], range(200, split))
    checks["fold_isolation"] = {"passed": bool(np.array_equal(a.mean_, b.mean_) and np.array_equal(a.std_, b.std_))}

    try:
        folds = walk_forward_splits(n - 200, 504, 126)
        assert_non_overlapping(folds)
        checks["non_overlapping_folds"] = {"passed": True, "n_folds": len(folds)}
    except (AssertionError, ValueError) as exc:
        checks["non_overlapping_folds"] = {"passed": False, "error": str(exc)}

    cfg = WalkForwardConfig(train_size=504 if n > 1100 else 252, test_size=126)
    strategies: dict[str, StrategyFactory] = {"ew": EqualWeight, "minvar": MinVariance}
    try:
        base = run_walk_forward(market, strategies, cfg)
        cut = base.dates[base.folds[min(1, len(base.folds) - 1)].test_start + 10]
        tampered = prices.copy()
        after = tampered.index > cut
        tampered.loc[after] = tampered.loc[after].to_numpy() * np.random.default_rng(3).uniform(
            0.7, 1.3, size=tampered.loc[after].shape)
        alt = run_walk_forward(MarketData("tampered", tampered, market.benchmark, market.currency), strategies, cfg)
        same = all(
            np.allclose(base.results[k].decisions.loc[:cut].to_numpy(), alt.results[k].decisions.loc[:cut].to_numpy())
            for k in strategies
        )
        dup_free = all(not v.net_returns.index.has_duplicates for v in base.results.values())
        checks["decisions_ignore_future"] = {"passed": bool(same and dup_free), "cut_date": str(cut.date())}
    except ValueError as exc:
        checks["decisions_ignore_future"] = {"passed": False, "error": str(exc)}

    return {"universe": market.name, "n_days": n, "n_assets": len(market.tickers),
            "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
            "passed": all(c["passed"] for c in checks.values()), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["v1", "synthetic"], default="v1")
    p.add_argument("--universe", choices=["nifty", "sp500", "both"], default="both")
    p.add_argument("--out", default="reports/phase0/leakage_scan.json")
    a = p.parse_args(argv)
    if a.source == "synthetic":
        markets = [make_synthetic_market(n_assets=15, n_days=1500, seed=11)]
    else:
        markets = [load_v1_universe(u) for u in (["nifty", "sp500"] if a.universe == "both" else [a.universe])]
    results = [scan(m) for m in markets]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results}, indent=2, default=float), encoding="utf-8")
    rows = []
    for r in results:
        for name, c in r["checks"].items():
            rows.append({"universe": r["universe"], "check": name, "passed": c["passed"]})
    print(pd.DataFrame(rows).to_string(index=False))
    ok = all(r["passed"] for r in results)
    (logger.success if ok else logger.error)(f"leakage scan {'PASSED' if ok else 'FAILED'} -> {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
