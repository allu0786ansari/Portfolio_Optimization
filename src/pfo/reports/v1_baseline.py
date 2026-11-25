"""Corrected v1 baseline (Phase 0 deliverable).

Re-runs the v1 strategies (equal weight, the v1 Markowitz rule, a PPO
allocator) plus a minimum-variance benchmark under the v2 timing contract:

* one exchange calendar per universe (no NSE/NYSE intersection, no FX mixing);
* features recomputed causally and standardized per fold;
* non-overlapping walk-forward folds, RL re-trained inside every fold;
* identical execution lag and transaction costs for every strategy;
* statistics on unique dates with bootstrap / Newey-West inference.

Usage::

    pfo-v1-baseline --source v1 --universe both          # real data (after fetch_data)
    pfo-v1-baseline --source synthetic --rl-timesteps 5000  # smoke run, no downloads
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from pfo import __version__
from pfo.backtest import metrics as M
from pfo.backtest.walkforward import (
    StrategyFactory,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)
from pfo.data.market import MarketData, load_v1_universe, make_synthetic_market
from pfo.strategies.classical import EqualWeight, MarkowitzV1, MinVariance

# What v1 reported (backtesting/backtest_results.json, SYSTEM_DESIGN.md) — for comparison only.
V1_CLAIMS: dict[str, Any] = {
    "universe": "40 assets, NSE and NYSE dates intersected, no FX",
    "windows": "30 overlapping 126-day windows stepped by 21 days (735 unique days counted as 3,780)",
    "rl_ppo": {"sharpe": 1.466, "cagr": 0.1995, "max_drawdown": -0.3278, "p_value_vs_ew": 4.27e-22},
    "markowitz_v1": {"sharpe": 0.774, "cagr": 0.0869, "max_drawdown": -0.4233},
    "equal_weight": {"sharpe": 1.241, "cagr": 0.1687, "max_drawdown": -0.4153},
}


def _strategies(rl_timesteps: int, seed: int, cost_bps: float, lag: int) -> dict[str, StrategyFactory]:
    s: dict[str, StrategyFactory] = {"equal_weight": EqualWeight, "min_variance": MinVariance, "markowitz_v1": MarkowitzV1}
    if rl_timesteps > 0:
        from pfo.rl.strategy import PPOAllocator

        s["rl_ppo"] = lambda: PPOAllocator(timesteps=rl_timesteps, seed=seed, cost_bps=cost_bps,
                                           execution_lag=lag)
    return s


def evaluate(res: WalkForwardResult, n_boot: int) -> dict:
    out: dict = {"strategies": {}, "vs_equal_weight": {}}
    ew = res.results["equal_weight"].net_returns
    for name, br in res.results.items():
        s = M.summary(br.net_returns, br.turnover, br.costs)
        s["sharpe_ci95"] = list(M.sharpe_confidence_interval(br.net_returns, n_boot=n_boot))
        s["gross_sharpe"] = M.sharpe(br.gross_returns)
        out["strategies"][name] = s
        if name != "equal_weight":
            out["vs_equal_weight"][name] = {
                "monthly_excess_test": M.paired_monthly_test(br.net_returns, ew),
                "sharpe_difference": M.sharpe_difference_bootstrap(br.net_returns, ew, n_boot=n_boot),
                "folds_won": int(_folds_won(res, name)),
                "n_folds": len(res.folds),
            }
    return out


def _folds_won(res: WalkForwardResult, name: str) -> int:
    fm = res.fold_metrics.pivot(index="fold", columns="strategy", values="sharpe")
    return int((fm[name] > fm["equal_weight"]).sum()) if name in fm else 0


def _fmt(x: float, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def render_markdown(universe: str, market: MarketData, cfg: WalkForwardConfig, res: WalkForwardResult,
                    ev: dict, source: str) -> str:
    lines = [f"### Universe: `{universe}` ({market.currency}, {len(market.tickers)} assets)", ""]
    lines.append(f"- Out-of-sample: **{ev['oos_start']} → {ev['oos_end']}**, {ev['oos_days']} unique trading days, "
                 f"{len(res.folds)} non-overlapping folds (train {cfg.train_size}d, test {cfg.test_size}d)")
    if market.dropped:
        lines.append(f"- Excluded (history covers < 95% of the calendar): {', '.join(market.dropped)}")
    lines.append(f"- Rebalance every {cfg.rebalance_every}d · execution lag {cfg.execution_lag}d · "
                 f"costs {cfg.cost_bps} bps per unit turnover · data source `{source}`")
    lines += ["", "| Strategy | Sharpe (95% CI) | CAGR | Vol | Max DD | CVaR95 (daily) | Turnover/yr | Cost drag/yr |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in ev["strategies"].items():
        lo, hi = s["sharpe_ci95"]
        lines.append(f"| {name} | {_fmt(s['sharpe'])} ({_fmt(lo)}, {_fmt(hi)}) | {_fmt(s['cagr'], True)} | "
                     f"{_fmt(s['ann_vol'], True)} | {_fmt(s['max_drawdown'], True)} | {_fmt(s['cvar_95'], True)} | "
                     f"{_fmt(s.get('annual_turnover'))} | {_fmt(s.get('annual_cost_drag'), True)} |")
    lines += ["", "| vs equal weight | Δ Sharpe (95% CI) | p (bootstrap) | mean monthly excess | NW t | p (NW) | folds won |",
              "|---|---|---|---|---|---|---|"]
    for name, v in ev["vs_equal_weight"].items():
        sd, mt = v["sharpe_difference"], v["monthly_excess_test"]
        lines.append(f"| {name} | {_fmt(sd['sharpe_diff'])} ({_fmt(sd['ci95'][0])}, {_fmt(sd['ci95'][1])}) | "
                     f"{sd['p_value']:.3f} | {mt['mean_monthly_excess']:.2%} | {_fmt(mt['t_stat_nw'])} | "
                     f"{mt['p_value']:.3f} | {v['folds_won']}/{v['n_folds']} |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = WalkForwardConfig(train_size=args.train_days, test_size=args.test_days,
                            rebalance_every=args.rebalance, execution_lag=args.execution_lag,
                            cost_bps=args.cost_bps)
    universes = ["nifty", "sp500"] if args.universe == "both" else [args.universe]
    if args.source == "synthetic":
        universes = ["synthetic"]
    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pfo_version": __version__, "python": platform.python_version(), "source": args.source,
        "config": vars(args), "v1_claims": V1_CLAIMS, "universes": {},
    }
    md = ["# Phase 0 — Corrected v1 baseline", "",
          f"Generated {report['generated_utc']} · pfo {__version__} · source `{args.source}`", ""]
    if args.source == "synthetic":
        md += ["> **Synthetic smoke run.** Prices come from a random factor model with no exploitable signal. "
               "These numbers only prove the pipeline runs end-to-end; they say nothing about real markets.", ""]
    for u in universes:
        t0 = time.time()
        market = make_synthetic_market(n_assets=args.synthetic_assets, n_days=args.synthetic_days, seed=args.seed) \
            if u == "synthetic" else load_v1_universe(u)
        res = run_walk_forward(market, _strategies(args.rl_timesteps, args.seed, args.cost_bps, args.execution_lag), cfg)
        ev = evaluate(res, args.n_boot)
        any_r = next(iter(res.results.values())).net_returns
        ev.update({"oos_start": str(any_r.index[0].date()), "oos_end": str(any_r.index[-1].date()),
                   "oos_days": int(len(any_r)), "n_folds": len(res.folds), "tickers": market.tickers,
                   "dropped_tickers": list(market.dropped),
                   "runtime_sec": round(time.time() - t0, 1)})
        report["universes"][u] = ev
        udir = out_dir / u
        udir.mkdir(exist_ok=True)
        pd.DataFrame({k: v.equity for k, v in res.results.items()}).to_csv(udir / "equity.csv")
        res.fold_metrics.to_csv(udir / "fold_metrics.csv", index=False)
        md += [render_markdown(u, market, cfg, res, ev, args.source), ""]
        logger.info(f"{u}: done in {ev['runtime_sec']}s")

    md += ["## What v1 claimed (not comparable — produced with look-ahead and overlapping windows)", "",
           "| Strategy | v1 Sharpe | v1 CAGR | v1 Max DD |", "|---|---|---|---|"]
    for k in ("rl_ppo", "markowitz_v1", "equal_weight"):
        c = V1_CLAIMS[k]
        md.append(f"| {k} | {c['sharpe']:.2f} | {c['cagr']:.1%} | {c['max_drawdown']:.1%} |")
    md += ["", f"v1 universe: {V1_CLAIMS['universe']}. v1 windows: {V1_CLAIMS['windows']}. "
           f"v1 RL p-value vs EW: {V1_CLAIMS['rl_ppo']['p_value_vs_ew']:.1e}.", ""]
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info(f"wrote {out_dir / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["v1", "synthetic"], default="v1")
    p.add_argument("--universe", choices=["nifty", "sp500", "both"], default="both")
    p.add_argument("--train-days", type=int, default=504)
    p.add_argument("--test-days", type=int, default=126)
    p.add_argument("--rebalance", type=int, default=21)
    p.add_argument("--execution-lag", type=int, default=1)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rl-timesteps", type=int, default=30_000, help="0 disables the RL strategy")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic-assets", type=int, default=20)
    p.add_argument("--synthetic-days", type=int, default=2000)
    p.add_argument("--out", default="reports/phase0")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
