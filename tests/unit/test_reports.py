"""The Phase 0 CLIs run end-to-end (synthetic data and the v1 on-disk layout)."""

import json

import pandas as pd
import pytest

from pfo.data.market import make_synthetic_market
from pfo.data.universes import NIFTY_V1, SP500_V1
from pfo.reports import leakage_scan, v1_baseline


@pytest.fixture()
def v1_files(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for seed, (tickers, bench) in enumerate([(SP500_V1, "sp500"), (NIFTY_V1, "nifty50")]):
        m = make_synthetic_market(n_assets=len(tickers), n_days=1300, seed=20 + seed)
        for syn, tic in zip(m.tickers, tickers, strict=True):
            pd.DataFrame({"Close": m.prices[syn]}).rename_axis("date").to_parquet(raw / f"{tic}.parquet")
        pd.DataFrame({"Close": m.benchmark}).rename_axis("date").to_parquet(raw / f"benchmark_{bench}.parquet")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_leakage_scan_cli_passes_on_synthetic(tmp_path):
    out = tmp_path / "scan.json"
    assert leakage_scan.main(["--source", "synthetic", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert res["passed"] and set(res["checks"]) >= {"feature_causality", "oracle_canary", "decisions_ignore_future"}


def test_leakage_scan_cli_on_v1_layout(v1_files):
    assert leakage_scan.main(["--source", "v1", "--universe", "both", "--out", "scan.json"]) == 0


def test_baseline_cli_writes_report_on_v1_layout(v1_files):
    rc = v1_baseline.main(["--source", "v1", "--universe", "both", "--rl-timesteps", "0", "--n-boot", "50",
                           "--train-days", "504", "--test-days", "126", "--out", "out"])
    assert rc == 0
    summary = json.loads((v1_files / "out" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["universes"]) == {"nifty", "sp500"}
    for u in summary["universes"].values():
        assert set(u["strategies"]) == {"equal_weight", "min_variance", "markowitz_v1"}
        assert u["n_folds"] >= 2
    md = (v1_files / "out" / "report.md").read_text(encoding="utf-8")
    assert "What v1 claimed" in md and "nifty" in md and "sp500" in md
    eq = pd.read_csv(v1_files / "out" / "sp500" / "equity.csv", index_col=0, parse_dates=True)
    assert eq.index.is_monotonic_increasing and not eq.index.has_duplicates


@pytest.mark.slow
def test_baseline_cli_synthetic_with_rl(tmp_path):
    rc = v1_baseline.main(["--source", "synthetic", "--rl-timesteps", "512", "--n-boot", "50",
                           "--synthetic-assets", "6", "--synthetic-days", "1000", "--out", str(tmp_path)])
    assert rc == 0
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["universes"]["synthetic"]
    assert "rl_ppo" in s["strategies"] and "rl_ppo" in s["vs_equal_weight"]
