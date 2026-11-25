"""Fresh-download pipeline (network replaced by a fake fetcher)."""

import json

import numpy as np
import pandas as pd

from pfo.data import download as D
from pfo.data.market import load_v1_universe
from pfo.data.universes import UNIVERSES


def fake_fetch(ticker, start, end):
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    idx = pd.bdate_range(start, periods=700, tz="America/New_York")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx)


def test_fresh_download_replaces_old_files_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    (tmp_path / "STALE.parquet").write_bytes(b"old v1 file")
    m = D.download(["sp500"], tmp_path, start="2020-01-01", fetch=fake_fetch, pause=0)
    assert not (tmp_path / "STALE.parquet").exists()
    n = len(UNIVERSES["sp500"]["tickers"]) + 1
    assert m["n_files"] == n and not m["failures"]
    manifest = json.loads((tmp_path / "_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"]["AAPL.parquet"]
    assert entry["rows"] == 700 and len(entry["sha256"]) == 64 and entry["problems"] == []
    df = pd.read_parquet(tmp_path / "AAPL.parquet")
    assert df.index.tz is None and list(df.columns) == D.COLUMNS


def test_downloaded_files_load_into_the_backtest_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    D.download(["nifty"], tmp_path / "raw", start="2020-01-01", fetch=fake_fetch, pause=0)
    m = load_v1_universe("nifty", raw_dir=tmp_path / "raw", processed_dir=tmp_path / "none")
    assert m.prices.shape == (700, 20) and m.benchmark is not None


def test_retries_then_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(ticker, start, end):
        if ticker == "AAPL":
            raise ConnectionError("rate limited")
        calls["n"] += 1
        return fake_fetch(ticker, start, end)

    m = D.download(["sp500"], tmp_path, fetch=flaky, pause=0)
    assert "AAPL" in m["failures"] and "AAPL.parquet" not in m["files"]


def test_validation_flags_bad_ticks():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(100.0, index=idx)
    close.iloc[150] = 300.0  # +200% bad tick
    df = pd.DataFrame({c: close for c in D.COLUMNS})
    assert any("daily move" in p for p in D.validate(df))
    assert any("rows" in p for p in D.validate(df.iloc[:100]))


def test_cli_exit_code_fails_when_a_benchmark_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)

    def no_index(ticker, start, end):
        return pd.DataFrame() if ticker.startswith("^") else fake_fetch(ticker, start, end)

    monkeypatch.setattr(D, "yfinance_fetcher", no_index)
    assert D.main(["--universe", "sp500", "--out", str(tmp_path)]) == 1
    manifest = json.loads((tmp_path / "_manifest.json").read_text(encoding="utf-8"))
    assert "^GSPC" in manifest["failures"]


def test_cli_succeeds_with_complete_data(tmp_path, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    monkeypatch.setattr(D, "yfinance_fetcher", fake_fetch)
    assert D.main(["--universe", "both", "--out", str(tmp_path)]) == 0
