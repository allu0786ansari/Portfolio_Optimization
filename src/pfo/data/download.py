"""Fresh market-data download (replaces v1 ``data/ingestion/fetch_data.py``).

v1 skipped any ticker whose file already existed ("Cache hit: loading from
disk"), so the data never refreshed, and its END_DATE was hard-coded. This
downloader:

* deletes the previous price files and re-downloads everything by default
  (``--resume`` keeps files that are already present and valid);
* downloads up to *today* from ``--start`` (default 2010-01-01);
* validates every series (enough rows, positive prices, no impossible jumps);
* writes ``data/raw/_manifest.json`` (time of download, library version,
  rows / first / last date / SHA-256 per file) so every result can be traced
  to an exact data snapshot.

File layout (unchanged, read by ``pfo.data.market.load_v1_universe``):
``data/raw/<TICKER>.parquet`` and ``data/raw/benchmark_<name>.parquet`` with
columns Open, High, Low, Close, Volume (dividend/split-adjusted).

Usage::

    pfo-download                      # fresh download of both universes + benchmarks
    pfo-download --universe sp500 --start 2015-01-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pfo.data.universes import UNIVERSES

Fetcher = Callable[[str, str, str], pd.DataFrame]
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MIN_ROWS = 252
MAX_ABS_DAILY_RETURN = 0.6  # larger moves are almost always bad ticks / unadjusted splits


def yfinance_fetcher(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily OHLCV for one ticker from Yahoo Finance."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True, actions=False)
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    missing = [c for c in COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df = df[COLUMNS].copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty == valid)."""
    problems = []
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (< {MIN_ROWS})")
    if (df["Close"] <= 0).any():
        problems.append("non-positive close prices")
    r = df["Close"].pct_change(fill_method=None).abs()
    if (r > MAX_ABS_DAILY_RETURN).any():
        worst = r.idxmax()
        problems.append(f"daily move {r.max():.0%} on {worst.date()} (bad tick or unadjusted split?)")
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_with_retries(fetch: Fetcher, ticker: str, start: str, end: str, retries: int = 4,
                       backoff: float = 2.0) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = _normalise(fetch(ticker, start, end))
            if not df.empty:
                return df
            last = RuntimeError("empty response")
        except Exception as exc:  # network errors, rate limits
            last = exc
        wait = backoff ** attempt
        logger.warning(f"{ticker}: attempt {attempt}/{retries} failed ({last}); retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"{ticker}: download failed after {retries} attempts: {last}")


def download(
    universes: list[str],
    out_dir: Path = Path("data/raw"),
    start: str = "2010-01-01",
    end: str | None = None,
    fresh: bool = True,
    fetch: Fetcher | None = None,
    pause: float = 0.3,
) -> dict:
    """Download universes + benchmarks into ``out_dir`` and write the manifest."""
    fetch = fetch or yfinance_fetcher
    end = end or (date.today() + pd.Timedelta(days=1)).isoformat()  # yfinance's end is exclusive
    out_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        old = list(out_dir.glob("*.parquet")) + list(out_dir.glob("_manifest.json"))
        for f in old:
            f.unlink()
        logger.info(f"fresh download: removed {len(old)} old files from {out_dir}")

    jobs: list[tuple[str, str, str]] = []  # (universe, ticker, filename)
    for u in universes:
        spec = UNIVERSES[u]
        jobs += [(u, t, f"{t.replace('/', '_')}.parquet") for t in spec["tickers"]]
        jobs.append((u, str(spec["benchmark_ticker"]), f"benchmark_{spec['benchmark']}.parquet"))

    files: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for u, ticker, fname in jobs:
        path = out_dir / fname
        if not fresh and path.exists():
            df = pd.read_parquet(path)
        else:
            try:
                df = fetch_with_retries(fetch, ticker, start, end)
            except RuntimeError as exc:
                failures[ticker] = str(exc)
                logger.error(str(exc))
                continue
            df.to_parquet(path)
            time.sleep(pause)
        problems = validate(df)
        files[fname] = {
            "ticker": ticker, "universe": u, "rows": int(len(df)),
            "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
            "sha256": _sha256(path), "problems": problems,
        }
        level = logger.warning if problems else logger.info
        level(f"{ticker:>14}: {len(df):5d} rows {df.index[0].date()} -> {df.index[-1].date()}"
              + (f"  PROBLEMS: {problems}" if problems else ""))

    try:
        import yfinance

        yf_version = yfinance.__version__
    except Exception:
        yf_version = "n/a"
    manifest = {
        "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "Yahoo Finance via yfinance (auto_adjust=True)",
        "yfinance_version": yf_version,
        "start": start, "end_exclusive": end, "universes": universes, "fresh": fresh,
        "n_files": len(files), "failures": failures, "files": files,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", choices=["nifty", "sp500", "both"], default="both")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None, help="exclusive end date (default: tomorrow, i.e. include today)")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--resume", action="store_true", help="keep existing files instead of a fresh download")
    a = p.parse_args(argv)
    universes = ["nifty", "sp500"] if a.universe == "both" else [a.universe]
    m = download(universes, Path(a.out), a.start, a.end, fresh=not a.resume)
    bad = {k: v["problems"] for k, v in m["files"].items() if v["problems"]}
    logger.info(f"wrote {m['n_files']} files + _manifest.json to {a.out}")
    if m["failures"]:
        logger.error(f"{len(m['failures'])} downloads failed: {list(m['failures'])}")
    if bad:
        logger.warning(f"{len(bad)} files have validation problems (see _manifest.json)")
    # Benchmarks are mandatory; individual stocks may fail (dropped later by coverage rules).
    bench_failed = [t for t in m["failures"] if t.startswith("^")]
    return 1 if bench_failed or m["n_files"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
