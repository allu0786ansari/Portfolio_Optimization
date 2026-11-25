# Phase 0 — How to run (Remediation)

This guide covers **Phase 0 only**: install, download fresh market data, pass the
leakage exit gate, and produce the corrected v1 baseline. Later phases have their
own `PHASEn_HOW_TO_RUN.md`.

Platform: **Windows 10/11 + VS Code + PowerShell** (Linux/macOS: use `make <task>`
instead of `.\tasks.ps1 <task>`). Docker is **not** needed for Phase 0.

What Phase 0 replaces:

| Old (v1) | Replaced by (Phase 0) |
|---|---|
| `data/ingestion/fetch_data.py` (kept stale files, END_DATE fixed at 2024-12-31) | `pfo-download`: fresh download to today, validated, with `data/raw/_manifest.json` |
| v1 processed features in `data/processed/` | causal features computed on the fly (`pfo.features`) |
| `backtesting/walk_forward.py` results (look-ahead, overlapping windows) | `pfo-v1-baseline` corrected report |
| `backtesting/backtest_results.json`, `backtest_report.html`, `saved_models/*.zip` | deleted (invalid) |

---

## 1. Prerequisites (once)

| Tool | Install | Check in a new PowerShell |
|---|---|---|
| Git for Windows | https://git-scm.com/download/win | `git --version` |
| VS Code | https://code.visualstudio.com | — |
| Python 3.12, 64-bit (tick **Add python.exe to PATH**) | https://www.python.org/downloads/ | `py -3.12 --version` |
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `uv --version` |

Allow local scripts once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## 2. Get the code

```powershell
git clone https://github.com/allu0786ansari/Portfolio_Optimization.git
cd Portfolio_Optimization
git checkout phase0
code .
```

Install the recommended extensions when VS Code asks.

## 3. Install

```powershell
.\tasks.ps1 setup      # creates .env from .env.example; open .env and set API_KEY
.\tasks.ps1 install    # .venv from uv.lock (3-6 min the first time)
```

VS Code: *Ctrl+Shift+P → Python: Select Interpreter → `.venv\Scripts\python.exe`*.

## 4. Offline checks (synthetic data, ~2 min)

```powershell
.\tasks.ps1 check
```

VS Code: *Ctrl+Shift+B*. Expect ruff and mypy clean, **11 leakage tests passed**,
**140 passed, 15 skipped** (the skipped tests need the data from step 5).

## 5. Download fresh market data

```powershell
.\tasks.ps1 data
```

* Deletes everything in `data\raw\` and downloads 20 Nifty 50 + 20 S&P 500 stocks and
  the two benchmark indices from **2010-01-01 to today** (Yahoo Finance, adjusted prices).
* Writes `data\raw\_manifest.json`: download time, rows, first/last date and SHA-256 per file,
  plus any validation problems.
* Exit code 1 if a benchmark failed; re-run (Yahoo sometimes rate-limits).
* Options: `.\tasks.ps1 data --start 2015-01-01`, `.\tasks.ps1 data --resume` (keep files).

## 6. Exit gate on real data

```powershell
.\tasks.ps1 leakage-real
```

Must print a table where every check is `True` and end with `leakage scan PASSED`
(exit code 0). Output: `reports\phase0\leakage_scan_real.json`.

Then re-run the tests; the 15 data-dependent tests now run too:

```powershell
.\tasks.ps1 test       # expect 155 passed, 0 skipped (152 + the 3 slow tests)
```

## 7. Corrected v1 baseline on real data (~15-40 min)

```powershell
.\tasks.ps1 phase0-baseline
```

Re-trains PPO inside every walk-forward fold for both universes (27 folds each with
data from 2010; about 7 minutes on a GitHub runner, longer on a laptop). Output in
`reports\phase0\real\`: `report.md` (open with *Ctrl+Shift+V*), `summary.json`,
`<universe>\equity.csv`, `fold_metrics.csv`.

Faster variants: `.\tasks.ps1 phase0-baseline --rl-timesteps 0` (no RL) or
`--universe sp500`. All flags: `uv run pfo-v1-baseline --help`.

## 8. Save the results in the run log

```powershell
New-Item -ItemType Directory -Force runlog\phase0\artifacts\real | Out-Null
Copy-Item reports\phase0\real\report.md, reports\phase0\real\summary.json, `
          reports\phase0\leakage_scan_real.json, data\raw\_manifest.json `
          -Destination runlog\phase0\artifacts\real\
git add runlog/phase0/artifacts/real
git commit -m "phase0: real-data run (local)"
git push
```

(The same run is also executed on GitHub by the *Phase 0 - real data run* workflow;
its outputs are already in `runlog\phase0\artifacts\real\`.)

## Troubleshooting

| Symptom | Fix |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or run tasks from VS Code. |
| `uv` not recognized | Install uv (step 1), then open a new terminal / restart VS Code. |
| `pfo-download` fails with rate-limit errors | Wait a few minutes and run `.\tasks.ps1 data --resume`. |
| `FileNotFoundError: No price files for nifty` | Run step 5 first. |
| `UnicodeEncodeError: 'charmap' codec` | Use `tasks.ps1` or a VS Code terminal (they set `PYTHONUTF8=1`), or `$env:PYTHONUTF8 = "1"`. |
| Tests not listed in the Testing panel | Select `.venv\Scripts\python.exe`, then *Testing → Refresh*. |
