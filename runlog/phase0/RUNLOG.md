# Phase 0 — Remediation (proposal Weeks 1–2) — RUNLOG

| | |
|---|---|
| Branch | `phase0` (from `main` @ `58291e5`) |
| Date | 2026-11-25 (IST) |
| Goal | Fix the v1 defects found in the audit; build the leakage-safe core every later phase depends on; get CI green. |
| Exit gate (proposal §17.1) | Leakage suite passes on v1 data; CI < 10 min. |
| Status | **Built, verified on synthetic data, CI green on GitHub.** Real-data gate must be run locally (`make leakage-real`) because Yahoo Finance is not reachable from the build sandbox. |

## 1. Environment used for the build

* Linux sandbox, Python 3.12.3, uv 0.11.32, no Docker daemon, outbound access limited to package registries and GitHub (Yahoo Finance and download.pytorch.org were blocked).
* Locked versions (`uv.lock`, 224 packages) include: torch 2.15.0, stable-baselines3 2.9.2, gymnasium 1.4.0, mlflow(-skinny) 3.18.0, cvxpy 1.9.3, fastapi 0.165.0, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, scipy 1.18.1.

## 2. What was built

### 2.1 New package `src/pfo` (leakage-safe core)

| Module | Purpose |
|---|---|
| `pfo/timing.py` | The timing contract (decide at close *t* with rows ≤ *t*; execute at *t + lag*; earn from *t + lag + 1*), `forward_returns`, oracle-canary scanner, generic future-shuffle causality checker. |
| `pfo/backtest/splits.py` | Walk-forward folds whose test windows **cannot** overlap (overlap raises unless explicitly allowed). |
| `pfo/backtest/costs.py` | Linear cost model (bps per unit one-way turnover). |
| `pfo/backtest/engine.py` | Exact daily simulator: execution lag, weight drift, costs vs drifted weights, cash remainder, long-only checks. |
| `pfo/backtest/metrics.py` | Sharpe/Sortino/CAGR/MDD/VaR/CVaR on unique monotonic dates (duplicates raise), stationary-bootstrap Sharpe CI, Newey-West t-test on non-overlapping calendar months, paired Sharpe-difference bootstrap. |
| `pfo/backtest/walkforward.py` | Fit per fold → decide on the test window via `History` (rows ≤ t only) → stitch → simulate every strategy identically. |
| `pfo/features/basic.py` | Causal v1 feature set (RSI fixed to price changes, rescaled) + `FoldScaler` fitted on training rows only. |
| `pfo/data/market.py`, `universes.py` | Per-market loader on one exchange calendar (no NSE∩NYSE intersection, no FX mixing), synthetic factor-model market. |
| `pfo/strategies/` | `Strategy` interface with structural information sets; `EqualWeight`, `MinVariance`, `MarkowitzV1` (v1 rule, honest docstring, no hard-coded ECOS). |
| `pfo/rl/env.py`, `rl/strategy.py` | `PortfolioEnvV2` (correct timing, concentration-capable action projection, costs vs drift, window-confined episodes, obs uses previous *target* weights) and a PPO allocator re-trained inside every fold. |
| `pfo/reports/v1_baseline.py` | CLI `pfo-v1-baseline`: the corrected v1 baseline report (JSON + Markdown + CSV). |
| `pfo/reports/leakage_scan.py` | CLI `pfo-leakage-scan`: the exit-gate checks on real or synthetic data (exit code 0 = pass). |

### 2.2 v1 defects → fixes

| # | v1 defect (audit) | Fix in phase 0 | Verified by |
|---|---|---|---|
| 1 | RL observed `r[t]` and was rewarded with `r[t]` (look-ahead) | `PortfolioEnvV2` rewards row `t+lag+1`; legacy `models/rl_agent/portfolio_env.py` now subclasses it | `test_env_observation_is_independent_of_future_returns`, `test_v1_timing_bug_is_exploitable_and_v2_timing_is_not`, `test_legacy_env_is_the_corrected_env` |
| 2 | Overlapping 126-day windows stepped by 21 days concatenated (735 days counted as 3,780) | `walk_forward_splits` (step = test length, overlap refused); metrics reject duplicated dates; `backtesting/walk_forward.py` now forwards to the corrected engine | `test_default_splits_never_overlap`, `test_v1_style_overlapping_splits_are_refused`, `test_metrics_reject_duplicated_dates`, `test_stitched_oos_dates_are_unique_and_monotonic` |
| 3 | One RL model trained once; "validation" on the full history | RL re-trained per fold in the baseline; `train_agent.py` validates on rows 70–85% only | `test_validation_env_only_uses_validation_rows`, `test_ppo_allocator_trains_and_decides` |
| 4 | No costs for RL in backtest; MVO held 126 days | One simulator, one cost model, one execution lag, same rebalance schedule for all strategies | `test_engine_charges_costs_on_drifted_weights` |
| 5 | softmax(a), a∈[−1,1] caps weights at 15.9% (N=40) | `project_action(a, scale=5)` → up to 99.8% | `test_project_action_and_bounds`; `artifacts/v1_timing_bug_demo.txt` |
| 6 | Raw features clipped to ±10 (RSI constant); NSE∩NYSE calendar; no FX | Fold scaler; per-market calendars (cross-market + FX → Phase 1) | `test_fold_scaler_uses_training_rows_only`, leakage scan |
| 7 | `/retrain/*` 404 (no router prefix); training blocked the event loop; MLflow at `localhost` | `prefix="/retrain"`, sync handlers (thread pool), job lock, `MLFLOW_TRACKING_URI` everywhere, errors → HTTP 500 | `test_retrain_routes_use_the_prefix_airflow_calls`, `test_retrain_status_requires_key` |
| 7b | `evaluate-champion` promoted the highest version number | Promotes the version of the best run; 5% rule works for negative Sharpe | `test_train_registers_this_runs_version_and_serving_spec` |
| 8 | Hot-reload cosmetic (served zip from disk); Prometheus counters never incremented | One model path: registry policy + `serving_spec.json`; local zip only with a spec; middleware records requests; `policy.predict` (clipped) replaces `_predict`; `date` honoured | `test_predictor_serves_local_model_with_spec`, `test_v1_zip_without_spec_is_not_served`, `test_predict_increments_prometheus_counter`, `test_date_is_forwarded_to_predictor` |
| 9 | CI lint red; `.env` copied into images; secrets in compose; unpinned MLflow; invalid `pyproject` backend; corrupted `.gitignore`; misnamed `.pre-commit-config.yml` | Ruff clean; Dockerfiles copy no `.env` (multi-stage, locked deps, CPU torch); compose reads secrets from `.env` (`:?` guards); pinned MLflow 3.18 server image with a named volume; valid `pyproject` + `uv.lock`; `.gitignore` repaired; `.pre-commit-config.yaml`; API key has no public default (fails closed) | CI job + `test_missing_server_key_fails_closed`; `docker compose config` validated |
| — | MLflow 3 defaults to `pt2` serialization, which drops `policy.predict` (found during Phase 0) | `serialization_format="pickle"` | registry load + predict in `test_train_registers_this_runs_version_and_serving_spec` |
| — | `END_DATE` hard-coded 2024-12-31 → nightly refresh fetched the same data | Empty `END_DATE` = today | — |
| — | Backtest HTML embedded plotly.js 4× (20 MB) | `include_plotlyjs="cdn"` once | — |

### 2.3 Tooling

* `pyproject.toml` (src layout, extras `data/ml/rl/serving/dashboard/dev`, console scripts) + `uv.lock`.
* `scripts/install_deps.sh` exports the lock, filters the CUDA wheels and installs CPU torch at the locked version (Docker + CI). Old `requirements/*.txt` removed (they had drifted).
* CI (`.github/workflows/ci.yml`): `uv lock --check` → ruff → mypy → leakage gate → full suite with coverage → synthetic leakage scan + baseline smoke (artifacts uploaded) → Docker build + smoke tests (health, fail-closed auth, `/retrain` prefix, no `.env` in image) for API and dashboard.
* CD now runs only after CI succeeds on `main`.
* `Makefile` targets: `install lint typecheck test leakage leakage-real phase0-smoke phase0-baseline …`.

## 3. Commands run and results

All outputs below are in `runlog/phase0/artifacts/`.

| Command | Result | Artifact |
|---|---|---|
| `uv lock --check; ruff check .; mypy; pytest tests/leakage; pytest tests --cov` | lock in sync · ruff clean · mypy clean (22 files) · **11/11 leakage tests pass** · **134 passed, 15 skipped**, coverage **94 %** of `src/pfo`, 45 s wall time | `ci_local_run.txt` |
| Same suite with synthetic files written in the v1 data layout (then deleted) | **149 passed, 0 skipped** (the 15 v1 data-dependent tests pass against the corrected env) | — |
| `pfo-leakage-scan --source synthetic` | 6/6 checks pass | `leakage_scan_synthetic.json` |
| `pfo-leakage-scan --source v1 --universe both` on synthetic v1-layout files | 12/12 checks pass (exercises the real-data code path) | `leakage_scan_v1layout_synthetic.json` |
| v1 timing-bug demo | same-day "oracle" Sharpe **24.84** under v1 timing vs **0.75** under v2 timing (EW 1.10) | `v1_timing_bug_demo.txt` |
| `pfo-v1-baseline --source synthetic --rl-timesteps 20000 --n-boot 500` (142 s) | 10 non-overlapping folds, 1,259 unique OOS days. EW Sharpe 0.84, min-var 0.82, Markowitz-v1 0.83, PPO −0.14; no strategy beats EW significantly (as expected on signal-free data) | `synthetic_smoke/report.md`, `summary.json` |

### 3.1 GitHub Actions (first push of `phase0`)

| Run | Job | Result | Duration |
|---|---|---|---|
| [CI #45660212807](https://github.com/allu0786ansari/Portfolio_Optimization/actions/runs/45660212807) | Lint, type-check, leakage gate, tests (+ synthetic scan and baseline smoke) | ✅ success | 1 min 32 s |
| | Docker build + smoke test (API and dashboard images, health, fail-closed auth, `/retrain` prefix, no `.env` in image) | ✅ success | 10 min 7 s (first build, cold cache) |

Exit-gate note: the test/leakage job runs in well under 10 minutes. The Docker job took about 10 minutes on a cold
build cache; later runs reuse the GitHub Actions layer cache. If it stays slow, Phase 6 will build images only on
`main` and on PRs.

## 3.2 Windows / VS Code support (added 2026-11-25, after the first push)

The developer works on Windows with VS Code, so every phase must run there natively.

| Change | Why |
|---|---|
| `tasks.ps1` (PowerShell 5.1+ task runner mirroring the `Makefile`) | `make` and bash are not available on Windows. |
| `.vscode/` settings, tasks (Windows → `tasks.ps1`, Linux/macOS → `make`), launch configs, extension recommendations | Run, test and debug every phase from VS Code. |
| `.gitattributes` (LF for `.sh`/Dockerfiles/Python, CRLF for `.ps1`) | Git for Windows converts to CRLF by default, which would break `install_deps.sh` inside Linux Docker images. |
| UTF-8 on every file read/write; `PYTHONUTF8=1` in tasks and VS Code terminals | Windows defaults to cp1252: writing the baseline report (`→`, `Δ`, `—`) would have raised `UnicodeEncodeError`. |
| `.env.example` without inline comments | VS Code's `envFile` parser would keep `# comment` as part of the value (e.g. `END_DATE`). |
| Portable SQLite URI in tests (`as_posix()`) | Windows paths with backslashes. |
| New CI job **windows-latest**: `tasks.ps1 install` → `check` → `phase0-smoke`, under Windows PowerShell 5.1 | Windows compatibility is tested on every push of every phase. |

Dependency check: `uv.lock` was scanned for Windows wheels. Every package has one except Linux-only CUDA
wheels, `triton` and `uvloop`, which the lock's platform markers already skip on Windows, and
`antlr4-python3-runtime` (pure-Python source build). PyPI's Windows PyTorch wheel is CPU-only.

Platform notes for later phases: Airflow (Phase 6) runs only inside Docker on Windows (it has no native
Windows support), and so do the Celery workers. GPU acceleration (simulated bifurcation, foundation
models) is optional and needs the CUDA build of PyTorch. Everything else (Qiskit, D-Wave Ocean, cvxpy,
Gurobi/HiGHS, LightGBM, Stable-Baselines3, MLflow, FastAPI, Streamlit) runs natively.

## 4. Decisions taken

1. **Execution lag 1 day by default** (decide at close *t*, trade at close *t+1*). Market-on-close (`--execution-lag 0`) is available for sensitivity.
2. **One exchange calendar per universe in Phase 0.** The 40-asset mixed universe returns in Phase 1 with proper FX and holiday handling.
3. **The RL environment shows the agent its previous target weights**, not drifted holdings, because with lag ≥ 1 the drift uses returns after *t*.
4. **No public API-key default**: unset `API_KEY` → 503 (fail closed).
5. **v1 model zips are not served** (no observation spec, trained with look-ahead). They stay in git for the record.
6. **v1 backtest JSON kept** for the dashboard but flagged as invalid on every page.
7. **Lockfile resolved against PyPI**; CPU torch is substituted at install time for Docker/CI (the PyTorch index is not reachable from the build sandbox).

## 5. Not done here / handed to later phases

* **Real-data exit gate and corrected numbers** — run `make data && make leakage-real && make phase0-baseline` locally and commit the outputs to `runlog/phase0/artifacts/real/` (instructions in `HOW_TO_RUN.md` §4).
* Docker images were not built in the sandbox (no daemon); the CI `docker` job builds and smoke-tests them on GitHub.
* Phase 1: point-in-time data platform, cross-market universe with FX, square-root impact costs, CPCV, DSR/PBO, SPA.
* Phase 6: training moves out of the API process into workers (ADR-03); Airflow 3; Grafana provisioning.
* v1 LSTM/ARIMA modules were not changed (no defect affects them; they are re-evaluated in Phase 2).
