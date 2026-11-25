# Run log

This folder is the project's lab notebook. Every phase of the v2 build
(proposal: `Document/portfolio_optimization_proposal_v2.docx`) gets its own
sub-folder with three things:

| File | Purpose |
|---|---|
| `phaseN/PHASEn_RUNLOG.md` | Phase *n* runlog: what was done, why, what was run, what came out, what is still open. |
| `phaseN/PHASEn_HOW_TO_RUN.md` | Instructions to run **that phase only** on your machine (Windows / VS Code first). |
| `phaseN/artifacts/` | Small outputs of the phase (reports, JSON, CI logs). Data and models are regenerated, never committed. |

Current files: [`phase0/PHASE0_RUNLOG.md`](phase0/PHASE0_RUNLOG.md) · [`phase0/PHASE0_HOW_TO_RUN.md`](phase0/PHASE0_HOW_TO_RUN.md)

A consolidated `COMPLETE_RUNLOG.md` is written at the end of Phase 7.

## Branch strategy

Each phase is built on its own branch, stacked on the previous one, and the
whole stack is merged into `main` at the end:

```
main ── phase0 ── phase1 ── phase2 ── phase3 ── phase4 ── phase5 ── phase6 ── phase7 ──► merge to main
```

* `phaseN` branches from `phaseN-1` (so later phases contain earlier fixes).
* CI runs on every push to every branch; a phase is "done" when its exit gate passes.
* Final step: merge `phase7` into `main` (fast-forward / PR), then tag `v2.0.0`.

## Status

| Phase | Weeks | Branch | Goal | Exit gate | Status |
|---|---|---|---|---|---|
| 0 Remediate | 1–2 | `phase0` | Fix v1 defects; leakage-safe core; CI green | Leakage suite passes on v1 data; CI < 10 min | ✅ done — exit gate passed on fresh real data; CI green (Ubuntu + Windows) |
| 1 Data & backtester | 3–5 | `phase1` | PIT data, calendars, FX, event-driven simulator, CPCV | 1/N and min-var reproduce reference stats | ⏳ |
| 2 Signals & risk | 6–8 | `phase2` | Forecasters, conformal, regimes, risk models | ≥ 1 forecaster passes rank-IC rule | ⏳ |
| 3 Classical & RL | 9–10 | `phase3` | Convex/robust/CVaR/HRP/BL/MIQP; redesigned RL | Strategy CPCV report with DSR/PBO | ⏳ |
| 4 Quantum track | 11–14 | `phase4` | IR compiler, QUBO/HUBO, simulators, hardware | Pre-registered benchmark executed | ⏳ |
| 5 Solver Router | 15–16 | `phase5` | Adapters, race/learned policies, certificates | Router regret target met | ⏳ |
| 6 Production | 17–18 | `phase6` | API v2, async jobs, MLOps, monitoring, paper trading | Load test meets SLOs; alert drills pass | ⏳ |
| 7 Validate & publish | 19–20 | `phase7` | Holdout unlock, reports, docs, demo | Acceptance criteria signed off | ⏳ |

## Platforms

Primary development platform: **Windows 11 + VS Code + PowerShell**. Each phase's
`PHASEn_HOW_TO_RUN.md` gives VS Code steps and `.\tasks.ps1 <task>` commands first, and
`make <task>` for Linux/macOS. CI runs Linux and Windows jobs on every push.
Docker Desktop (WSL 2 backend) is needed only for the container services
(MLflow server, Airflow, full stack).

## v1 → v2 replacement tracker

Phases are developed and run one at a time; each replaces the old v1 parts it covers.

| v1 component | Replaced in | Status |
|---|---|---|
| Market data download (`data/ingestion/fetch_data.py`) | Phase 0 (`pfo-download`, fresh) → Phase 1 (point-in-time platform, FX, DVC remote) | ✅ Phase 0 part done |
| v1 features (`feature_engineering.py`, `data/processed`) | Phase 0 (causal features) → Phase 1 (feature store) | ✅ Phase 0 part done |
| Walk-forward backtest + results (`backtesting/`) | Phase 0 (corrected engine) → Phase 1 (event-driven, CPCV) | ✅ Phase 0 part done |
| RL environment / training (`models/rl_agent/`) | Phase 0 (timing fix) → Phase 3 (redesign) | ✅ Phase 0 part done |
| Macro features (`macro_features.py`) | Phase 1 | ⏳ |
| LSTM / ARIMA forecasting (`models/forecasting/`) | Phase 2 | ⏳ |
| Markowitz / Black-Litterman (`models/classical/`) | Phase 3 | ⏳ |
| API, dashboard, Airflow, monitoring (`serving/`, `dashboard/`, `mlops/`) | Phase 0 (defect fixes) → Phase 6 (rebuild) | ✅ Phase 0 part done |
| README, SYSTEM_DESIGN | Phase 7 | ⏳ |

## Conventions

* Dates are ISO (YYYY-MM-DD), times IST unless stated.
* Every number quoted in a RUNLOG must come from an artifact in that phase's `artifacts/`
  folder or from a command whose exact invocation is written next to it.
* Synthetic-data results are always labelled as such; they prove plumbing, not performance.
