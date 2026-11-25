# Phase 0 — How to run

Phase 0 delivers the leakage-safe core (`src/pfo`), fixes the v1 defects and
produces the **corrected v1 baseline**. Everything except the real-data steps
runs offline on synthetic data.

## 0. Prerequisites

* Python 3.12 (3.11 also works) and [uv](https://docs.astral.sh/uv/) ≥ 0.11
  (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
* Git, and Docker only if you want the container stack.
* Internet access to Yahoo Finance for the real-data steps (step 4).

```bash
git clone https://github.com/allu0786ansari/Portfolio_Optimization.git
cd Portfolio_Optimization
git checkout phase0
cp .env.example .env        # then edit: at least API_KEY (any long random string)
```

## 1. Install (≈2–5 min, first time)

```bash
make install                 # = uv sync --extra dev --extra rl --extra ml --extra serving --extra data
                             #   + pre-commit install
```

On Windows without `make`, run the commands shown in the `Makefile` directly, e.g.
`uv sync --extra dev --extra rl --extra ml --extra serving --extra data`.
`uv sync` installs PyPI's PyTorch, which on Linux includes CUDA (~3 GB). For a
CPU-only environment use `PYTHON=.venv/bin/python bash scripts/install_deps.sh "dev rl ml serving data"`
after `uv venv`.

## 2. Quality gates (offline, ≈1 min)

```bash
make lint          # ruff — expect "All checks passed!"
make typecheck     # mypy on src/pfo — expect "Success: no issues found"
make leakage       # leakage suite + synthetic leakage scan — the Phase 0 exit gate
make test          # full suite with coverage (expect 134 passed, 15 skipped without data)
```

The 15 skipped tests are v1 tests that need downloaded data; they run after step 4.

## 3. Synthetic smoke run of the corrected baseline (≈3 min)

```bash
make phase0-smoke
# -> reports/phase0/synthetic/report.md, summary.json, <universe>/equity.csv, fold_metrics.csv
```

Synthetic prices have no exploitable signal, so no strategy should beat equal
weight with significance. If something does, suspect leakage.

## 4. Real data: exit gate + corrected v1 baseline

```bash
make data                    # downloads 40 tickers + 2 benchmarks (START_DATE..today) into data/raw
make leakage-real            # exit gate on real data -> reports/phase0/leakage_scan_real.json (exit code 0 = pass)
make phase0-baseline         # corrected baseline -> reports/phase0/real/report.md   (~10-25 min, RL re-trained per fold)
pytest tests -m "not slow"   # now the 15 data-dependent v1 tests run as well
```

Useful options (`uv run pfo-v1-baseline --help`):

| Flag | Default | Meaning |
|---|---|---|
| `--universe nifty\|sp500\|both` | both | Each market on its own exchange calendar (no NSE/NYSE intersection, no FX mixing) |
| `--train-days / --test-days` | 504 / 126 | Walk-forward windows; test windows never overlap |
| `--rebalance` | 21 | Trading days between decisions |
| `--execution-lag` | 1 | Decide at close *t*, trade at close *t+1* |
| `--cost-bps` | 10 | Cost per unit of one-way turnover |
| `--rl-timesteps` | 30000 | PPO steps per fold (0 disables RL) |

**Please commit the real-data outputs** for the run log:

```bash
mkdir -p runlog/phase0/artifacts/real
cp reports/phase0/real/report.md reports/phase0/real/summary.json reports/phase0/leakage_scan_real.json \
   runlog/phase0/artifacts/real/
git add runlog/phase0/artifacts/real && git commit -m "phase0: real-data corrected baseline" && git push
```

## 5. (Optional) MLflow training + API with the fixed code

```bash
docker compose up -d mlflow                         # pinned MLflow 3.18 server on :5000
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m models.rl_agent.train_agent --algo ppo --universe sp500 --timesteps 50000
uv run uvicorn serving.main:app --port 8000          # /ready turns 200 once the champion + spec load
curl -s -X POST localhost:8000/predict -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
     -d '{"tickers":["AAPL","MSFT","NVDA"],"date":"2024-06-28"}'
```

Note: the v1 model zips in `models/rl_agent/saved_models/` were trained in the
leaky v1 environment and have no `serving_spec`; the API deliberately refuses to
serve them. Train a new model as above.

## 6. (Optional) Full container stack

```bash
# .env must define API_KEY, TRAINING_SERVICE_API_KEY (= API_KEY), AIRFLOW_FERNET_KEY,
# AIRFLOW_WEBSERVER_SECRET_KEY, AIRFLOW_ADMIN_PASSWORD, GRAFANA_ADMIN_PASSWORD
make up            # mlflow, api, dashboard, prometheus, grafana, airflow
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: No price files for nifty` | Run `make data` (needs internet). |
| `/predict` returns 503 "API_KEY is not configured" | Set `API_KEY` in `.env` / environment (no default by design). |
| `/ready` is 503 | No champion in MLflow and no local model with a `.spec.json` — train one (step 5). |
| `uv lock --check` fails in CI | You edited `pyproject.toml`; run `uv lock` and commit `uv.lock`. |
| docker compose complains `set AIRFLOW_FERNET_KEY in .env` | Fill the Airflow/Grafana secrets in `.env`. |
