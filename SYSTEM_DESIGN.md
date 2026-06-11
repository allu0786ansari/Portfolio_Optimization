# AI-Powered Portfolio Optimization Engine

**Author:** Allaudin Ansari
**Repository:** github.com/allu0786ansari/Portfolio_Optimization  
**Stack:** PyTorch · Stable-Baselines3 · FastAPI · Streamlit · MLflow · Airflow · Prometheus · Docker  

---

## 1. Architecture Overview

The system is organised into five layers. Data flows sequentially from left to right; the retraining loop feeds back from monitoring to data ingestion.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Ingestion & Feature Engineering                               │
│  yfinance → 40 assets (20 Nifty50 + 20 S&P500) + 2 benchmarks              │
│  Features: log_return, momentum_21/63/126d, volatility_21/63d,               │
│            beta_63d, rsi_14, price_vs_ma50, price_vs_ma200                   │
│  Output: versioned Parquet files (DVC tracked)                               │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  Layer 2: Forecasting Models                                                 │
│  LSTM (PyTorch): seq_len=30, hidden=64, 2 layers, dropout=0.2               │
│  Input: 10 features × 30 days per asset                                      │
│  Output: predicted next-day log return per asset                             │
│  Baseline: ARIMA (statsmodels) for directional accuracy comparison           │
│  Hyperparameter tuning: Optuna (100 trials)                                  │
│  Tracking: MLflow experiment "lstm_forecaster"                               │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ expected returns
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  Layer 3: Portfolio Optimisation                                             │
│                                                                              │
│  RL Agent (primary)                  Classical Baselines                     │
│  ─────────────────                   ────────────────────                    │
│  PortfolioEnv (custom Gymnasium)     Markowitz MVO (cvxpy)                  │
│  State: (40 assets × 11 dims) = 440  Ledoit-Wolf covariance shrinkage       │
│  Action: Box(-1, 1, 40) → softmax    Max weight per asset: 40%              │
│  Reward: log_return − TC penalty     Monthly rebalancing (21-day)           │
│  TC rate: 0.1% per unit of turnover  Black-Litterman model                  │
│  Episode: 252 steps, random start    Equal-weight 1/N baseline              │
│                                                                              │
│  Algorithms: PPO (100k steps) and SAC (150k steps)                          │
│  Tracking: MLflow experiment "rl_agent_training"                            │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  Layer 4: Walk-Forward Backtesting                                           │
│  Train: 504 days (2 years) │ Test: 126 days (6 months) │ Step: 21 days      │
│  30 rolling windows, 3,780 total test days (2021-11-30 → 2024-12-26)        │
│  Metrics: Sharpe, Sortino, Calmar, CAGR, max drawdown, VaR 95%, CVaR 95%   │
│  Statistical test: two-sided t-test on monthly excess returns vs EW         │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────────┐
│  Layer 5: Serving, Monitoring & MLOps                                        │
│  FastAPI (/health /ready /predict /metrics /retrain/*)                       │
│  Streamlit dashboard (4 pages: Performance, Allocation, Comparison, Frontier)│
│  Prometheus scrapes /metrics every 15s → Grafana alerts                     │
│  Airflow DAG triggers FastAPI /retrain/* endpoints nightly                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Service topology (docker-compose)

| Container | Port | Role |
|---|---|---|
| `portfolio-api` | 8000 | FastAPI — inference + retraining endpoints |
| `portfolio-dashboard` | 8501 | Streamlit — visualisation |
| `portfolio-mlflow` | 5000 | MLflow tracking server (SQLite backend) |
| `portfolio-prometheus` | 9090 | Metrics scraper |
| `portfolio-grafana` | 3000 | Dashboards and alerting |
| `portfolio-airflow-webserver` | 8080 | DAG management UI |
| `portfolio-airflow-scheduler` | — | Nightly trigger (SequentialExecutor) |

---

## 2. Key Design Decisions and Tradeoffs

### 2.1 Reinforcement Learning over Markowitz as the primary model

Markowitz Mean-Variance Optimisation assumes static, normally-distributed asset returns and is highly sensitive to estimation errors in expected returns and the covariance matrix. Both assumptions are violated in real markets, particularly during regime changes.

The RL agent (PPO) treats portfolio allocation as a sequential decision problem. The key advantages in this implementation:

- **Transaction costs are native to the reward signal.** The reward function subtracts `0.1% × turnover` at every step, so the agent learns naturally to trade only when expected return improvement exceeds friction cost. Markowitz has no such mechanism without explicit penalty terms.
- **No distributional assumptions.** The agent learns from the empirical return distribution directly.
- **Downside-aware training.** The per-step reward is `log_return − TC_penalty`. The Sortino-based `step_reward` function penalises only negative excess returns, not all volatility, making the agent more aggressive during uptrends and conservative during drawdowns.

The tradeoff is sample efficiency — PPO required 100,000 environment steps to converge, compared to Markowitz which has a closed-form solution. This is acceptable in a nightly retraining context but would be prohibitive for intraday strategies.

Markowitz is retained as a live baseline in the backtesting engine. Walk-forward results confirm the RL agent outperforms it by 0.69 Sharpe points (1.47 vs 0.77).

### 2.2 Preventing look-ahead bias in backtesting

The most common mistake in financial ML is data leakage, where future information inadvertently enters training. Three mechanisms prevent it here:

**Temporal walk-forward split.** The dataset is never shuffled. For each of 30 windows: train on the preceding 504 days, evaluate on the next 126 days, step forward by 21 days. The model never sees test data during training.

**Per-window feature scaling.** The `RobustScaler` in `dataset.py` is fitted exclusively on each training window using `fit_transform`, then applied forward to the corresponding test window using `transform`. A scaler fitted on the full dataset would introduce leakage because test-period statistics would influence the scaling of training data.

**Benchmark refit per window.** The Markowitz optimiser recomputes expected returns and the Ledoit-Wolf covariance matrix from each training window only. No look-forward covariance is used.

The statistical significance test (`ttest_excess_returns` in `metrics.py`) aggregates monthly excess returns across all 30 windows and runs a two-sided t-test against the equal-weight benchmark. Result: t = 11.10, p = 4.27 × 10⁻²², confirming the alpha is not attributable to chance.

### 2.3 Ledoit-Wolf covariance shrinkage

With 40 assets and a 2-year training window (~504 observations), the sample covariance matrix is near-singular and noisy. The Ledoit-Wolf estimator (`sklearn.covariance.LedoitWolf`) blends the sample covariance with a scaled identity matrix:

```
Σ_shrunk = (1 − α) × Σ_sample + α × μ × I
```

The shrinkage coefficient α is estimated analytically, not by cross-validation. This produces a well-conditioned, invertible matrix — necessary for the cvxpy quadratic programming solver in `markowitz.py`. The maximum single-asset weight is capped at 40% to prevent extreme concentration in the optimal solution.

### 2.4 Airflow as a pure HTTP orchestrator

A common anti-pattern in MLOps is deploying heavy ML libraries inside the Airflow worker. This creates dependency conflicts, bloated images, and scheduler instability.

In this system, `retrain_dag.py` contains zero ML code. Every task is a `requests.post()` call to a FastAPI `/retrain/*` endpoint:

```
fetch_new_data → engineer_features → [train_ppo ∥ train_sac] → evaluate_champion → notify
```

PPO and SAC training run in parallel inside the API container, which already has torch, SB3, gymnasium, cvxpy, and MLflow installed. Airflow only needs `requests`, `loguru`, and `flask` (for the Grafana webhook server). The Airflow Docker image is under 500 MB; the full ML stack stays in the API image.

This also decouples scheduling from execution: if the API container is restarted mid-training, the training continues. Airflow only polls the result via `/retrain/status`.

### 2.5 Zero-downtime model hot-reload

The `ModelRegistry` class in `model_loader.py` runs a background thread that polls the MLflow registry every 300 seconds for a new champion alias. When a new version is detected, it loads the new policy into memory under a `threading.RLock` before releasing the old one. All in-flight requests complete against the old model; subsequent requests use the new one. No server restart is required.

The champion-challenger promotion logic requires the challenger to improve the current champion's validation Sharpe ratio by at least 5% before being promoted. This threshold prevents unnecessary model churn from noise in the evaluation episodes.

### 2.6 Observation space design

The RL state vector concatenates the flattened feature matrix with the current portfolio weights:

```
obs = [asset_features (40 × 10 = 400 dims)] + [current_weights (40 dims)] = 440 dims
```

Including current weights in the state gives the agent information about its own position, enabling it to reason about turnover costs when deciding whether to rebalance. Without this, the agent has no way to differentiate between holding a position (zero TC) and initiating it (positive TC).

Feature values are clipped to [−10, 10] at prediction time to handle outliers from high-volatility periods without breaking the softmax projection.

---

## 3. Walk-Forward Backtest Results

All numbers are sourced from `backtesting/backtest_results.json` — 30 windows, 3,780 total test days.

| Metric | RL-PPO Agent | Markowitz (MVO) | Equal-Weight (1/N) |
|---|---|---|---|
| **Sharpe Ratio** | **1.47** | 0.77 | 1.24 |
| **Sortino Ratio** | **1.41** | 0.74 | 1.18 |
| **Calmar Ratio** | **0.61** | 0.21 | 0.41 |
| **CAGR** | **20.0%** | 8.7% | 16.9% |
| **Max Drawdown** | **−32.8%** | −42.3% | −41.5% |
| **VaR 95% (daily)** | −1.24% | −1.13% | −1.30% |
| **CVaR 95% (daily)** | **−1.80%** | −1.61% | −1.88% |
| **t-statistic vs EW** | **11.10** | — | — |
| **p-value vs EW** | **4.3 × 10⁻²²** | — | — |

The alpha over the equal-weight benchmark is statistically significant at p ≪ 0.001. Markowitz underperforms equal-weight on CAGR and drawdown in this dataset, consistent with the known sensitivity of MVO to estimation error in expected returns.

---

## 4. Data Flow: Inference Request

```
Client
  │  POST /predict  {"tickers": ["RELIANCE.NS", "TCS.NS", ...]}
  │  Header: X-API-Key
  ▼
FastAPI (serving/main.py)
  │  1. verify_api_key() — compares header to API_KEY env var
  │  2. registry.is_loaded check → 503 if model not ready
  │  3. predict_weights(tickers)
  ▼
predictor.py
  │  1. _load_features() — reads Parquet feature files (cached in memory)
  │  2. _load_model()    — loads PPO/SAC .zip from saved_models/ (cached)
  │  3. Build obs vector: flatten feature_matrix[latest_t] + equal_weights (440 dims)
  │  4. model.predict(obs, deterministic=True) → raw logits (40 dims)
  │  5. Extract subset for requested tickers → softmax → weights
  ▼
FastAPI
  │  record_request(latency_ms)  — Prometheus counter + histogram
  │  return PredictResponse
  ▼
Client  {"weights": {...}, "weights_sum": 1.0, "latency_ms": ..., "algo": "PPO"}
```

Target latency: p99 < 200 ms. Achieved via in-memory feature caching after first request.

---

## 5. Data Flow: Nightly Retraining

```
18:00 IST (Mon–Fri)
  │
Airflow Scheduler
  │  triggers retrain_portfolio_agent DAG
  ▼
task_fetch_data
  │  POST /retrain/fetch-data → API container
  │  API: removes Parquet files older than 24h, calls yfinance for all 40 tickers
  │  Result: fresh raw Parquet files in data/raw/
  ▼
task_engineer_features
  │  POST /retrain/engineer-features → API container
  │  API: recomputes all 10 features per ticker, writes data/processed/
  ▼
task_train_ppo ─────────────────────────────────────────── (parallel)
  │  POST /retrain/train?algo=ppo                          │
  │  API: instantiates PortfolioEnv, trains PPO 100k steps │
  │  Logs val_sharpe, val_sortino to MLflow               ─┤ task_train_sac
  │  Registers model artifact in MLflow registry           │  POST /retrain/train?algo=sac
  │                                                        │  (150k steps)
  ▼ ◄──────────────────────────────────────────────────────┘
task_evaluate_champion
  │  POST /retrain/evaluate-champion → API container
  │  API: fetches current champion Sharpe from MLflow registry
  │  Compares best new run Sharpe to champion × 1.05
  │  If challenger wins: set_registered_model_alias("PortfolioAgent", "champion", v)
  ▼
task_notify_result
  │  Logs promotion result (champion retained / new version promoted)
  │
  ▼ (if promoted)
model_loader.py background thread (polls every 300s)
  │  Detects new champion version in registry
  │  Hot-reloads policy under RLock — zero downtime
```

---

## 6. Monitoring and Drift Detection

Prometheus scrapes the `/metrics` endpoint on the API container every 15 seconds. Four metrics are exported:

| Metric | Type | Description |
|---|---|---|
| `portfolio_requests_total` | Counter | Total prediction requests, labelled by status |
| `portfolio_request_latency_ms` | Histogram | Request latency; buckets at 10/25/50/100/200/500/1000 ms |
| `portfolio_return_daily` | Gauge | Most recent daily return recorded via `/predict` |
| `portfolio_rolling_sharpe_30d` | Gauge | Rolling 30-day annualised Sharpe, updated on each request |

The rolling Sharpe gauge is the primary drift signal. It is computed from a 30-element deque of recent daily returns. When it drops below 0.5, Grafana fires an alert to the webhook server (`alert_trigger.py`), which triggers the Airflow DAG via `airflow dags trigger`.

The alert threshold of 0.5 was chosen to be approximately one standard deviation below the backtested Sharpe of 1.47, giving a meaningful signal of regime change while avoiding false positives from short-term noise.

---

## 7. Failure Modes and Mitigations

| Failure | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MLflow registry unreachable at startup | Medium | API starts without a loaded model; `/ready` returns 503 | Background loading thread retries silently; `/predict` returns 503 with a clear message until model is available |
| Champion model file corrupted or missing | Low | `_load_model()` raises `FileNotFoundError` | Fallback: tries both `ppo_agent.zip` and `sac_agent.zip`; returns 503 with descriptive error if neither exists |
| RL agent returns degenerate weights (all in one asset) | Low | Portfolio overconcentrated | Softmax projection guarantees all-positive weights; 40% max-weight cap in Markowitz does not apply to RL (mitigated by transaction cost penalty in reward) |
| Feature data unavailable for a ticker | Medium | Prediction skips that ticker | `predict_weights()` returns `skipped_tickers` in response; requires ≥ 2 valid tickers or raises 422 |
| Airflow DAG silently fails | Medium | Model not retrained; no drift correction | `email_on_failure: False` is a current limitation — should be set to `True` with a valid SMTP config for production |
| cvxpy solver failure (Markowitz) | Low | Walk-forward window produces equal-weight fallback | `markowitz_weights()` catches all solver exceptions and returns `np.ones(N)/N` with a warning log |
| Training run does not beat champion by 5% threshold | Expected | No model churn; current champion retained | Logged explicitly; Airflow task succeeds with `"Current champion retained"` message |
| Docker image too large for free-tier deployment | Addressed | Deployment blocked | CPU-only torch wheel (`+cpu` index), multi-stage Dockerfile, per-service requirements files |

---

## 8. Repository Structure

```
portfolio-optimizer/
├── data/
│   ├── config.py                 ← tickers, dates, paths (loaded from .env)
│   └── ingestion/
│       ├── fetch_data.py         ← yfinance OHLCV download
│       ├── feature_engineering.py← 10 features per ticker → Parquet
│       └── macro_features.py     ← VIX, US 10Y yield, sector ETFs
├── models/
│   ├── forecasting/
│   │   ├── lstm_model.py         ← ReturnLSTM (PyTorch nn.Module)
│   │   ├── dataset.py            ← ReturnSequenceDataset, make_dataloaders
│   │   └── train_forecaster.py   ← training loop + MLflow logging
│   ├── rl_agent/
│   │   ├── portfolio_env.py      ← PortfolioEnv (Gymnasium)
│   │   ├── reward.py             ← step_reward, sortino_reward
│   │   ├── train_agent.py        ← PPO/SAC training + champion-challenger
│   │   └── saved_models/         ← ppo_agent.zip, sac_agent.zip
│   └── classical/
│       ├── markowitz.py          ← cvxpy MVO + Ledoit-Wolf
│       └── black_litterman.py    ← Black-Litterman model
├── backtesting/
│   ├── walk_forward.py           ← 30-window rolling backtest engine
│   ├── metrics.py                ← Sharpe, Sortino, CVaR, drawdown, t-test
│   ├── report.py                 ← HTML report generation (Plotly)
│   └── backtest_results.json     ← pre-computed results (read by dashboard)
├── serving/
│   ├── main.py                   ← FastAPI app, lifespan, middleware, auth
│   ├── predictor.py              ← inference logic, feature loading, softmax
│   ├── model_loader.py           ← ModelRegistry, hot-reload background thread
│   ├── metrics.py                ← Prometheus counters, gauges, histograms
│   ├── retrain_router.py         ← /retrain/* endpoints (called by Airflow)
│   └── schemas.py                ← Pydantic request/response models
├── dashboard/
│   ├── app.py                    ← Streamlit entry point
│   ├── data_utils.py             ← cached data loading, API client
│   └── pages/
│       ├── 1_Performance.py      ← equity curves, drawdown, rolling Sharpe
│       ├── 2_Allocation.py       ← live weights from /predict
│       ├── 3_Comparison.py       ← RL vs Markowitz vs EW table + charts
│       └── 4_Frontier.py         ← Markowitz efficient frontier (Monte Carlo)
├── mlops/
│   ├── airflow/dags/retrain_dag.py ← pure HTTP orchestrator (no ML code)
│   └── monitoring/
│       ├── prometheus.yml        ← scrape config (15s interval)
│       ├── grafana_dashboard.json← pre-built dashboard JSON
│       └── alert_trigger.py      ← Flask webhook: Grafana → Airflow trigger
├── tests/                        ← pytest suite (unit + integration, mocked)
├── docker-compose.yml            ← 8-service local stack
├── .github/workflows/
│   ├── ci.yml                    ← ruff lint → pytest → Docker build → smoke test
│   └── cd.yml                    ← push to Docker Hub on merge to main
└── requirements/
    ├── api.txt                   ← FastAPI service dependencies
    ├── dashboard.txt             ← Streamlit service dependencies
    └── airflow.txt               ← Airflow orchestrator dependencies (minimal)
```

---

## 9. Known Limitations

**No live paper trading.** The system predicts weights but does not execute trades. Integration with the Alpaca Markets API for live paper trading is stubbed in `fetch_data.py` but not wired to the serving layer.

**SQLite for MLflow backend.** The current `mlflow.db` is a single-file SQLite database, suitable for a solo developer setup. A production deployment would replace this with PostgreSQL and S3 artifact storage.

**SequentialExecutor in Airflow.** PPO and SAC training are parallelised within the API container but Airflow uses SequentialExecutor, meaning no DAG-level parallelism. Upgrading to CeleryExecutor with a Redis broker would allow independent worker scaling.

**No position sizing or risk limits.** The system outputs percentage weights, not position sizes. It does not enforce notional limits, sector constraints, or regulatory position limits.

**Cold start latency.** The free-tier deployment on Render sleeps after 15 minutes of inactivity, causing a ~30-second cold start on the first request. For interview demos, trigger a `/health` call before presenting to warm the container.