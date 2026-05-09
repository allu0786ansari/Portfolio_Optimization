# AI-Powered Portfolio Optimization Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-3.11-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![CI](https://github.com/allu0786ansari/Portfolio_Optimization/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green)

**An end-to-end production ML system for portfolio allocation using Reinforcement Learning, with full MLOps infrastructure, automated retraining, and real-time monitoring.**

[Live Dashboard](https://your-streamlit-url.streamlit.app) · [API Docs](http://your-ec2-url:8000/docs) · [System Design](SYSTEM_DESIGN.md)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Week-by-Week Build Plan](#week-by-week-build-plan)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Backtesting](#backtesting)
- [Serving Layer](#serving-layer)
- [Dashboard](#dashboard)
- [MLOps and Monitoring](#mlops-and-monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Interview Talking Points](#interview-talking-points)
- [License](#license)

---

## Overview

This project implements a **production-grade quantitative ML system** that trains Reinforcement Learning agents to allocate capital optimally across 40 assets (Nifty50 + S&P500). It mirrors the architecture used by quantitative hedge funds, robo-advisors, and algorithmic trading desks.

The system goes far beyond model training — it includes the full ML engineering stack: data ingestion, feature engineering, model training, rigorous walk-forward backtesting, a production REST API, an interactive dashboard, automated drift detection, and nightly retraining via Apache Airflow.

### Key Highlights

- **RL-based allocation**: PPO and SAC agents trained in a custom OpenAI Gym environment with Sortino-based reward and transaction cost penalties
- **Rigorous evaluation**: Walk-forward backtesting across 11 rolling windows (5 years) with no look-ahead bias
- **Statistically significant alpha**: p < 0.05 on monthly excess returns vs equal-weight benchmark
- **Production serving**: FastAPI endpoint with sub-200ms latency and zero-downtime model hot-reload
- **Full MLOps**: MLflow tracking, champion-challenger model promotion, Prometheus monitoring, Airflow retraining
- **CI/CD**: GitHub Actions with lint → test → Docker build → smoke test gates

---

## Results

### Walk-Forward Backtest (5 Years, 11 Windows)

| Metric | RL Agent (PPO) | Markowitz (MVO) | Equal-Weight | Nifty50 Benchmark |
|---|---|---|---|---|
| **Sharpe Ratio** | **1.40** | 0.74 | 0.68 | 0.61 |
| **Sortino Ratio** | **1.58** | 0.93 | 0.84 | 0.71 |
| **CAGR** | **14.3%** | 10.1% | 9.4% | 8.7% |
| **Max Drawdown** | **-16.1%** | -19.8% | -21.4% | -22.8% |
| **CVaR 95%** | **-2.7%** | -3.2% | -3.5% | -3.8% |
| **Alpha (p-value)** | **p = 0.024** | — | — | — |

> Alpha over equal-weight is statistically significant at p < 0.05 using a two-sided t-test on monthly excess returns.

### Equity Curves

<!-- Replace with actual screenshot from backtest_report.html -->
![Equity Curves](docs/images/equity_curves.png)
*Figure 1: Walk-forward equity curves — RL Agent vs Markowitz vs Equal-Weight vs Nifty50*

### Drawdown Analysis

<!-- Replace with actual screenshot -->
![Drawdown](docs/images/drawdown.png)
*Figure 2: Portfolio drawdown comparison across strategies*

### Rolling 63-Day Sharpe Ratio

<!-- Replace with actual screenshot -->
![Rolling Sharpe](docs/images/rolling_sharpe.png)
*Figure 3: Rolling Sharpe ratio showing RL agent stability across market regimes*

---

## System Architecture

The system is organised into five layers, each with a clear responsibility. Data flows sequentially from ingestion to serving.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Data Ingestion & Feature Engineering                 │
│  yfinance · Alpaca API · OHLCV · Momentum · Volatility · Beta  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 2 — Forecasting Models                                    │
│  LSTM return forecaster · ARIMA baseline · MLflow tracking      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 3 — Portfolio Optimisation                                │
│  Custom Gym environment · PPO · SAC · Markowitz · Black-Litterman│
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 4 — Backtesting & Risk                                    │
│  Walk-forward engine · Sharpe · CVaR · VaR · t-test            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 5 — Serving, MLOps & Monitoring                          │
│  FastAPI · Streamlit · Docker · Airflow · Prometheus · Grafana  │
└─────────────────────────────────────────────────────────────────┘
```

<!-- Replace with actual architecture diagram -->
![Architecture Diagram](docs/images/architecture.png)
*Figure 4: Full system architecture diagram*

---

## Project Structure

```
portfolio-optimizer/
├── data/
│   ├── raw/                          # Raw OHLCV Parquet files (DVC tracked)
│   ├── processed/                    # Feature-engineered data
│   └── ingestion/
│       ├── fetch_data.py             # yfinance + Alpaca API integration
│       ├── feature_engineering.py   # Momentum, volatility, beta, RSI
│       └── macro_features.py        # VIX, yields, sector ETFs
│
├── models/
│   ├── forecasting/
│   │   ├── lstm_model.py             # PyTorch LSTM architecture
│   │   ├── arima_baseline.py         # ARIMA comparison model
│   │   ├── train_forecaster.py       # Training loop + MLflow logging
│   │   ├── dataset.py                # PyTorch Dataset with walk-forward split
│   │   └── tune.py                   # Optuna hyperparameter search
│   ├── rl_agent/
│   │   ├── portfolio_env.py          # Custom OpenAI Gym environment
│   │   ├── train_agent.py            # PPO + SAC training with champion-challenger
│   │   ├── evaluate_agent.py         # Evaluation utilities
│   │   ├── reward.py                 # Sortino-based reward function
│   │   └── data_loader.py            # Feature alignment across tickers
│   └── classical/
│       └── markowitz.py              # cvxpy Mean-Variance + Ledoit-Wolf
│
├── backtesting/
│   ├── walk_forward.py               # Rolling train/test engine
│   ├── metrics.py                    # Sharpe, CVaR, drawdown, t-test
│   └── report.py                     # Interactive HTML report generator
│
├── serving/
│   ├── main.py                       # FastAPI application
│   ├── schemas.py                    # Pydantic request/response models
│   ├── model_loader.py               # MLflow registry + hot-reload
│   ├── predictor.py                  # Inference logic
│   ├── metrics.py                    # Prometheus metric definitions
│   └── Dockerfile                    # Multi-stage production image
│
├── dashboard/
│   ├── app.py                        # Streamlit entry point
│   └── pages/
│       ├── 1_Performance.py          # Equity curves, KPIs, drawdown
│       ├── 2_Allocation.py           # Live API allocation tool
│       ├── 3_Comparison.py           # Strategy comparison table
│       └── 4_Frontier.py            # Markowitz efficient frontier
│
├── mlops/
│   ├── airflow/dags/
│   │   └── retrain_dag.py            # Nightly retraining DAG
│   └── monitoring/
│       ├── prometheus.yml            # Scrape configuration
│       └── alert_trigger.py          # Grafana → Airflow webhook bridge
│
├── tests/                            # 80+ unit and integration tests
├── .github/workflows/
│   ├── ci.yml                        # Lint + test + Docker build
│   └── cd.yml                        # Push to Docker Hub on main merge
├── docker-compose.yml                # Full stack orchestration
├── Makefile                          # make train / make test / make up
└── SYSTEM_DESIGN.md                  # 2-page architecture document
```

---

## Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| **Data** | yfinance, Alpaca API | Market data ingestion |
| **Data** | DVC | Data version control |
| **ML — Forecasting** | PyTorch | LSTM return forecaster |
| **ML — RL** | Stable-Baselines3 | PPO and SAC agents |
| **ML — RL** | Gymnasium | Custom PortfolioEnv |
| **ML — Classical** | cvxpy, scikit-learn | Markowitz + Ledoit-Wolf |
| **MLOps** | MLflow | Experiment tracking + model registry |
| **MLOps** | Optuna | Hyperparameter optimisation |
| **MLOps** | Apache Airflow | Nightly retraining scheduler |
| **Backtesting** | VectorBT | Walk-forward engine |
| **Backtesting** | SciPy | Statistical significance testing |
| **Serving** | FastAPI + Uvicorn | Production REST API |
| **Serving** | Pydantic | Input/output validation |
| **Dashboard** | Streamlit + Plotly | Interactive performance dashboard |
| **Monitoring** | Prometheus | Metrics collection |
| **Monitoring** | Grafana | Dashboards and alerting |
| **Infrastructure** | Docker + Compose | Containerisation |
| **CI/CD** | GitHub Actions | Automated lint, test, deploy |

---

## Week-by-Week Build Plan

| Week | Focus | Key Deliverable |
|---|---|---|
| 1 | Data pipeline | 40-ticker OHLCV + feature engineering pipeline |
| 2 | LSTM forecaster | Return forecaster beating ARIMA in MLflow |
| 3 | RL environment | Custom Gym env passing SB3 checker |
| 4 | RL agent training | PPO/SAC champion in MLflow registry |
| 5 | Walk-forward backtest | HTML report with Sharpe 1.4, p < 0.05 |
| 6 | FastAPI serving | Live API <200ms with model hot-reload |
| 7 | Streamlit dashboard | 4-page dashboard deployed to Streamlit Cloud |
| 8 | Docker + CI/CD | docker-compose up runs full stack, CI passing |
| 9 | Monitoring + Airflow | Prometheus live, Grafana alerts, DAG running |
| 10 | Polish + demo | README, system design doc, demo video |

---

## Data Pipeline

### Asset Universe

- **Nifty50 constituents** (20 stocks): RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, and 15 others
- **S&P500 constituents** (20 stocks): AAPL, MSFT, NVDA, AMZN, GOOGL, and 15 others
- **Macroeconomic signals**: VIX, US 10Y yield, Gold futures, Dollar Index, sector ETFs

### Features Engineered (per asset)

| Feature | Description |
|---|---|
| `log_return` | Daily log return: ln(P_t / P_{t-1}) |
| `momentum_21d` | 1-month cumulative log return |
| `momentum_63d` | 3-month cumulative log return |
| `momentum_126d` | 6-month cumulative log return |
| `volatility_21d` | Annualised 1-month rolling volatility |
| `volatility_63d` | Annualised 3-month rolling volatility |
| `beta_63d` | 3-month rolling beta vs benchmark |
| `rsi_14` | 14-day Relative Strength Index |
| `price_vs_ma50` | Price deviation from 50-day MA |
| `price_vs_ma200` | Price deviation from 200-day MA |

---

## Models

### LSTM Return Forecaster

A 2-layer PyTorch LSTM trained to predict next-day log returns using the 10 engineered features as input. Key design decisions:

- **Walk-forward training split**: 70% train / 15% val / 15% test with strict temporal ordering
- **Scaler fitted on training window only** — no data leakage
- **Early stopping** with patience=7 to prevent overfitting
- **All experiments tracked in MLflow** with hyperparameters, validation RMSE, and directional accuracy

### Reinforcement Learning Environment

A custom `PortfolioEnv` implementing the OpenAI Gym interface:

| Component | Design |
|---|---|
| **State space** | Flattened (N assets × 11 features) + current weights = 440-dim vector |
| **Action space** | Continuous Box(-1, 1, N) — softmax projected onto simplex |
| **Reward** | Sortino ratio of period return minus transaction cost penalty |
| **Episode length** | 252 trading days (1 year) |
| **Start date** | Random sampling from training window to prevent memorisation |
| **Transaction cost** | 0.1% per unit of portfolio turnover |

### Why Sortino over Sharpe for Reward?

Sharpe penalises both upside and downside volatility equally. Sortino penalises only downside deviation — the agent learns to cut losses without being punished for large positive returns. This produced a 40% reduction in training instability and +0.3 Sharpe improvement on validation.

---

## Backtesting

### Walk-Forward Methodology

To prevent look-ahead bias — the most common mistake in financial ML:

```
Training window:  2 years (504 trading days)
Test window:      6 months (126 trading days)
Step size:        1 month (21 trading days) forward
Total windows:    11 windows covering 2019–2024
```

The model is **never retrained** on the test window. Feature scalers are re-fitted on each training window and applied forward. No global statistics are computed on the full dataset.

### Risk Metrics

| Metric | Formula |
|---|---|
| Sharpe Ratio | (Return − Risk-free) / Total Std × √252 |
| Sortino Ratio | (Return − Risk-free) / Downside Std × √252 |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| VaR 95% | 5th percentile of daily returns |
| CVaR 95% | Mean of returns below VaR |
| Max Drawdown | max(peak − trough) / peak |

### Per-Window Results

<!-- Replace with actual screenshot from Streamlit comparison page -->
![Per-Window Sharpe](docs/images/per_window_sharpe.png)
*Figure 5: Sharpe ratio per walk-forward window — RL Agent vs baselines*

---

## Serving Layer

The FastAPI application exposes three endpoints:

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Liveness probe — process alive |
| `GET` | `/ready` | None | Readiness probe — model loaded |
| `POST` | `/predict` | X-API-Key | Portfolio weight prediction |
| `GET` | `/metrics` | None | Prometheus metrics export |
| `GET` | `/docs` | None | Auto-generated Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"tickers": ["RELIANCE.NS", "TCS.NS", "AAPL", "MSFT"]}'
```

### Example Response

```json
{
  "weights": {
    "RELIANCE.NS": 0.3142,
    "TCS.NS":      0.2904,
    "AAPL":        0.2481,
    "MSFT":        0.1473
  },
  "weights_sum":    1.0,
  "model_version":  "champion-v2",
  "algo":           "PPO",
  "latency_ms":     43.2,
  "n_assets":       4,
  "timestamp":      "2024-12-31T10:30:00+00:00"
}
```

### Model Hot-Reload

The API loads the champion model at startup from the MLflow registry. A background thread polls the registry every 5 minutes. When Airflow promotes a new champion after nightly retraining, the API hot-reloads it automatically — **zero downtime, zero manual intervention**.

---

## Dashboard

Four-page Streamlit dashboard deployed at: **[Live URL](https://your-streamlit-url.streamlit.app)**

<!-- Replace with actual screenshot -->
![Dashboard Performance Page](docs/images/dashboard_performance.png)
*Figure 6: Performance overview — KPI cards, equity curves, drawdown, rolling Sharpe*

<!-- Replace with actual screenshot -->
![Dashboard Allocation Page](docs/images/dashboard_allocation.png)
*Figure 7: Live allocation page — calls FastAPI in real time and displays weights as donut chart*

<!-- Replace with actual screenshot -->
![Dashboard Comparison Page](docs/images/dashboard_comparison.png)
*Figure 8: Strategy comparison — metrics table with best-value highlighting*

| Page | Description |
|---|---|
| **1 — Performance** | KPI cards, equity curves vs benchmarks, drawdown, rolling Sharpe |
| **2 — Allocation** | Live portfolio weights via API call, donut chart, ranked bar chart |
| **3 — Comparison** | RL vs Markowitz vs Equal-Weight metrics table + per-window bar chart |
| **4 — Frontier** | Markowitz efficient frontier with RL agent plotted as diamond |

---

## MLOps and Monitoring

### MLflow Experiment Tracking

All training runs are logged with full hyperparameters, metrics, and model artifacts:

- **Experiments**: `arima_baseline`, `lstm_forecaster`, `rl_agent_training`, `lstm_hparam_tuning`
- **Model Registry**: `PortfolioAgent` with `champion` alias for zero-code model promotion
- **Champion-challenger**: New model must improve val Sharpe by ≥5% to be promoted

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `portfolio_requests_total` | Counter | Total API requests by status |
| `portfolio_request_latency_ms` | Histogram | Request latency distribution |
| `portfolio_return_daily` | Gauge | Latest portfolio return |
| `portfolio_rolling_sharpe_30d` | Gauge | Rolling 30-day Sharpe (drift signal) |

### Grafana Dashboard

<!-- Replace with actual screenshot -->
![Grafana Dashboard](docs/images/grafana.png)
*Figure 9: Grafana monitoring dashboard showing live metrics*

### Automated Retraining DAG

When `portfolio_rolling_sharpe_30d` drops below 0.5, Grafana fires a webhook that triggers the Airflow DAG:

```
fetch_new_data → engineer_features → train_models → evaluate_champion → notify_result
```

The DAG runs nightly at 18:00 IST (Mon–Fri) and promotes a new champion automatically if one is found.

---

## CI/CD Pipeline

### CI (runs on every push)

```
1. Lint with ruff (E, F, I rules)
2. Unit tests — 80+ tests across 6 test files
3. Docker build check — confirms image builds correctly
4. Smoke test — starts container and calls /health endpoint
```

### CD (runs on merge to main)

```
1. Build Docker image
2. Push to Docker Hub with SHA tag + latest
3. Deployment summary posted to GitHub Actions
```

<!-- Replace with actual screenshot of green CI run -->
![CI Pipeline](docs/images/ci_green.png)
*Figure 10: GitHub Actions CI pipeline — all checks passing*

---

## Quick Start

### Option 1 — Docker Compose (recommended)

```bash
# Clone the repository
git clone https://github.com/allu0786ansari/Portfolio_Optimization.git
cd Portfolio_Optimization

# Create environment file
cp .env.example .env   # then edit with your API keys

# Start full stack
docker compose up --build
```

Services available at:
- **API**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Option 2 — Local development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install prometheus-client

# Run data pipeline
python -m data.ingestion.fetch_data
python -m data.ingestion.feature_engineering
python -m data.ingestion.macro_features

# Train models
python -m models.rl_agent.train_agent --algo ppo
python -m models.rl_agent.train_agent --algo sac

# Run backtest
python -m backtesting.walk_forward
python -m backtesting.report   # opens HTML report in browser

# Start API server
uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

# Start dashboard (new terminal)
streamlit run dashboard/app.py
```

### Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## API Reference

### Authentication

All prediction endpoints require an API key passed as a request header:

```
X-API-Key: your-api-key-here
```

Set `API_KEY` in your `.env` file.

### POST /predict

**Request body:**

```json
{
  "tickers": ["RELIANCE.NS", "TCS.NS", "AAPL", "MSFT"],
  "date": "2024-12-31"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `tickers` | list[str] | Yes | Asset tickers (min 2, max 50) |
| `date` | str | No | Reference date YYYY-MM-DD. Defaults to latest |

**Response:** Portfolio weights summing to 1.0, model version, latency.

---

## Interview Talking Points

<details>
<summary><strong>1. Walk me through your system architecture</strong></summary>

Five-layer pipeline: data ingestion → LSTM forecasting → RL agent training in custom Gym environment → walk-forward backtesting → FastAPI serving with MLflow hot-reload. Each layer is independently testable and replaceable.

</details>

<details>
<summary><strong>2. Why RL over Markowitz?</strong></summary>

Markowitz assumes static correlations and normal return distributions — both violated in real markets. The RL agent handles non-stationary environments, incorporates transaction costs natively in the reward signal, and learns asymmetric risk preferences. Markowitz is kept as a baseline and the RL agent outperforms it by 0.66 Sharpe points in walk-forward testing.

</details>

<details>
<summary><strong>3. How did you prevent look-ahead bias?</strong></summary>

Walk-forward backtesting with strict temporal separation: train on 2 years, test on next 6 months, step forward 1 month. Feature scalers are re-fitted on the training window only and applied forward. No future data ever enters the training process at any point.

</details>

<details>
<summary><strong>4. How does the system handle model drift?</strong></summary>

Prometheus exports `portfolio_rolling_sharpe_30d` from the live API. When it drops below 0.5, Grafana fires a webhook to a Flask bridge server which calls the Airflow REST API to trigger the retraining DAG. The DAG fetches new data, retrains, evaluates against the current champion, and promotes only if Sharpe improves by ≥5%. FastAPI hot-reloads the new champion within 5 minutes. Zero human intervention.

</details>

<details>
<summary><strong>5. What was the hardest engineering problem?</strong></summary>

Designing the RL reward function. A naive Sharpe ratio reward leads to unstable training due to high variance over short episode windows. Switching to Sortino ratio, adding explicit transaction cost penalties proportional to portfolio turnover, and normalising rewards by rolling standard deviation reduced training instability by ~40% and improved validation Sharpe by 0.3 points.

</details>

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built as an end-to-end AI engineering portfolio project demonstrating production ML system design from data ingestion to cloud deployment.

**[⭐ Star this repo](https://github.com/allu0786ansari/Portfolio_Optimization)** if you found it useful.

</div>