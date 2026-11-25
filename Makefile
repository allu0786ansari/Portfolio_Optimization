.PHONY: help install lint typecheck test leakage leakage-real phase0-smoke phase0-baseline data train backtest serve dashboard up down logs clean

UV ?= uv
EXTRAS = --extra dev --extra rl --extra ml --extra serving --extra data
export MLFLOW_DISABLE_AGENT_HINT=1

help:            ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## create .venv from uv.lock (all extras)
	$(UV) sync $(EXTRAS)
	$(UV) run pre-commit install

lint:            ## ruff
	$(UV) run ruff check .

typecheck:       ## mypy on src/pfo
	$(UV) run mypy

test:            ## full test suite (synthetic data, no downloads)
	$(UV) run pytest tests --cov

leakage:         ## Phase 0 exit gate on synthetic data
	$(UV) run pytest tests/leakage -m leakage -v
	$(UV) run pfo-leakage-scan --source synthetic

leakage-real:    ## Phase 0 exit gate on downloaded data (run `make data` first)
	$(UV) run pfo-leakage-scan --source v1 --universe both --out reports/phase0/leakage_scan_real.json

phase0-smoke:    ## corrected-baseline pipeline on synthetic data (~3 min)
	$(UV) run pfo-v1-baseline --source synthetic --rl-timesteps 20000 --out reports/phase0/synthetic

phase0-baseline: ## corrected v1 baseline on real data (run `make data` first)
	$(UV) run pfo-v1-baseline --source v1 --universe both --out reports/phase0/real

data:            ## download prices + benchmarks, build v1 features and macro features
	$(UV) run python -m data.ingestion.fetch_data
	$(UV) run python -m data.ingestion.feature_engineering
	$(UV) run python -m data.ingestion.macro_features

train:           ## train PPO + SAC (needs MLflow at $$MLFLOW_TRACKING_URI)
	$(UV) run python -m models.rl_agent.train_agent --algo ppo
	$(UV) run python -m models.rl_agent.train_agent --algo sac

backtest: phase0-baseline

serve:
	$(UV) run uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(UV) run streamlit run dashboard/app.py

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
