.PHONY: install lint test data train backtest serve dashboard up down logs clean

install:
	pip install -e ".[dev]"

lint:
	ruff check . --fix

test:
	pytest tests/ -v --tb=short

data:
	python -m data.ingestion.fetch_data
	python -m data.ingestion.feature_engineering
	python -m data.ingestion.macro_features

train:
	python -m models.rl_agent.train_agent --algo ppo
	python -m models.rl_agent.train_agent --algo sac

backtest:
	python -m backtesting.walk_forward
	python -m backtesting.report

serve:
	uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	docker system prune -f