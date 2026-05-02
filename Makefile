# Run these commands from the project root with: make <target>
# On Windows: install make via  winget install GnuWin32.Make
# Or just copy-paste the commands directly from the recipes below.

.PHONY: install lint test data clean

install:
	pip install -e ".[dev]"

lint:
	ruff check . --fix
	mypy data/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=data --cov-report=term-missing

data:
	python -m data.ingestion.fetch_data
	python -m data.ingestion.feature_engineering
	python -m data.ingestion.macro_features

clean:
	find . -type d -name __pycache__ -exec rmdir /s /q {} + 2>nul || true
	find . -name "*.pyc" -delete 2>nul || true