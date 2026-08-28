PYTHON=.venv\Scripts\python
PIP=.venv\Scripts\pip

.PHONY: venv install test lint run

venv:
	python -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

test:
	$(PYTHON) -m pytest -q

lint:
	$(PIP) install ruff
	ruff check src tests

run:
	$env:PYTHONPATH="src"; $(PYTHON) -m quality_engine.run --limit 5 --output output/features.parquet
