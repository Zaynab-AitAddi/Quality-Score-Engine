# Running the Quality Score Engine

Quick steps to run locally and next steps to make the project production-ready.

1. Create virtual environment and install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

2. Run tests

```powershell
python -m pytest -q
```

3. Profile dataset (optional)

```powershell
python scripts/profile_data.py --data data
```

4. Run a small batch scoring (5 listings)

```powershell
#$env:PYTHONPATH="src"
.venv\Scripts\python -m quality_engine.run --limit 5 --output output/features.parquet
```

5. Inspect generated features

```powershell
$env:PYTHONPATH="src"; .venv\Scripts\python scripts/show_features.py
```

Improvement checklist to make this "perfect":

- Calibration: collect ~200 human-rated photos, run `scripts/create_calibration_template.py` to generate the CSV template, then implement calibration mapping for the aesthetic model.
- Room classifier: gather labeled room photos, run `scripts/evaluate_room_classifier.py` to measure baseline accuracy; if <85% train a small classifier on cached embeddings.
- Blur precision: annotate a small validation set to measure blur precision ≥0.90 and tune thresholds in `config/thresholds.yaml`.
- Add integration tests and sample fixtures (small set of photos + metadata) under `tests/fixtures/` so CI can validate end-to-end runs.
- Add pre-commit hooks (ruff, black if desired) and a `Makefile` for common tasks.
- Create a small demo notebook showing how scores map to human judgments and examples of explanations.
- Package & release: add an sdist/wheel pipeline and changelog for releases.

If you want, I can implement any of the checklist items next — tell me which one to focus on.
