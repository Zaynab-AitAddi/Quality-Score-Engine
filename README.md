# Listing Quality Score Engine

Content-only **0–100 quality score** for vacation rental listings, with explainable component breakdown. Built for cold-start ranking (no bookings required).

## Quick start

```bash
# 1. Extract data (if not already done)
tar -xf ranking-data.zip -C data

# 2. Create virtual environment and install
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# 3. Profile the dataset
python scripts/profile_data.py

# 4. Run quality scoring (test with 5 listings)
python -m quality_engine.run --limit 5

# 5. Full batch run
python -m quality_engine.run --output output/features.parquet
```

## Architecture

```
Listing + Photos → Feature extraction → Component scores → Weighted 0–100 → Feature table
```

Components (weights in `config/weights.yaml`):

- **Photo technical** (20) — blur, exposure, resolution, duplicates
- **Photo aesthetic** (15) — pretrained/heuristic aesthetic signal
- **Room coverage** (10) — bedroom, bathroom, kitchen, etc.
- **Amenity check** (10) — declared vs visually detected
- **Text quality** (15) — languages, title, description length
- **Completeness** (30) — metadata, calendar, pricing

## Output

Parquet feature table (`output/features.parquet`) — see [CONTRACT.md](CONTRACT.md).

## Project layout

```
config/          YAML weights, thresholds, model settings
src/quality_engine/   Core engine modules
scripts/         Data profiling and utilities
tests/           Unit tests
data/            Extracted dataset (from zip)
output/          Generated feature tables
artifacts/       Cached embeddings and models
```

## Tests

```bash
pytest
```

## Next steps (from spec)

1. Human-rated aesthetic calibration (~200 photos)
2. Room classifier evaluation (≥85% accuracy gate)
3. Blur precision evaluation (≥0.90 gate)
4. Human ranking sanity check (30 listings × 2 raters)
5. Moroccan interior bias analysis
