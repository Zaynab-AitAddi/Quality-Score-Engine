# Trevo Listing Quality Score Engine

**Prepared by Zaynab AITADDI**

**Trevo internship project · Project A: Quality Score Engine**

This repository contains the Python batch-scoring part of the Trevo Ranking Lab internship brief. It produces a content-only quality score for vacation-rental listings during the booking cold-start period. Each result includes a 0–100 score, component scores, and an explanation intended to help identify listing improvements.

This is a standalone prototype using the supplied anonymized data package. It does not connect to Trevo production systems. It implements **Project A only**; the separate TypeScript ranking library, synthetic event generator, golden-query harness, replay tool, and demo UI described as Project B are not in this repository.

## Project Status Against the Brief

| Brief deliverable                                                                     | Current repository status                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python batch scorer and explainable feature table                                     | Implemented. Run from the repository root; see [CONTRACT.md](CONTRACT.md).                                                                                                                                                                                                                                                                                                                                                                                            |
| Technical photo checks: blur, exposure, resolution, near-duplicates                   | Implemented with OpenCV and perceptual hashing. The configured gates still need labeled evaluation evidence.                                                                                                                                                                                                                                                                                                                                                          |
| Room coverage using zero-shot vision                                                  | OpenCLIP zero-shot classification is implemented and runs on CPU. The current fallback is filename-based, not the proposed classifier trained on cached embeddings.                                                                                                                                                                                                                                                                                                   |
| Amenity visual cross-check                                                            | OpenCLIP support is implemented. Set `QUALITY_ENGINE_USE_CLIP=1` to enable both room and amenity vision. Without it, amenity scoring is neutral.                                                                                                                                                                                                                                                                                                                      |
| Aesthetic predictor calibrated on about 200 photos, independently rated by two people | **Not complete.** The current aesthetic score is a colorfulness/contrast/sharpness heuristic. A linear calibration utility exists, but ratings, calibration results, and held-out evaluation evidence are not included.                                                                                                                                                                                                                                               |
| Room accuracy at least 85%; blur precision at least 90%                               | Evaluation utilities exist, but no labeled evaluation sets or results are included. The code reports metrics; passing these targets has not been demonstrated.                                                                                                                                                                                                                                                                                                        |
| Human ranking comparison on 30 listings rated by two people                           | **Not implemented/evidenced.**                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Error analysis for Moroccan interiors                                                 | **Not implemented/evidenced.**                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Quality signal contributes 40 points as described in the brief                        | **Configuration discrepancy to resolve:** current weights assign 20 technical + 15 aesthetic + 10 room coverage + 10 amenity check = **55 points** to photo/image-related components. The brief says photo signals carry 40 points. This README does not change weights because the brief excerpt does not specify an unambiguous revised allocation. Confirm the intended component weights with the project supervisor before treating the current config as final. |

The current component weights are in [config/weights.yaml](config/weights.yaml); they sum to 100. The engine validates that total and the feature-table validator checks generated output. Do not describe this prototype as having met the internship evaluation targets until the labeled sets have been run and the results recorded.

## What the Scorer Produces

For each eligible listing, the engine writes one row to a Parquet feature table containing:

- The overall `quality_score` from 0 to 100.
- Six component scores: photo technical, photo aesthetic, room coverage, amenity check, text quality, and completeness.
- Photo status, schema/configuration versions, and a JSON explanation with strengths, issues, suggestions, and flagged photos.

Listings whose `vertical` is blank or `stay` are scored. The score is a weighted content signal; it does not use bookings, impressions, or host behavior.

## Run on Windows

Open PowerShell in the repository root, the folder containing `pyproject.toml` and `README.md`.

### Install

Python 3.10 or newer is required. From the repository root:

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The dependency set includes PyTorch and OpenCLIP and may take time and disk space to install. It is configured for CPU use; a GPU is not required.

### Check data

The scorer expects these inputs under `data/`:

- `data/listings.csv` with at least a `listing_id` column.
- `data/photos.csv` with `entity_id` and `file` columns.
- Photo files at paths referenced by `photos.csv`, normally under `data/photos/`.

The data package and photos are private project data and are not tracked by Git. If Trevo provided `ranking-data.zip` in the repository root, extract it locally:

```powershell
Expand-Archive -LiteralPath .\ranking-data.zip -DestinationPath .\data -Force
```

Check that the CSV files are directly inside `data/`. If the archive has a different layout, arrange the files to match the paths above. See [data/README.md](data/README.md).

### Score five listings

For the full Project A visual baseline, enable OpenCLIP for room and amenity inference:

```powershell
$env:QUALITY_ENGINE_USE_CLIP = "1"
python -m quality_engine.run --limit 5 --output .\output\sample.parquet
```

The command prints a ranked report and writes `output/sample.parquet`. OpenCLIP downloads pretrained weights the first time they are needed; later runs use its local cache. The Hugging Face unauthenticated-request warning concerns download rate limits, not listing data. The scorer runs inference locally and does not send listing photos to the model hub.

To score the entire input dataset, omit `--limit`:

```powershell
python -m quality_engine.run --output .\output\features.parquet
```

The default output path is `output/features.parquet`. Parent directories are created automatically. For offline or lightweight room fallback testing, disable CLIP:

```powershell
$env:QUALITY_ENGINE_USE_CLIP = "0"
python -m quality_engine.run --limit 5 --output .\output\sample-fallback.parquet
```

The fallback identifies room types from photo filenames; it is not equivalent to visual classification. With CLIP disabled, amenity scoring uses a neutral score. To see all CLI options, run `python -m quality_engine.run --help`.

### Validate output and run tests

```powershell
python scripts\validate_feature_table.py .\output\features.parquet
python -m pytest -q
python -m ruff check src tests scripts
```

The validator checks required columns, unique/non-empty listing IDs, score ranges and finiteness, explanation JSON, and the configured weight total. The repository test suite checks code behavior; it does not replace labeled model evaluation.

## Components and Configuration

Current configuration lives in `config/` and is loaded from YAML.

| Component       | Current weight | Current method                                                                                                                                |
| --------------- | -------------: | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Photo technical |            20% | OpenCV blur, exposure, resolution and image metrics; perceptual hashes for near-duplicates.                                                   |
| Photo aesthetic |            15% | Heuristic based on colorfulness, contrast, and sharpness; optional linear calibration file. This is not a pretrained aesthetic predictor yet. |
| Room coverage   |            10% | OpenCLIP zero-shot room classification when enabled; filename fallback if unavailable.                                                        |
| Amenity check   |            10% | OpenCLIP visual comparison when enabled; neutral score when visual detection is disabled/unavailable.                                         |
| Text quality    |            15% | Heuristic language-marker checks, title length, and description length bands. Language detection is not a full language-identification model. |
| Completeness    |            30% | Listing fields, amenities relative to capacity, pricing, and calendar/availability fields.                                                    |

Weights must sum to 100. Thresholds are in `config/thresholds.yaml`, room and amenity prompts in `config/room_labels.yaml`, and model settings in `config/models.yaml`. Changing weight values or output semantics should be reviewed against the brief and versioned according to [CONTRACT.md](CONTRACT.md).

## Understanding the Console Report

- `SCORE` is the weighted overall result. `LEVEL` groups it as `EXCELLENT` (80+), `GOOD` (65–79.9), `NEEDS WORK` (50–64.9), or `LOW` (below 50). Rows are sorted from highest score to lowest.
- Component abbreviations are `TECH`, `AESTH`, `ROOMS`, `AMEN`, `TEXT`, and `COMPLETE`. Each component is a score from 0 to 100 before weighting.
- A photo flag gives the technical issue, then a room guess and classifier confidence, then pixel dimensions and megapixels. For example, `Below configured resolution target; room guess: living room (46% classifier confidence); 1440 x 648 px (0.93 MP)` means the image is below the configured resolution threshold; “living room” is the classifier's top room category. The percentage expresses classifier confidence, not image quality and not a calibrated probability of correctness.
- `unknown room` means a recognized room label did not meet the configured confidence threshold; when possible, the top guess is shown in parentheses. A blur flag is a separate technical signal.
- Current resolution thresholds are 0.3 MP minimum and 2 MP for the configured good target. The current report flags scores below 40 on its normalized resolution scale; this is a tunable project rule, not a universal definition of unusable image quality.
- A neutral amenity score means visual amenity detection was not available, not that the listing's declared amenities were confirmed or disproved.

## Evaluation Workflow

Evaluation requires locally prepared, labeled CSVs. Keep labels and photos private and do not send Trevo data to public repositories, public model hubs, or third-party labeling services. Follow the internship's data-handling rules and have ratings produced within the approved private workflow.

Room labels CSV format:

```csv
photo_file,label
photos/adv_0001/photo1.jpg,bedroom
```

Blur labels CSV format uses the same `photo_file,label` columns, with labels such as `blurry` and `sharp`. Example commands:

```powershell
python scripts\evaluate_room_classifier.py .\private-eval\room_labels.csv --photos-root .\data
python scripts\evaluate_quality_gates.py --room-labels .\private-eval\room_labels.csv --blur-labels .\private-eval\blur_labels.csv --photos-root .\data
```

The gate utility reports room accuracy and blur precision/recall; compare those results with the brief's targets of at least 85% room accuracy and 90% blur precision. It does not by itself create the held-out datasets or establish that targets have passed.

For aesthetic calibration, collect the planned human ratings first. The calibration CSV needs `photo_file` and either `avg_rating` or both `rater1` and `rater2` columns. Then run:

```powershell
python scripts\calibrate_aesthetic_model.py --csv .\private-eval\aesthetic_ratings.csv
```

The default calibration artifact is `artifacts/aesthetic_calibrator.yaml`. Do not claim calibration or independent two-rater agreement until those ratings have been collected and reviewed. Human ranking correlation for 30 listings and Moroccan-interior error analysis remain separate evaluation tasks.

## Data Handling

The internship brief requires private repositories and local handling of the anonymized listing/photo package. Do not make this repository public, commit raw data or photos, attempt re-identification, or upload listing content to public model hubs or third-party labeling services. OpenCLIP downloads pretrained model weights from the Hugging Face Hub; inference is local. Generated feature tables, evaluation labels, calibrated artifacts, and model artifacts derived from Trevo data must also remain private and be handled under the project's retention/deletion rules.

## Repository Layout

```text
config/                  Weights, thresholds, labels/prompts, model settings
data/                    Private local input CSVs and photos; not tracked by Git
output/                  Generated feature tables; not tracked by Git
scripts/                 Scoring support, profiling, calibration, evaluation, validation
src/quality_engine/      Python Project A implementation
tests/                   Automated code tests
CONTRACT.md              Project A feature-table contract
pyproject.toml           Package and development-tool configuration
```

## Scope and Remaining Work

This repository is intended to deliver the Project A quality feature table that Project B can consume. Project B's ranking pipeline and its behavioral features are a separate project and are not implemented here. Before presenting Project A as matching the brief, the priority items are:

1. Confirm and resolve the 40-point photo-weight requirement against the current 55-point image-related weighting; update the config and versioned contract only after the intended allocation is agreed.
2. Replace or explicitly approve the heuristic aesthetic baseline, then collect the planned roughly 200 photos with independent ratings from two raters, calibrate, and report held-out results.
3. Build held-out room and blur label sets, report the 85% and 90% gates, and implement the cached-embedding classifier fallback if room accuracy misses its target.
4. Complete the 30-listing/two-rater ranking sanity check and document Moroccan-interior error analysis.
5. Run and validate the full dataset to demonstrate that every eligible listing has one valid output row.

Passing unit tests, lint, and feature-table validation verifies software behavior and table structure; it does not prove these model and human-evaluation requirements have been met.
