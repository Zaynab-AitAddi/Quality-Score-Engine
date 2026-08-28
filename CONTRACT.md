# Feature table contract — Quality Score Engine v1

## Purpose

One row per listing. This table is the interface between the quality engine and the ranking system.

## Required columns

| Column | Type | Description |
|--------|------|-------------|
| `listing_id` | string | Unique listing identifier |
| `quality_score` | float | Weighted 0–100 score |
| `photo_technical` | float | Component score 0–100 |
| `photo_aesthetic` | float | Component score 0–100 |
| `room_coverage` | float | Component score 0–100 |
| `amenity_check` | float | Component score 0–100 |
| `text_quality` | float | Component score 0–100 |
| `completeness` | float | Component score 0–100 |
| `photo_status` | string | `ok`, `no_photos`, etc. |
| `version` | string | Output schema version |
| `quality_engine_version` | string | Engine release |
| `config_version` | string | Config snapshot |
| `explanation_json` | string | JSON with positive/issues/suggestions |

## Weight configuration

Component weights are defined in `config/weights.yaml` and **must sum to 100**.

## Versioning

Any change to column names, semantics, or weight definitions requires a contract version bump.
