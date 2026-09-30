"""Validation helpers for the quality-score feature contract and evaluation gates."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from quality_engine.config import validate_weights_sum_to_100
from quality_engine.features.image.room_coverage import classify_photo
from quality_engine.features.image.technical import analyze_photo

REQUIRED_FEATURE_COLUMNS = [
    "listing_id",
    "quality_score",
    "photo_technical",
    "photo_aesthetic",
    "room_coverage",
    "amenity_check",
    "text_quality",
    "completeness",
    "photo_status",
    "version",
    "quality_engine_version",
    "config_version",
    "explanation_json",
]


def _valid_explanation(value: object) -> bool:
    if not isinstance(value, str):
        return False

    def reject_non_finite(constant: str) -> None:
        raise ValueError(f"Invalid JSON numeric constant: {constant}")

    try:
        explanation = json.loads(value, parse_constant=reject_non_finite)
    except (json.JSONDecodeError, ValueError):
        return False
    required_sections = ("positive", "issues", "suggestions")
    return isinstance(explanation, dict) and all(
        isinstance(explanation.get(section), list) for section in required_sections
    )


def validate_feature_table(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a feature table against the Project A contract."""
    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    row_count = len(df)
    listing_ids = df["listing_id"].astype("string").str.strip()
    valid_listing_ids = listing_ids.notna() & listing_ids.ne("")
    listing_missing = int((~valid_listing_ids).sum())
    duplicate_listing_ids = int(
        listing_ids[valid_listing_ids].duplicated().sum())

    score_columns = [
        "quality_score",
        "photo_technical",
        "photo_aesthetic",
        "room_coverage",
        "amenity_check",
        "text_quality",
        "completeness",
    ]
    numeric_scores = {
        column: pd.to_numeric(df[column], errors="coerce") for column in score_columns
    }
    quality = numeric_scores["quality_score"]
    quality_invalid = int(((quality < 0) | (quality > 100)).sum())
    non_finite = int((~np.isfinite(quality.to_numpy())).sum()
                     ) if row_count else 0

    component_cols = [
        "photo_technical",
        "photo_aesthetic",
        "room_coverage",
        "amenity_check",
        "text_quality",
        "completeness",
    ]
    component_out_of_range = 0
    non_finite_components = 0
    for col in component_cols:
        component = numeric_scores[col]
        component_out_of_range += int(((component < 0)
                                      | (component > 100)).sum())
        non_finite_components += int(
            (~np.isfinite(component.to_numpy())).sum())

    invalid_explanations = int(
        (~df["explanation_json"].map(_valid_explanation)).sum())

    validate_weights_sum_to_100()

    return {
        "row_count": row_count,
        "missing_columns": missing,
        "listing_id_missing": listing_missing,
        "duplicate_listing_ids": duplicate_listing_ids,
        "quality_score_out_of_range": quality_invalid,
        "component_scores_out_of_range": component_out_of_range,
        "non_finite_component_scores": non_finite_components,
        "non_finite_quality_scores": non_finite,
        "invalid_explanation_json": invalid_explanations,
        "passes": row_count > 0
        and not missing
        and listing_missing == 0
        and duplicate_listing_ids == 0
        and quality_invalid == 0
        and non_finite == 0
        and component_out_of_range == 0
        and non_finite_components == 0
        and invalid_explanations == 0,
    }


def validate_feature_table_file(path: str | Path) -> dict[str, Any]:
    table_path = Path(path)
    df = pd.read_parquet(table_path)
    return validate_feature_table(df)


def _resolve_photo_path(photo_file: str, photos_root: str | Path | None = None) -> Path:
    path = Path(photo_file)
    if not path.is_absolute() and photos_root is not None:
        path = Path(photos_root) / path
    return path


def evaluate_room_accuracy(
    labeled_df: pd.DataFrame,
    *,
    photos_root: str | Path | None = None,
    predictor: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate room coverage accuracy for a labeled CSV of photo_file,label rows."""
    required = {"photo_file", "label"}
    if not required.issubset(labeled_df.columns):
        raise ValueError(
            "Labeled data must contain photo_file and label columns")

    predictor = predictor or classify_photo
    y_true: list[str] = []
    y_pred: list[str] = []

    for _, row in labeled_df.iterrows():
        path = _resolve_photo_path(str(row["photo_file"]), photos_root)
        prediction = predictor(path)
        predicted_label = getattr(prediction, "label", str(prediction))
        y_true.append(str(row["label"]).strip())
        y_pred.append(str(predicted_label).strip())

    accuracy = float(accuracy_score(y_true, y_pred))
    report = {
        "accuracy": round(accuracy, 4),
        "row_count": len(labeled_df),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred))).tolist(),
        "labels": sorted(set(y_true) | set(y_pred)),
    }
    return report


def evaluate_blur_precision(
    labeled_df: pd.DataFrame,
    *,
    photos_root: str | Path | None = None,
    detector: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate blur precision: positive class is blurry."""
    required = {"photo_file", "label"}
    if not required.issubset(labeled_df.columns):
        raise ValueError(
            "Labeled data must contain photo_file and label columns")

    detector = detector or analyze_photo
    y_true: list[bool] = []
    y_pred: list[bool] = []

    for _, row in labeled_df.iterrows():
        raw_label = row["label"]
        truth = raw_label if isinstance(raw_label, (bool, np.bool_)) else str(
            raw_label).strip().lower() in {"blurry", "true", "1", "yes"}
        path = _resolve_photo_path(str(row["photo_file"]), photos_root)
        metrics = detector(path)
        prediction = getattr(metrics, "is_blurry", False)
        y_true.append(bool(truth))
        y_pred.append(bool(prediction))

    precision = precision_score(
        y_true, y_pred, pos_label=True, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=[False, True]).tolist()
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "row_count": len(labeled_df),
        "confusion_matrix": confusion,
        "labels": [False, True],
    }


def run_quality_gates(
    *,
    feature_table: str | Path | None = None,
    room_labels_csv: str | Path | None = None,
    blur_labels_csv: str | Path | None = None,
    photos_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the key Project A quality gates and return the summary."""
    summary: dict[str, Any] = {}

    if feature_table is not None:
        summary["feature_table"] = validate_feature_table_file(feature_table)

    if room_labels_csv is not None:
        room_df = pd.read_csv(room_labels_csv)
        summary["room_accuracy"] = evaluate_room_accuracy(
            room_df, photos_root=photos_root)

    if blur_labels_csv is not None:
        blur_df = pd.read_csv(blur_labels_csv)
        summary["blur_precision"] = evaluate_blur_precision(
            blur_df, photos_root=photos_root)

    return summary
