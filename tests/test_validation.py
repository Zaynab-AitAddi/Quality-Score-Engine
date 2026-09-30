"""Test feature-table contracts, labeled quality gates, and CLIP fallback behavior."""

import pandas as pd

from quality_engine.features.image import room_coverage
from quality_engine.features.image.room_coverage import classify_photo
from quality_engine.validation import (
    evaluate_blur_precision,
    evaluate_room_accuracy,
    validate_feature_table,
)


def test_validate_feature_table_contract():
    df = pd.DataFrame(
        [{
            "listing_id": "lst_0001",
            "quality_score": 80.0,
            "photo_technical": 75.0,
            "photo_aesthetic": 70.0,
            "room_coverage": 85.0,
            "amenity_check": 80.0,
            "text_quality": 60.0,
            "completeness": 90.0,
            "photo_status": "ok",
            "version": "quality-v1",
            "quality_engine_version": "1.0.0",
            "config_version": "quality-config-v1",
            "explanation_json": '{"positive": ["ok"], "issues": [], "suggestions": []}',
        }]
    )
    result = validate_feature_table(df)
    assert result["passes"] is True
    assert result["row_count"] == 1

    df.loc[0, "quality_score"] = float("nan")
    result = validate_feature_table(df)
    assert result["non_finite_quality_scores"] == 1
    assert result["passes"] is False

    df.loc[0, "quality_score"] = 80.0
    df.loc[0, "photo_technical"] = float("inf")
    result = validate_feature_table(df)
    assert result["non_finite_component_scores"] == 1
    assert result["passes"] is False


def test_feature_table_rejects_duplicate_ids_and_invalid_explanations():
    row = {
        "listing_id": "lst_0001",
        "quality_score": 80.0,
        "photo_technical": 75.0,
        "photo_aesthetic": 70.0,
        "room_coverage": 85.0,
        "amenity_check": 80.0,
        "text_quality": 60.0,
        "completeness": 90.0,
        "photo_status": "ok",
        "version": "quality-v1",
        "quality_engine_version": "1.0.0",
        "config_version": "quality-config-v1",
        "explanation_json": '{"positive": [], "issues": [], "suggestions": []}',
    }
    df = pd.DataFrame([row, {**row, "explanation_json": "not json"}])

    result = validate_feature_table(df)

    assert result["duplicate_listing_ids"] == 1
    assert result["invalid_explanation_json"] == 1
    assert result["passes"] is False

    df = pd.DataFrame([{
        **row,
        "explanation_json": '{"positive": [NaN], "issues": [], "suggestions": []}',
    }])
    result = validate_feature_table(df)
    assert result["invalid_explanation_json"] == 1
    assert result["passes"] is False


def test_room_accuracy_evaluation():
    df = pd.DataFrame([
        {"photo_file": "a.jpg", "label": "bedroom"},
        {"photo_file": "b.jpg", "label": "living"},
    ])

    def fake_predictor(path):
        return type("Prediction", (), {"label": "bedroom" if "a" in str(path) else "living"})()

    result = evaluate_room_accuracy(df, predictor=fake_predictor)
    assert result["row_count"] == 2
    assert result["accuracy"] == 1.0


def test_blur_precision_evaluation():
    df = pd.DataFrame([
        {"photo_file": "a.jpg", "label": "blurry"},
        {"photo_file": "b.jpg", "label": "sharp"},
        {"photo_file": "c.jpg", "label": "blurry"},
    ])

    def fake_detector(path):
        class Metrics:
            is_blurry = "a" in str(path) or "c" in str(path)
        return Metrics()

    result = evaluate_blur_precision(df, detector=fake_detector)
    assert result["row_count"] == 3
    assert result["precision"] == 1.0


def test_clip_disabled_uses_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("QUALITY_ENGINE_USE_CLIP", "0")
    img = tmp_path / "dummy.jpg"
    img.write_bytes(b"not-a-real-image")

    pred = classify_photo(img)
    assert pred.label in {"other", "exterior"}
    assert pred.confidence >= 0.0


def test_room_coverage_rejects_known_label_below_confidence_threshold(monkeypatch, tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.touch()
    monkeypatch.setattr(
        room_coverage,
        "classify_photo",
        lambda path: room_coverage.RoomPrediction(path, "bedroom", 0.2),
    )

    result = room_coverage.score_room_coverage([photo])

    assert "bedroom" not in result.covered
    assert "bedroom" in result.missing
    assert len(result.predictions) == 1
    assert result.predictions[0].confidence == 0.2
