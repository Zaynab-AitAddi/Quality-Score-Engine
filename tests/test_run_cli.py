"""Test CLI feature-row generation and bad-photo details in scoring explanations."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quality_engine.features.image import amenities
from quality_engine.features.image.room_coverage import RoomCoverageResult, RoomPrediction
from quality_engine.features.image.technical import PhotoTechnicalMetrics, TechnicalFeatureResult
from quality_engine.ingestion.loaders import load_listings, load_photos
from quality_engine.run import _print_console_report, build_feature_rows
from quality_engine.schemas.listing import ListingRecord, PhotoRecord
from quality_engine.scoring.weighted_score import build_explanation


def test_build_feature_rows_writes_expected_columns(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    listings = pd.DataFrame(
        [{
            "listing_id": "lst_0001",
            "vertical": "stay",
            "name": "Cozy apartment",
            "description": "Nice apartment with kitchen and pool.",
            "city": "Agadir",
            "country": "Morocco",
            "capacity": 4,
            "bedrooms": 2,
            "bathrooms": 2,
            "base_price": 500,
            "currency": "MAD",
            "instant_booking": True,
            "is_calendar_synced": True,
            "has_active_ical": False,
            "pricing_days_180": 60,
            "blocked_days_180": 10,
            "amenities": '["Wi-Fi", "Kitchen", "Pool"]',
            "photo_count": 0,
            "property_type": "apartment",
            "status": "APPROVED",
        }]
    )
    listings.to_csv(data_dir / "listings.csv", index=False)

    photos = pd.DataFrame(
        columns=["entity_id", "file", "seq", "is_cover", "width", "height", "bytes"])
    photos.to_csv(data_dir / "photos.csv", index=False)

    rows = build_feature_rows(data_dir)
    assert rows
    row = rows[0]
    assert row["listing_id"] == "lst_0001"
    assert "quality_score" in row
    assert "explanation_json" in row
    assert row["photo_status"] in {"ok", "no_photos"}


def test_build_explanation_reports_bad_photo_paths():
    explanation = build_explanation(
        technical_reasons=["1 blurry photo(s)"],
        aesthetic_reasons=[],
        room_reasons=[],
        amenity_reasons=[],
        text_reasons=[],
        completeness_reasons=[],
        undeclared_amenities=[],
        missing_rooms=[],
        blurry_count=1,
        bad_photos=[{
            "path": "data/photos/adv_0001/cover.jpg",
            "reason": "blurry"
        }],
    )

    assert explanation.bad_photos[0].path == "data/photos/adv_0001/cover.jpg"
    assert explanation.bad_photos[0].reason == "blurry"


def test_pipeline_reuses_room_prediction_for_bad_photo_details(monkeypatch, tmp_path):
    pipeline = import_module("quality_engine.pipeline.score_listing")
    photo_path = tmp_path / "cover.jpg"
    photo_path.touch()
    prediction = RoomPrediction(photo_path, "bedroom", 0.2)
    room_result = RoomCoverageResult(
        score=0.0,
        missing=["bedroom"],
        predictions=[prediction],
    )
    monkeypatch.setattr(
        pipeline,
        "score_technical_quality",
        lambda paths: TechnicalFeatureResult(
            score=40.0,
            blurry_count=1,
            photo_metrics=[PhotoTechnicalMetrics(valid=True, is_blurry=True)],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "score_aesthetic_quality",
        lambda paths, metrics: SimpleNamespace(score=50.0, reasons=[]),
    )
    monkeypatch.setattr(pipeline, "score_room_coverage",
                        lambda *args, **kwargs: room_result)
    monkeypatch.setattr(
        pipeline,
        "score_amenity_check",
        lambda *args: SimpleNamespace(score=50.0,
                                      reasons=[], undeclared_detected=[]),
    )
    monkeypatch.setattr(
        pipeline, "score_text_quality", lambda listing: SimpleNamespace(
            score=50.0, reasons=[])
    )
    monkeypatch.setattr(
        pipeline,
        "score_completeness",
        lambda listing: SimpleNamespace(score=50.0, reasons=[]),
    )
    classifier_calls = []
    monkeypatch.setattr(
        pipeline,
        "classify_photo",
        lambda path: classifier_calls.append(path),
        raising=False,
    )

    result = pipeline.score_listing(
        ListingRecord(listing_id="lst_0001", bedrooms=1),
        [PhotoRecord(entity_id="adv_0001", file="cover.jpg")],
        tmp_path,
    )

    assert classifier_calls == []
    assert result.explanation.bad_photos[0].reason == (
        "Blurry photo; room guess: unknown room "
        "(top guess bedroom, 20% classifier confidence)"
    )


def test_console_report_explains_photo_flags(capsys, tmp_path):
    row = {
        "listing_id": "lst_0001",
        "quality_score": 70.0,
        "photo_technical": 60.0,
        "photo_aesthetic": 70.0,
        "room_coverage": 60.0,
        "amenity_check": 50.0,
        "text_quality": 70.0,
        "completeness": 80.0,
        "photo_status": "ok",
        "explanation_json": (
            '{"positive": [], "issues": [], "suggestions": [], "bad_photos": '
            '[{"path": "photos/cover.jpg", "reason": '
            '"Below configured resolution target; room guess: living room '
            '(46% classifier confidence)", "width": 720, "height": 540}]}'
        ),
    }

    _print_console_report(pd.DataFrame([row]), tmp_path / "features.parquet")

    report = capsys.readouterr().out
    assert "Photo flags (problem, room guess, image size):" in report
    assert "room guess: living room (46% classifier confidence)" in report
    assert "720 x 540 px (0.39 MP)" in report
    assert "File: photos/cover.jpg" in report


def test_amenity_score_is_neutral_when_visual_detection_is_disabled(monkeypatch):
    monkeypatch.setenv("QUALITY_ENGINE_USE_CLIP", "0")
    result = amenities.score_amenity_check([], ["Wi-Fi", "Pool"])

    assert result.score == 50.0
    assert result.declared == ["pool", "wifi"]
    assert result.detected == []
    assert result.declared_not_detected == []


def test_explanation_includes_unconfirmed_amenities_and_neutral_status():
    explanation = build_explanation(
        technical_reasons=[],
        aesthetic_reasons=[],
        room_reasons=[],
        amenity_reasons=[
            "Wifi declared but not visually confirmed",
            "Visual amenity detection unavailable; neutral score used",
        ],
        text_reasons=[],
        completeness_reasons=[],
        undeclared_amenities=[],
        missing_rooms=[],
        blurry_count=0,
    )

    assert explanation.issues == [
        "Wifi declared but not visually confirmed",
        "Visual amenity detection unavailable; neutral score used",
    ]

    undeclared_explanation = build_explanation(
        technical_reasons=[],
        aesthetic_reasons=[],
        room_reasons=[],
        amenity_reasons=["Pool detected but not declared"],
        text_reasons=[],
        completeness_reasons=[],
        undeclared_amenities=["pool"],
        missing_rooms=[],
        blurry_count=0,
    )
    assert undeclared_explanation.issues == ["Pool detected but not declared"]


def test_loader_defaults_blank_csv_fields_and_parses_false(tmp_path: Path):
    path = tmp_path / "listings.csv"
    pd.DataFrame([{
        "listing_id": "lst_0002",
        "vertical": None,
        "capacity": None,
        "bedrooms": "",
        "bathrooms": None,
        "base_price": None,
        "instant_booking": "False",
        "is_calendar_synced": None,
    }]).to_csv(path, index=False)

    [listing] = load_listings(path)

    assert listing.listing_id == "lst_0002"
    assert listing.vertical == ""
    assert listing.capacity == listing.bedrooms == 0
    assert listing.bathrooms == listing.base_price == 0.0
    assert listing.instant_booking is False
    assert listing.is_calendar_synced is False

    photos_path = tmp_path / "photos.csv"
    pd.DataFrame([{
        "entity_id": "adv_0002",
        "file": "photos/adv_0002/cover.jpg",
        "seq": None,
        "is_cover": "False",
        "width": None,
        "height": None,
        "bytes": None,
    }]).to_csv(photos_path, index=False)
    [photo] = load_photos(photos_path)["adv_0002"]

    assert photo.seq == photo.width == photo.height == photo.bytes == 0
    assert photo.is_cover is False
