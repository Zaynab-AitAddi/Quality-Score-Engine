"""Test configured weight totals, weighted-score bounds, defaults, and monotonicity."""

import pytest

from quality_engine.config import validate_weights_sum_to_100
from quality_engine.features.image import technical
from quality_engine.schemas.output import ComponentScores
from quality_engine.scoring import weighted_score
from quality_engine.scoring.weighted_score import compute_weighted_score


def test_weights_sum_to_100():
    assert validate_weights_sum_to_100() == 100.0


def test_score_range_0_100():
    components = ComponentScores(
        photo_technical=100,
        photo_aesthetic=100,
        room_coverage=100,
        amenity_check=100,
        text_quality=100,
        completeness=100,
    )
    assert compute_weighted_score(components) == 100.0

    empty = ComponentScores()
    assert compute_weighted_score(empty) == 0.0


def test_score_monotonicity():
    base = ComponentScores(
        photo_technical=50,
        photo_aesthetic=50,
        room_coverage=50,
        amenity_check=50,
        text_quality=50,
        completeness=50,
    )
    improved = base.model_copy(update={"photo_technical": 80})
    assert compute_weighted_score(improved) >= compute_weighted_score(base)


def test_invalid_weights_are_rejected_before_scoring(monkeypatch):
    invalid_weights = [
        {
            "photo_technical": -10,
            "photo_aesthetic": 30,
            "room_coverage": 20,
            "amenity_check": 20,
            "text_quality": 20,
            "completeness": 20,
        },
        {
            "photo_technical": float("nan"),
            "photo_aesthetic": 15,
            "room_coverage": 10,
            "amenity_check": 10,
            "text_quality": 15,
            "completeness": 50,
        },
        {"photo_technical": 100},
    ]

    for weights in invalid_weights:
        monkeypatch.setattr(weighted_score, "load_weights",
                            lambda weights=weights: {"weights": weights})
        with pytest.raises(ValueError):
            compute_weighted_score(ComponentScores())


def test_opencv_thread_count_is_configurable(monkeypatch):
    original_threads = technical.cv2.getNumThreads()
    try:
        monkeypatch.setenv("QUALITY_ENGINE_OPENCV_THREADS", "2")
        assert technical.configure_opencv_threads() == 2
        assert technical.cv2.getNumThreads() == 2

        monkeypatch.setenv("QUALITY_ENGINE_OPENCV_THREADS", "invalid")
        with pytest.raises(ValueError, match="must be a positive integer"):
            technical.configure_opencv_threads()
    finally:
        technical.cv2.setNumThreads(original_threads)


def test_image_worker_count_is_configurable_and_bounded(monkeypatch):
    monkeypatch.setenv("QUALITY_ENGINE_IMAGE_WORKERS", "3")

    assert technical._image_worker_count(10) == 3
    assert technical._image_worker_count(2) == 2

    monkeypatch.setenv("QUALITY_ENGINE_IMAGE_WORKERS", "0")
    with pytest.raises(ValueError, match="must be a positive integer"):
        technical._image_worker_count(10)
