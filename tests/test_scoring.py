"""Unit tests for quality score engine."""

from quality_engine.config import validate_weights_sum_to_100
from quality_engine.schemas.output import ComponentScores
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
