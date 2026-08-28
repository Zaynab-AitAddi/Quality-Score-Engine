"""Tests for text quality features."""

from quality_engine.features.text.language import detect_languages, score_text_quality
from quality_engine.schemas.listing import ListingRecord


def test_detect_french_and_arabic():
    text = "Bel appartement avec deux chambres مرحبا"
    langs = detect_languages(text)
    assert langs["arabic"] is True


def test_missing_description():
    listing = ListingRecord(listing_id="lst_test", name="", description="")
    result = score_text_quality(listing)
    assert result.score < 50
