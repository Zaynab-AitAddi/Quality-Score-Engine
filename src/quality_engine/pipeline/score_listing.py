"""End-to-end quality scoring pipeline for a single listing."""

from __future__ import annotations

import logging
from pathlib import Path

from quality_engine.features.image.aesthetic import score_aesthetic_quality
from quality_engine.features.image.amenities import score_amenity_check
from quality_engine.features.image.room_coverage import score_room_coverage
from quality_engine.features.image.technical import score_technical_quality
from quality_engine.features.metadata.completeness import score_completeness
from quality_engine.features.text.language import score_text_quality
from quality_engine.schemas.listing import ListingRecord, PhotoRecord
from quality_engine.schemas.output import ComponentScores, QualityResult
from quality_engine.scoring.weighted_score import assemble_result, build_explanation

logger = logging.getLogger(__name__)


def resolve_photo_paths(data_root: Path, photos: list[PhotoRecord]) -> list[Path]:
    paths: list[Path] = []
    for photo in photos:
        candidate = data_root / photo.file
        if candidate.exists():
            paths.append(candidate)
        else:
            alt = data_root.parent / photo.file
            if alt.exists():
                paths.append(alt)
    return paths


def score_listing(
    listing: ListingRecord,
    photos: list[PhotoRecord],
    data_root: Path,
) -> QualityResult:
    photo_paths = resolve_photo_paths(data_root, photos)
    photo_status = "ok" if photo_paths else "no_photos"

    logger.info("listing=%s photos=%s", listing.listing_id, len(photo_paths))

    technical = score_technical_quality(photo_paths)
    aesthetic = score_aesthetic_quality(photo_paths)
    rooms = score_room_coverage(photo_paths, bedrooms=max(listing.bedrooms, 1))
    amenities = score_amenity_check(photo_paths, listing.amenities)
    text = score_text_quality(listing)
    completeness = score_completeness(listing)

    components = ComponentScores(
        photo_technical=technical.score,
        photo_aesthetic=aesthetic.score,
        room_coverage=rooms.score,
        amenity_check=amenities.score,
        text_quality=text.score,
        completeness=completeness.score,
    )

    explanation = build_explanation(
        technical_reasons=technical.reasons,
        aesthetic_reasons=aesthetic.reasons,
        room_reasons=rooms.reasons,
        amenity_reasons=amenities.reasons,
        text_reasons=text.reasons,
        completeness_reasons=completeness.reasons,
        undeclared_amenities=amenities.undeclared_detected,
        missing_rooms=rooms.missing,
        blurry_count=technical.blurry_count,
    )

    result = assemble_result(listing.listing_id, components, explanation, photo_status)
    logger.info("listing=%s final_score=%.1f", listing.listing_id, result.quality_score)
    return result
