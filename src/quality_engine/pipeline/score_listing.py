"""End-to-end quality scoring pipeline for a single listing."""

from __future__ import annotations

import logging
from pathlib import Path

from quality_engine.config import load_thresholds
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
    aesthetic = score_aesthetic_quality(photo_paths, technical.photo_metrics)
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

    bad_photos = []
    room_predictions = {
        prediction.path: prediction for prediction in rooms.predictions}
    room_confidence_threshold = load_thresholds()[
        "room"]["confidence_threshold"]
    for path, metrics in zip(photo_paths, technical.photo_metrics):
        if not metrics.valid:
            continue
        room_prediction = room_predictions.get(path)
        room_guess = room_prediction.label if room_prediction else "other"
        if room_prediction is None:
            room_description = "room guess unavailable"
        else:
            room_label = room_guess.replace("_", " ")
            if room_label == "living":
                room_label = "living room"
            recognized_room = room_label in {
                "bedroom", "bathroom", "kitchen", "living room", "exterior", "pool", "view"
            }
            if room_prediction.confidence >= room_confidence_threshold and recognized_room:
                room_description = (
                    f"room guess: {room_label} "
                    f"({room_prediction.confidence:.0%} classifier confidence)"
                )
            else:
                room_description = (
                    f"room guess: unknown room (top guess {room_label}, "
                    f"{room_prediction.confidence:.0%} classifier confidence)"
                )
        if metrics.is_blurry:
            bad_photos.append({
                "path": str(path),
                "reason": f"Blurry photo; {room_description}",
                "width": metrics.width,
                "height": metrics.height,
            })
        elif metrics.resolution_score < 40:
            bad_photos.append({
                "path": str(path),
                "reason": f"Below configured resolution target; {room_description}",
                "width": metrics.width,
                "height": metrics.height,
            })

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
        bad_photos=bad_photos,
    )

    result = assemble_result(
        listing.listing_id, components, explanation, photo_status)
    logger.info("listing=%s final_score=%.1f",
                listing.listing_id, result.quality_score)
    return result
