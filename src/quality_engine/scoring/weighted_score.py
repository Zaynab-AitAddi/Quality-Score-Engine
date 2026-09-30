"""Weighted score aggregation and explainability."""

from __future__ import annotations

from quality_engine.config import (
    load_models_config,
    load_weights,
    validate_weights_sum_to_100,
)
from quality_engine.schemas.output import ComponentScores, Explanation, QualityResult


def compute_weighted_score(components: ComponentScores) -> float:
    weights = load_weights()["weights"]
    validate_weights_sum_to_100(weights)
    total = (
        components.photo_technical * weights["photo_technical"]
        + components.photo_aesthetic * weights["photo_aesthetic"]
        + components.room_coverage * weights["room_coverage"]
        + components.amenity_check * weights["amenity_check"]
        + components.text_quality * weights["text_quality"]
        + components.completeness * weights["completeness"]
    ) / 100.0
    return round(min(max(total, 0.0), 100.0), 2)


def build_explanation(
    *,
    technical_reasons: list[str],
    aesthetic_reasons: list[str],
    room_reasons: list[str],
    amenity_reasons: list[str],
    text_reasons: list[str],
    completeness_reasons: list[str],
    undeclared_amenities: list[str],
    missing_rooms: list[str],
    blurry_count: int,
    bad_photos: list[dict[str, str | int | None]] | None = None,
) -> Explanation:
    positive: list[str] = []
    issues: list[str] = []
    suggestions: list[str] = []
    bad_photos = bad_photos or []

    for reason in technical_reasons + aesthetic_reasons + room_reasons + amenity_reasons:
        lower = reason.lower()
        if any(tok in lower for tok in ("good", "detected and declared", "coverage")):
            positive.append(reason)
        elif any(tok in lower for tok in (
            "blurry",
            "duplicate",
            "low-resolution",
            "missing",
            "no ",
            "not visually confirmed",
            "detected but not declared",
            "unavailable",
        )):
            issues.append(reason)

    for reason in text_reasons:
        if "missing" in reason.lower() or "short" in reason.lower() or "empty" in reason.lower():
            issues.append(reason)
        else:
            positive.append(reason)

    for reason in completeness_reasons:
        if reason.startswith("Missing"):
            issues.append(reason)

    if blurry_count:
        suggestions.append(f"Replace {blurry_count} blurry photo(s)")
    for room in missing_rooms:
        suggestions.append(f"Add a clear {room} photo")
    for amenity in undeclared_amenities:
        suggestions.append(
            f"Consider declaring the visually detected {amenity.replace('_', ' ')}")

    if any("arabic" in i.lower() for i in issues):
        suggestions.append("Add an Arabic description")

    return Explanation(
        positive=positive[:10],
        issues=issues[:10],
        suggestions=suggestions[:8],
        bad_photos=[
            {
                "path": str(item.get("path", "")),
                "reason": str(item.get("reason", "unknown")),
                "width": item.get("width"),
                "height": item.get("height"),
            }
            for item in bad_photos
        ],
    )


def assemble_result(
    listing_id: str,
    components: ComponentScores,
    explanation: Explanation,
    photo_status: str = "ok",
) -> QualityResult:
    cfg = load_weights()
    models = load_models_config()
    return QualityResult(
        listing_id=listing_id,
        quality_score=compute_weighted_score(components),
        components=components,
        explanation=explanation,
        version="quality-v1",
        quality_engine_version=models.get("quality_engine_version", "1.0.0"),
        config_version=cfg.get("config_version", "quality-config-v1"),
        photo_status=photo_status,
    )
