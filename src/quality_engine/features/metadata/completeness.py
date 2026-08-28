"""Listing metadata completeness scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from quality_engine.config import load_thresholds
from quality_engine.schemas.listing import ListingRecord


@dataclass
class CompletenessResult:
    score: float
    filled_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def score_completeness(listing: ListingRecord) -> CompletenessResult:
    thresholds = load_thresholds()["completeness"]
    checks: dict[str, bool] = {
        "title": bool(listing.name.strip()),
        "description": bool(listing.description.strip()),
        "amenities": len(listing.amenities) > 0,
        "capacity": listing.capacity > 0,
        "bedrooms": listing.bedrooms > 0,
        "price": listing.base_price > 0,
        "city": bool(listing.city),
        "photos": listing.photo_count > 0,
        "calendar": listing.pricing_days_180 >= thresholds["min_calendar_depth_days"],
        "instant_book": listing.instant_booking,
        "ical": listing.has_active_ical or listing.is_calendar_synced,
    }

    # Peer-adjusted expectation: larger listings should have more amenities
    min_amenities = 3 if listing.capacity <= 2 else 5 if listing.capacity <= 4 else 7
    if len(listing.amenities) >= min_amenities:
        checks["amenities"] = True

    filled = [k for k, ok in checks.items() if ok]
    missing = [k for k, ok in checks.items() if not ok]
    score = (len(filled) / len(checks)) * 100

    reasons = [f"{field} present" for field in filled[:5]]
    for field in missing:
        reasons.append(f"Missing or weak: {field.replace('_', ' ')}")

    return CompletenessResult(score=float(score), filled_fields=filled, missing_fields=missing, reasons=reasons)
