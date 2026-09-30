"""Load listings and photos from CSV / parquet sources."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from quality_engine.schemas.listing import ListingRecord, PhotoRecord


def _is_missing(value: object) -> bool:
    return value is None or (not isinstance(value, (list, dict)) and pd.isna(value))


def _as_text(value: object) -> str:
    return "" if _is_missing(value) else str(value)


def _as_int(value: object) -> int:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        return 0
    return int(float(value))


def _as_float(value: object) -> float:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        return 0.0
    return float(value)


def _as_bool(value: object) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Cannot interpret {value!r} as a boolean")
    return bool(value)


def _parse_amenities(value: object) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return [str(a) for a in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = [text]
        if isinstance(parsed, list):
            return [str(a) for a in parsed]
    return [a.strip() for a in text.split("|") if a.strip()]


def load_listings(path: Path) -> list[ListingRecord]:
    df = pd.read_csv(path)
    records: list[ListingRecord] = []
    for row in df.to_dict(orient="records"):
        listing_id = _as_text(row.get("listing_id", "")).strip()
        if not listing_id:
            continue
        records.append(
            ListingRecord(
                listing_id=listing_id,
                vertical=_as_text(row.get("vertical", "")),
                name=_as_text(row.get("name", "")),
                description=_as_text(row.get("description", "")),
                city=_as_text(row.get("city", "")),
                country=_as_text(row.get("country", "")),
                capacity=_as_int(row.get("capacity")),
                bedrooms=_as_int(row.get("bedrooms")),
                bathrooms=_as_float(row.get("bathrooms")),
                base_price=_as_float(row.get("base_price")),
                currency=_as_text(row.get("currency", "")),
                instant_booking=_as_bool(row.get("instant_booking", False)),
                is_calendar_synced=_as_bool(
                    row.get("is_calendar_synced", False)),
                has_active_ical=_as_bool(row.get("has_active_ical", False)),
                pricing_days_180=_as_int(row.get("pricing_days_180")),
                blocked_days_180=_as_int(row.get("blocked_days_180")),
                amenities=_parse_amenities(row.get("amenities")),
                photo_count=_as_int(row.get("photo_count")),
                property_type=_as_text(row.get("property_type", "")),
                status=_as_text(row.get("status", "")),
            )
        )
    return records


def load_photos(path: Path) -> dict[str, list[PhotoRecord]]:
    df = pd.read_csv(path)
    grouped: dict[str, list[PhotoRecord]] = defaultdict(list)
    for row in df.to_dict(orient="records"):
        entity_id = _as_text(row.get("entity_id", "")).strip()
        file_path = _as_text(row.get("file", "")).strip()
        if not entity_id or not file_path:
            continue
        grouped[entity_id].append(
            PhotoRecord(
                entity_id=entity_id,
                file=file_path,
                seq=_as_int(row.get("seq")),
                is_cover=_as_bool(row.get("is_cover", False)),
                width=_as_int(row.get("width")),
                height=_as_int(row.get("height")),
                bytes=_as_int(row.get("bytes")),
            )
        )
    for photos in grouped.values():
        photos.sort(key=lambda p: p.seq)
    return dict(grouped)
