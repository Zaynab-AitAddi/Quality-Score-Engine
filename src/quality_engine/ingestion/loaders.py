"""Load listings and photos from CSV / parquet sources."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from quality_engine.schemas.listing import ListingRecord, PhotoRecord


def _parse_amenities(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
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
        listing_id = str(row.get("listing_id", ""))
        if not listing_id:
            continue
        records.append(
            ListingRecord(
                listing_id=listing_id,
                vertical=str(row.get("vertical", "") or ""),
                name=str(row.get("name", "") or ""),
                description=str(row.get("description", "") or ""),
                city=str(row.get("city", "") or ""),
                country=str(row.get("country", "") or ""),
                capacity=int(row.get("capacity") or 0),
                bedrooms=int(row.get("bedrooms") or 0),
                bathrooms=float(row.get("bathrooms") or 0),
                base_price=float(row.get("base_price") or 0),
                currency=str(row.get("currency", "") or ""),
                instant_booking=bool(row.get("instant_booking", False)),
                is_calendar_synced=bool(row.get("is_calendar_synced", False)),
                has_active_ical=bool(row.get("has_active_ical", False)),
                pricing_days_180=int(row.get("pricing_days_180") or 0),
                blocked_days_180=int(row.get("blocked_days_180") or 0),
                amenities=_parse_amenities(row.get("amenities")),
                photo_count=int(row.get("photo_count") or 0),
                property_type=str(row.get("property_type", "") or ""),
                status=str(row.get("status", "") or ""),
            )
        )
    return records


def load_photos(path: Path) -> dict[str, list[PhotoRecord]]:
    df = pd.read_csv(path)
    grouped: dict[str, list[PhotoRecord]] = defaultdict(list)
    for row in df.to_dict(orient="records"):
        entity_id = str(row.get("entity_id", ""))
        if not entity_id:
            continue
        grouped[entity_id].append(
            PhotoRecord(
                entity_id=entity_id,
                file=str(row.get("file", "")),
                seq=int(row.get("seq") or 0),
                is_cover=bool(row.get("is_cover", False)),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
                bytes=int(row.get("bytes") or 0),
            )
        )
    for photos in grouped.values():
        photos.sort(key=lambda p: p.seq)
    return dict(grouped)
