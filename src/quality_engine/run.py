"""CLI entry point for batch quality scoring."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from quality_engine import __version__
from quality_engine.config import PROJECT_ROOT, validate_weights_sum_to_100
from quality_engine.ingestion.loaders import load_listings, load_photos
from quality_engine.pipeline.score_listing import score_listing


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def run_batch(
    listings_path: Path,
    photos_path: Path,
    data_root: Path,
    output_path: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    validate_weights_sum_to_100()
    listings = load_listings(listings_path)
    photo_map = load_photos(photos_path)

    if limit:
        listings = listings[:limit]

    rows: list[dict] = []
    for listing in listings:
        photos = photo_map.get(listing.listing_id, [])
        result = score_listing(listing, photos, data_root)
        row = {
            "listing_id": result.listing_id,
            "quality_score": result.quality_score,
            "photo_technical": result.components.photo_technical,
            "photo_aesthetic": result.components.photo_aesthetic,
            "room_coverage": result.components.room_coverage,
            "amenity_check": result.components.amenity_check,
            "text_quality": result.components.text_quality,
            "completeness": result.components.completeness,
            "photo_status": result.photo_status,
            "version": result.version,
            "quality_engine_version": result.quality_engine_version,
            "config_version": result.config_version,
            "explanation_json": json.dumps(result.explanation.model_dump(), ensure_ascii=False),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Listing Quality Score Engine")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "listings.csv")
    parser.add_argument("--photos-meta", type=Path, default=PROJECT_ROOT / "data" / "photos.csv")
    parser.add_argument("--photos", type=Path, default=PROJECT_ROOT / "data", help="Root folder for photo files")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "features.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Process only N listings (for testing)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    logging.info("quality_engine v%s", __version__)
    df = run_batch(args.input, args.photos_meta, args.photos, args.output, args.limit)
    logging.info("Wrote %s rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
