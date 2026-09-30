"""CLI entrypoint for the Quality Score Engine."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from quality_engine.config import PROJECT_ROOT
from quality_engine.features.image.technical import configure_opencv_threads
from quality_engine.ingestion.loaders import load_listings, load_photos
from quality_engine.pipeline.score_listing import score_listing

configure_opencv_threads()


def _score_bar(score: float, width: int = 20) -> str:
    filled = round(max(0.0, min(100.0, score)) / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colorize(text: str, color: str | None = None) -> str:
    if not _supports_color() or color is None:
        return text
    palette = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{palette.get(color, '')}{text}{palette['reset']}"


def _score_label(score: float) -> str:
    if score >= 80:
        return "EXCELLENT"
    if score >= 65:
        return "GOOD"
    if score >= 50:
        return "NEEDS WORK"
    return "LOW"


def _score_color(score: float) -> str:
    if score >= 80:
        return "green"
    if score >= 65:
        return "cyan"
    if score >= 50:
        return "yellow"
    return "red"


def _items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _print_items(label: str, values: list[str]) -> None:
    if not values:
        return
    print(f"   {label}:")
    for value in values:
        wrapped = textwrap.wrap(value, width=82) or [""]
        print(f"      - {wrapped[0]}")
        for continuation in wrapped[1:]:
            print(f"        {continuation}")


def _print_console_report(df: pd.DataFrame, output: Path) -> None:
    """Print a polished summary without changing the feature contract."""
    ranked = df.sort_values(
        "quality_score", ascending=False).reset_index(drop=True)
    avg_score = ranked["quality_score"].mean() if not ranked.empty else 0.0
    print()
    print("+" + "=" * 92 + "+")
    print("|" + _colorize(" QUALITY SCORE REPORT ", "bold").center(92) + "|")
    print("+" + "=" * 92 + "+")
    print(f"Saved parquet: {output}")
    print(f"Listings scored: {len(ranked)}    Average: {avg_score:.1f}/100")
    print()
    print("RANK  LISTING    SCORE  LEVEL        TECH  AESTH  ROOMS  AMEN  TEXT  COMPLETE  STATUS")
    print("-" * 104)
    for rank, row in ranked.iterrows():
        label = _score_label(row["quality_score"])
        score_text = f"{row['quality_score']:.1f}"
        label_text = _colorize(label, _score_color(row["quality_score"]))
        print(
            f"{rank + 1:>4}  {row['listing_id']!s:<9}  "
            f"{score_text:>5}  {label_text:<11}  "
            f"{row['photo_technical']:>5.1f} {row['photo_aesthetic']:>6.1f} "
            f"{row['room_coverage']:>6.1f} {row['amenity_check']:>5.1f} "
            f"{row['text_quality']:>5.1f} {row['completeness']:>9.1f}  {row['photo_status']}"
        )

    print()
    print("LISTING DETAILS")
    print("-" * 104)
    for rank, row in ranked.iterrows():
        explanation = json.loads(row["explanation_json"])
        score = float(row["quality_score"])
        label = _colorize(_score_label(score), _score_color(score))
        print(
            f"\n{rank + 1}. {row['listing_id']}  {score:.1f}/100 {_score_bar(score)}  {label}"
        )
        print(
            f"   Components: technical {row['photo_technical']:.1f} | aesthetic {row['photo_aesthetic']:.1f} | "
            f"rooms {row['room_coverage']:.1f} | amenities {row['amenity_check']:.1f} | "
            f"text {row['text_quality']:.1f} | completeness {row['completeness']:.1f}"
        )
        positive = _items(explanation.get("positive"))
        issues = _items(explanation.get("issues"))
        suggestions = _items(explanation.get("suggestions"))
        bad_photos = explanation.get("bad_photos", [])
        _print_items("Strengths", positive)
        _print_items("Issues", issues)
        if bad_photos:
            print("   Photo flags (problem, room guess, image size):")
            for item in bad_photos:
                reason = str(item.get("reason", "unknown")).replace("_", " ")
                width = item.get("width")
                height = item.get("height")
                if width is not None and height is not None:
                    megapixels = int(width) * int(height) / 1_000_000
                    print(
                        f"      - {reason}; {width} x {height} px ({megapixels:.2f} MP)"
                    )
                else:
                    print(f"      - {reason}")
                photo_path = str(item.get("path", ""))
                if photo_path:
                    for line in textwrap.wrap(f"File: {photo_path}", width=78):
                        print(f"        {line}")
        _print_items("Next steps", suggestions)
    print("\n" + "=" * 104)


def _match_listing_photos(listing_id: str, photos_by_entity: dict[str, list[Any]], data_root: Path) -> list[Any]:
    """Match a listing to its photo records using the numeric suffix heuristic.

    The provided data uses photo entity ids such as adv_0001 while listing ids use
    lst_0001; the shared numeric suffix is therefore the most reliable stable key.
    """
    suffix = re.search(r"(\d+)$", listing_id)
    target = suffix.group(1) if suffix else ""

    matches: list[Any] = []
    for entity_id, photo_records in photos_by_entity.items():
        entity_suffix = re.search(r"(\d+)$", entity_id)
        entity_target = entity_suffix.group(1) if entity_suffix else ""
        if not target:
            continue
        if entity_target == target:
            matches.extend(photo_records)

    if matches:
        return matches

    # Fallback: use the same suffix in file paths when a direct entity match is absent.
    for photo_records in photos_by_entity.values():
        for photo in photo_records:
            if target and target in str(photo.file):
                matches.append(photo)
    return matches


def build_feature_rows(data_dir: Path = PROJECT_ROOT / "data", limit: int | None = None) -> list[dict[str, Any]]:
    configure_opencv_threads()
    listings = load_listings(data_dir / "listings.csv")
    photos_by_entity = load_photos(data_dir / "photos.csv")

    rows: list[dict[str, Any]] = []
    for listing in listings[:limit] if limit is not None else listings:
        if listing.vertical and listing.vertical != "stay":
            continue
        matched_photos = _match_listing_photos(
            listing.listing_id, photos_by_entity, data_dir)
        if matched_photos:
            photo_records = matched_photos
        else:
            photo_records = []

        result = score_listing(listing, photo_records, data_dir)
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
            "explanation_json": json.dumps(result.explanation.model_dump(), separators=(",", ":")),
        }
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the quality score engine over a dataset.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data",
                        help="Data directory containing listings.csv and photos.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional maximum number of listings to score")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT /
                        "output" / "features.parquet", help="Parquet file to write")
    args = parser.parse_args()

    rows = build_feature_rows(args.data, args.limit)
    if not rows:
        print(f"No rows were generated from {args.data}")
        return

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    _print_console_report(df, args.output)


if __name__ == "__main__":
    main()
