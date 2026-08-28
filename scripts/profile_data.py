#!/usr/bin/env python3
"""Profile the anonymized listing dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quality_engine.config import PROJECT_ROOT


def profile(data_dir: Path, output: Path) -> None:
    listings = pd.read_csv(data_dir / "listings.csv")
    photos = pd.read_csv(data_dir / "photos.csv")
    cities = pd.read_csv(data_dir / "cities.csv")

    stay_listings = listings[listings["vertical"] == "stay"] if "vertical" in listings else listings

    lines = [
        "# Data Profile",
        "",
        f"Generated from: `{data_dir}`",
        "",
        "## Listings",
        "",
        f"- Total rows: **{len(listings)}**",
        f"- Stay listings: **{len(stay_listings)}**",
        f"- Columns: `{', '.join(listings.columns.tolist())}`",
        "",
        "### Missing values (listings)",
        "",
    ]

    missing = listings.isnull().sum()
    for col, count in missing.items():
        if count:
            lines.append(f"- `{col}`: {count}")

    lines.extend(
        [
            "",
            "### Cities",
            "",
            f"- Unique cities in listings: **{listings['city'].nunique()}**",
            f"- Top cities: {listings['city'].value_counts().head(5).to_dict()}",
            "",
            "### Amenities sample",
            "",
            f"- Listings with amenities field: **{listings['amenities'].notna().sum()}**",
            "",
            "## Photos",
            "",
            f"- Photo metadata rows: **{len(photos)}**",
            f"- Entities with photos: **{photos['entity_id'].nunique()}**",
            f"- Avg photos per entity: **{photos.groupby('entity_id').size().mean():.1f}**",
            "",
            "## Cities reference",
            "",
            f"- Reference cities: **{len(cities)}**",
            "",
            "## Notes",
            "",
            "- Primary scoring target is `vertical=stay` listings.",
            "- Photo files live under `data/photos/{listing_id}/`.",
            "- Use `python -m quality_engine.run` to generate the feature table.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data_profile.md")
    args = parser.parse_args()
    profile(args.data, args.output)


if __name__ == "__main__":
    main()
