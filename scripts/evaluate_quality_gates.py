#!/usr/bin/env python3
"""Run feature-table validation and optional room-accuracy and blur-precision gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from quality_engine.validation import run_quality_gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project A quality gates")
    parser.add_argument("--feature-table", type=Path, default=None,
                        help="Optional parquet feature table to validate")
    parser.add_argument("--room-labels", type=Path, default=None,
                        help="CSV with photo_file,label for room classification")
    parser.add_argument("--blur-labels", type=Path, default=None,
                        help="CSV with photo_file,label for blur precision")
    parser.add_argument("--photos-root", type=Path, default=None,
                        help="Root directory containing the photo files")
    args = parser.parse_args()

    summary = run_quality_gates(
        feature_table=args.feature_table,
        room_labels_csv=args.room_labels,
        blur_labels_csv=args.blur_labels,
        photos_root=args.photos_root,
    )
    if not summary:
        print("No gate inputs provided. Pass --feature-table, --room-labels, or --blur-labels.")
        return

    print(summary)


if __name__ == "__main__":
    main()
