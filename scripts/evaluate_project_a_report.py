#!/usr/bin/env python3
"""Generate a concise Project A evaluation summary report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quality_engine.validation import run_quality_gates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Project A evaluation summary")
    parser.add_argument("--feature-table", type=Path, default=None)
    parser.add_argument("--room-labels", type=Path, default=None)
    parser.add_argument("--blur-labels", type=Path, default=None)
    parser.add_argument("--photos-root", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts") / "project_a_eval_report.json")
    args = parser.parse_args()

    report = run_quality_gates(
        feature_table=args.feature_table,
        room_labels_csv=args.room_labels,
        blur_labels_csv=args.blur_labels,
        photos_root=args.photos_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(report).to_json(args.out)
    print(f"Saved evaluation report to {args.out}")
    print(report)


if __name__ == "__main__":
    main()
