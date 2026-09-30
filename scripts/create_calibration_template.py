#!/usr/bin/env python3
"""Create a CSV template for aesthetic calibration ratings."""

from __future__ import annotations

import csv
from pathlib import Path


def main(out: Path | None = None) -> None:
    p = out or Path("data") / "calibration_template.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    headers = ["photo_file", "rater1", "rater2", "avg_rating"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        # Add an example row
        writer.writerow(["photos/adv_0001/photo1.jpg", "", "", ""])
    print(f"Wrote calibration template to {p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create calibration CSV template")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
