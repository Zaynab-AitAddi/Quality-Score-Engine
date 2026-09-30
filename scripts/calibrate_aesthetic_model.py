#!/usr/bin/env python3
"""Fit a linear calibration curve for the aesthetic score against human ratings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quality_engine.calibration import fit_calibration, save_calibrator
from quality_engine.config import PROJECT_ROOT
from quality_engine.features.image.aesthetic import _aesthetic_heuristic


def load_ratings(csv_path: Path) -> tuple[list[float], list[float]]:
    df = pd.read_csv(csv_path)
    required = {"photo_file"}
    if not required.issubset(df.columns):
        raise ValueError("Calibration CSV must contain a photo_file column")

    if "avg_rating" in df.columns:
        ratings = df["avg_rating"].astype(float).tolist()
    else:
        if not {"rater1", "rater2"}.issubset(df.columns):
            raise ValueError(
                "Calibration CSV must contain avg_rating or rater1/rater2 columns")
        ratings = ((df["rater1"].fillna(0).astype(float) +
                   df["rater2"].fillna(0).astype(float)) / 2.0).tolist()

    raw_scores: list[float] = []
    valid_ratings: list[float] = []
    for _, row in df.iterrows():
        rating = float(ratings[len(raw_scores)])
        if pd.isna(rating):
            continue
        photo_file = str(row["photo_file"]).strip()
        photo_path = Path(photo_file)
        if not photo_path.is_absolute():
            photo_path = PROJECT_ROOT / photo_file
        if not photo_path.exists():
            continue
        raw_scores.append(_aesthetic_heuristic(photo_path))
        valid_ratings.append(float(rating))

    if not raw_scores:
        raise ValueError(f"No valid calibration rows found in {csv_path}")

    if len(raw_scores) < 2:
        raise ValueError(
            "Need at least two rated photos to fit a calibration model")

    return raw_scores, valid_ratings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the aesthetic calibration model")
    parser.add_argument("--csv", type=Path, required=True,
                        help="CSV with photo_file and human rating columns")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" /
                        "aesthetic_calibrator.yaml", help="Where to save the trained calibrator")
    args = parser.parse_args()

    raw_scores, ratings = load_ratings(args.csv)
    calibrator = fit_calibration(raw_scores, ratings)
    save_calibrator(calibrator, args.out)
    print(f"Saved calibrator to {args.out}")
    print(f"slope={calibrator.slope:.4f} intercept={calibrator.intercept:.4f}")


if __name__ == "__main__":
    main()
