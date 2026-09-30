#!/usr/bin/env python3
"""Evaluate the room-type classifier on a labeled CSV.

CSV format: photo_file,label
Example:
photos/adv_0001/photo1.jpg,bedroom
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from quality_engine.features.image.room_coverage import classify_photo


def main(labeled_csv: Path, photos_root: Path | None = None) -> None:
    df = pd.read_csv(labeled_csv)
    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        path = Path(row["photo_file"])
        if photos_root and not path.is_absolute():
            path = photos_root / path
        pred = classify_photo(path)
        y_true.append(row["label"])
        y_pred.append(pred.label)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate room classifier")
    parser.add_argument("labeled_csv", type=Path)
    parser.add_argument("--photos-root", type=Path, default=None)
    args = parser.parse_args()
    main(args.labeled_csv, args.photos_root)
