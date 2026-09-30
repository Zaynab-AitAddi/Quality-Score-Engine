"""Simple calibration utilities for aesthetic scoring.

Provides a small linear calibration (slope, intercept) to map raw model
predictions to human ratings (0-100).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LinearRegression


@dataclass
class LinearCalibrator:
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, x: float) -> float:
        return float(np.clip(self.slope * x + self.intercept, 0.0, 100.0))


def fit_calibration(raw_scores: list[float], human_scores: list[float]) -> LinearCalibrator:
    X = np.array(raw_scores).reshape(-1, 1)
    y = np.array(human_scores).reshape(-1, 1)
    if len(X) < 2:
        return LinearCalibrator()
    model = LinearRegression()
    model.fit(X, y)
    return LinearCalibrator(slope=float(model.coef_[0][0]), intercept=float(model.intercept_[0]))


def save_calibrator(calibrator: LinearCalibrator, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"slope": calibrator.slope,
                       "intercept": calibrator.intercept}, fh)


def load_calibrator(path: Path) -> LinearCalibrator | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return LinearCalibrator(slope=float(data.get("slope", 1.0)), intercept=float(data.get("intercept", 0.0)))
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
        return None
