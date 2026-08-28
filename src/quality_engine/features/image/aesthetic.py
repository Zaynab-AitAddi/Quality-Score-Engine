"""Aesthetic quality using a lightweight heuristic (calibration-ready)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from quality_engine.features.image.technical import analyze_photo
from quality_engine.calibration import load_calibrator
from quality_engine.config import PROJECT_ROOT


@dataclass
class AestheticFeatureResult:
    score: float
    reasons: list[str] = field(default_factory=list)


def _aesthetic_heuristic(path: Path) -> float:
    """Proxy score until human-rated calibration dataset is applied."""
    metrics = analyze_photo(path)
    if not metrics.valid:
        return 0.0

    img = cv2.imread(str(path))
    if img is None:
        return 0.0

    # Colorfulness + contrast + sharpness proxy
    (b, g, r) = cv2.split(img.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = np.sqrt(np.mean(rg**2) + np.mean(yb**2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    sharp = min(metrics.sharpness / 150.0, 1.0)

    raw = 0.35 * min(colorfulness / 40.0, 1.0) + 0.35 * \
        min(contrast / 60.0, 1.0) + 0.30 * sharp
    return float(np.clip(raw * 100, 0, 100))


def score_aesthetic_quality(photo_paths: list[Path]) -> AestheticFeatureResult:
    if not photo_paths:
        return AestheticFeatureResult(score=0.0, reasons=["No photos for aesthetic scoring"])

    scores = [_aesthetic_heuristic(p) for p in photo_paths if p.exists()]
    if not scores:
        return AestheticFeatureResult(score=0.0, reasons=["No valid photos for aesthetic scoring"])

    avg = float(np.mean(scores))
    # Apply optional calibration if available
    calib = load_calibrator(PROJECT_ROOT / "artifacts" /
                            "aesthetic_calibrator.yaml")
    if calib is not None:
        avg = calib.apply(avg)
    reasons = ["Good composition and lighting"] if avg >= 65 else [
        "Photo aesthetics could be improved"]
    return AestheticFeatureResult(score=avg, reasons=reasons)
