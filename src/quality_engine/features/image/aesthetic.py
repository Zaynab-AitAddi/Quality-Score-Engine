"""Aesthetic quality using a lightweight heuristic (calibration-ready)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quality_engine.calibration import load_calibrator
from quality_engine.config import PROJECT_ROOT
from quality_engine.features.image.technical import PhotoTechnicalMetrics, analyze_photo


@dataclass
class AestheticFeatureResult:
    score: float
    reasons: list[str] = field(default_factory=list)


def _aesthetic_heuristic(
    path: Path,
    metrics: PhotoTechnicalMetrics | None = None,
) -> float:
    """Proxy score until human-rated calibration dataset is applied."""
    metrics = metrics or analyze_photo(path)
    if not metrics.valid:
        return 0.0

    # Colorfulness + contrast + sharpness proxy
    sharp = min(metrics.sharpness / 150.0, 1.0)

    raw = 0.35 * min(metrics.colorfulness / 40.0, 1.0) + 0.35 * \
        min(metrics.contrast / 60.0, 1.0) + 0.30 * sharp
    return float(np.clip(raw * 100, 0, 100))


def score_aesthetic_quality(
    photo_paths: list[Path],
    photo_metrics: list[PhotoTechnicalMetrics] | None = None,
) -> AestheticFeatureResult:
    if not photo_paths:
        return AestheticFeatureResult(score=0.0, reasons=["No photos for aesthetic scoring"])

    scores = [
        _aesthetic_heuristic(
            path,
            photo_metrics[index] if photo_metrics is not None and index < len(
                photo_metrics) else None,
        )
        for index, path in enumerate(photo_paths)
        if path.exists()
    ]
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
