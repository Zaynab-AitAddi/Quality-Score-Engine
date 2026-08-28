"""Technical photo quality: blur, exposure, brightness, resolution, duplicates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from quality_engine.config import load_thresholds


@dataclass
class PhotoTechnicalMetrics:
    sharpness: float = 0.0
    is_blurry: bool = False
    brightness: float = 0.0
    exposure_score: float = 0.0
    megapixels: float = 0.0
    resolution_score: float = 0.0
    width: int = 0
    height: int = 0
    valid: bool = True
    error: str | None = None


@dataclass
class TechnicalFeatureResult:
    score: float
    reasons: list[str] = field(default_factory=list)
    blurry_count: int = 0
    duplicate_groups: int = 0
    photo_metrics: list[PhotoTechnicalMetrics] = field(default_factory=list)


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0) * 100)


def analyze_photo(path: Path) -> PhotoTechnicalMetrics:
    thresholds = load_thresholds()
    blur_cfg = thresholds["blur"]
    exposure_cfg = thresholds["exposure"]
    res_cfg = thresholds["resolution"]

    metrics = PhotoTechnicalMetrics()
    try:
        img = cv2.imread(str(path))
        if img is None:
            metrics.valid = False
            metrics.error = "unreadable"
            return metrics

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        metrics.width = w
        metrics.height = h
        metrics.megapixels = (w * h) / 1_000_000

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics.sharpness = float(lap_var)
        metrics.is_blurry = lap_var < blur_cfg["blurry_threshold"]

        metrics.brightness = float(np.mean(gray))
        dark = exposure_cfg["dark_threshold"]
        bright = exposure_cfg["bright_threshold"]
        if metrics.brightness < dark:
            metrics.exposure_score = _normalize(metrics.brightness, 0, dark)
        elif metrics.brightness > bright:
            metrics.exposure_score = _normalize(255 - metrics.brightness, 0, 255 - bright)
        else:
            metrics.exposure_score = 100.0

        metrics.resolution_score = _normalize(
            metrics.megapixels,
            res_cfg["min_megapixels"],
            res_cfg["good_megapixels"],
        )
    except Exception as exc:  # noqa: BLE001
        metrics.valid = False
        metrics.error = str(exc)
    return metrics


def _count_duplicate_groups(paths: list[Path], threshold: int) -> int:
    hashes: list[imagehash.ImageHash] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                hashes.append(imagehash.phash(img))
        except Exception:  # noqa: BLE001
            continue

    if len(hashes) < 2:
        return 0

    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i] - hashes[j] <= threshold:
                union(i, j)

    roots = {find(i) for i in range(len(hashes))}
    # Groups with more than one member are duplicate clusters
    root_counts: dict[int, int] = {}
    for i in range(len(hashes)):
        root = find(i)
        root_counts[root] = root_counts.get(root, 0) + 1
    return sum(1 for count in root_counts.values() if count > 1)


def score_technical_quality(photo_paths: list[Path]) -> TechnicalFeatureResult:
    thresholds = load_thresholds()
    reasons: list[str] = []

    if not photo_paths:
        return TechnicalFeatureResult(score=0.0, reasons=["No photos available"])

    metrics_list: list[PhotoTechnicalMetrics] = []
    for path in photo_paths:
        if path.exists():
            metrics_list.append(analyze_photo(path))
        else:
            metrics_list.append(PhotoTechnicalMetrics(valid=False, error="missing"))

    valid = [m for m in metrics_list if m.valid]
    if not valid:
        return TechnicalFeatureResult(
            score=0.0,
            reasons=["No valid photos could be processed"],
            photo_metrics=metrics_list,
        )

    blurry_count = sum(1 for m in valid if m.is_blurry)
    avg_sharpness = np.mean([m.sharpness for m in valid])
    avg_exposure = np.mean([m.exposure_score for m in valid])
    avg_resolution = np.mean([m.resolution_score for m in valid])

    blur_cfg = thresholds["blur"]
    sharpness_score = _normalize(
        avg_sharpness,
        blur_cfg["blurry_threshold"],
        blur_cfg["sharp_threshold"],
    )

    duplicate_groups = _count_duplicate_groups(
        [p for p in photo_paths if p.exists()],
        thresholds["duplicates"]["hash_threshold"],
    )

    # Weight sub-signals within technical module
    component = 0.35 * sharpness_score + 0.25 * avg_exposure + 0.25 * avg_resolution
    duplicate_penalty = min(duplicate_groups * 8, 25)
    component = max(0.0, component - duplicate_penalty)

    if blurry_count:
        reasons.append(f"{blurry_count} blurry photo(s)")
    if duplicate_groups:
        reasons.append(f"{duplicate_groups} near-duplicate photo group(s)")
    low_res = sum(1 for m in valid if m.resolution_score < 40)
    if low_res:
        reasons.append(f"{low_res} low-resolution photo(s)")
    if not reasons:
        reasons.append("Good technical photo quality")

    return TechnicalFeatureResult(
        score=float(np.clip(component, 0, 100)),
        reasons=reasons,
        blurry_count=blurry_count,
        duplicate_groups=duplicate_groups,
        photo_metrics=metrics_list,
    )
