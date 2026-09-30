"""Cross-check declared amenities against visual detection in photos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from quality_engine.config import load_room_labels, load_thresholds

AMENITY_ALIASES: dict[str, list[str]] = {
    "pool": ["pool", "swimming pool", "piscine"],
    "air_conditioning": ["air conditioning", "ac", "climatisation", "clim"],
    "tv": ["tv", "television", "télévision"],
    "washing_machine": ["washing machine", "washer", "lave-linge", "laundry"],
    "balcony": ["balcony", "terrace", "balcon", "terrasse"],
    "wifi": ["wifi", "wi-fi", "internet"],
    "parking": ["parking", "garage"],
    "kitchen": ["kitchen", "cuisine"],
    "heating": ["heating", "heater", "chauffage"],
    "hot_tub": ["hot tub", "jacuzzi", "spa"],
}


@dataclass
class AmenityCheckResult:
    score: float
    declared: list[str] = field(default_factory=list)
    detected: list[str] = field(default_factory=list)
    undeclared_detected: list[str] = field(default_factory=list)
    declared_not_detected: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _normalize_amenity(text: str) -> str | None:
    lower = text.lower().strip()
    for key, aliases in AMENITY_ALIASES.items():
        for alias in aliases:
            if alias in lower or lower in alias:
                return key
    return None


def map_declared_amenities(amenities: list[str]) -> set[str]:
    mapped: set[str] = set()
    for item in amenities:
        key = _normalize_amenity(item)
        if key:
            mapped.add(key)
    return mapped


@lru_cache
def _get_amenity_clip():
    if os.getenv("QUALITY_ENGINE_USE_CLIP", "0").lower() not in {"1", "true", "yes", "on"}:
        return None

    try:
        import open_clip
        import torch

        cfg = load_room_labels()
        model_cfg = __import__("quality_engine.config", fromlist=[
                               "load_models_config"]).load_models_config()
        clip_cfg = model_cfg.get("clip", {})
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
            force_quick_gelu=clip_cfg.get("quick_gelu", True),
        )
        device = clip_cfg.get("device", "cpu")
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(
            clip_cfg.get("model_name", "ViT-B-32"))

        labels = cfg["amenity_labels"]
        prompts = [cfg["amenity_prompts"][label] for label in labels]
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return model, preprocess, text_features, labels, device, torch
    except Exception:  # noqa: BLE001
        return None


def detect_amenities_in_photo(path: Path, threshold: float) -> set[str]:
    bundle = _get_amenity_clip()
    if bundle is None:
        return set()

    model, preprocess, text_features, labels, device, torch = bundle
    try:
        image = preprocess(Image.open(path).convert("RGB")
                           ).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @
                     text_features.T).softmax(dim=-1).cpu().numpy()[0]
        detected = {labels[i] for i, p in enumerate(probs) if p >= threshold}
        return detected
    except Exception:  # noqa: BLE001
        return set()


def score_amenity_check(photo_paths: list[Path], declared_amenities: list[str]) -> AmenityCheckResult:
    thresholds = load_thresholds()
    detection_threshold = thresholds["amenity"]["detection_threshold"]
    mild_penalty = thresholds["amenity"]["declared_not_detected_penalty"]

    declared = map_declared_amenities(declared_amenities)
    clip_enabled = os.getenv("QUALITY_ENGINE_USE_CLIP", "0").lower() in {
        "1", "true", "yes", "on"
    }
    vision_available = clip_enabled and _get_amenity_clip() is not None
    has_photos = any(path.exists() for path in photo_paths)
    if not vision_available or not has_photos:
        return AmenityCheckResult(
            score=50.0,
            declared=sorted(declared),
            reasons=["Visual amenity detection unavailable; neutral score used"],
        )

    detected: set[str] = set()
    for path in photo_paths:
        if path.exists():
            detected |= detect_amenities_in_photo(path, detection_threshold)

    matched = declared & detected
    undeclared_detected = detected - declared
    declared_not_detected = declared - detected

    if not declared and not detected:
        score = 50.0
    elif not declared:
        score = 70.0 + min(len(undeclared_detected) * 5, 20)
    else:
        match_ratio = len(matched) / len(declared)
        score = match_ratio * 100
        score -= len(declared_not_detected) * mild_penalty * 10
        score += min(len(undeclared_detected) * 3, 10)

    score = float(np.clip(score, 0, 100))
    reasons: list[str] = []
    for amenity in sorted(matched):
        reasons.append(
            f"{amenity.replace('_', ' ').title()} detected and declared")
    for amenity in sorted(declared_not_detected):
        reasons.append(
            f"{amenity.replace('_', ' ').title()} declared but not visually confirmed")
    for amenity in sorted(undeclared_detected):
        reasons.append(
            f"{amenity.replace('_', ' ').title()} detected but not declared")

    if not reasons:
        reasons.append("Amenity information available")

    return AmenityCheckResult(
        score=score,
        declared=sorted(declared),
        detected=sorted(detected),
        undeclared_detected=sorted(undeclared_detected),
        declared_not_detected=sorted(declared_not_detected),
        reasons=reasons,
    )
