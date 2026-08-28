"""Zero-shot room type classification via CLIP (with CPU fallback)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from quality_engine.config import load_room_labels, load_thresholds

EXPECTED_ROOMS = ["bedroom", "bathroom", "kitchen", "living", "exterior", "pool", "view"]


@dataclass
class RoomPrediction:
    path: Path
    label: str
    confidence: float


@dataclass
class RoomCoverageResult:
    score: float
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    predictions: list[RoomPrediction] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@lru_cache
def _get_clip_model():
    try:
        import open_clip
        import torch

        cfg = load_room_labels()
        model_cfg = __import__("quality_engine.config", fromlist=["load_models_config"]).load_models_config()
        clip_cfg = model_cfg.get("clip", {})
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
        )
        device = clip_cfg.get("device", "cpu")
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(clip_cfg.get("model_name", "ViT-B-32"))

        labels = cfg["room_labels"]
        prompts = [cfg["room_prompts"][label] for label in labels]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return model, preprocess, text_features, labels, device, torch
    except Exception:  # noqa: BLE001
        return None


def _classify_with_clip(path: Path) -> RoomPrediction | None:
    bundle = _get_clip_model()
    if bundle is None:
        return None
    model, preprocess, text_features, labels, device, torch = bundle
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return RoomPrediction(path=path, label=labels[idx], confidence=float(probs[idx]))
    except Exception:  # noqa: BLE001
        return None


def _classify_fallback(path: Path) -> RoomPrediction:
    """Filename/path heuristic when CLIP is unavailable."""
    name = path.name.lower()
    if "cover" in name or "exterior" in name:
        return RoomPrediction(path=path, label="exterior", confidence=0.4)
    return RoomPrediction(path=path, label="other", confidence=0.2)


def classify_photo(path: Path) -> RoomPrediction:
    pred = _classify_with_clip(path)
    if pred is not None:
        return pred
    return _classify_fallback(path)


def score_room_coverage(photo_paths: list[Path], bedrooms: int = 1) -> RoomCoverageResult:
    thresholds = load_thresholds()
    conf_threshold = thresholds["room"]["confidence_threshold"]

    if not photo_paths:
        return RoomCoverageResult(
            score=0.0,
            missing=EXPECTED_ROOMS.copy(),
            reasons=["No photos for room coverage"],
        )

    predictions: list[RoomPrediction] = []
    for path in photo_paths:
        if path.exists():
            pred = classify_photo(path)
            if pred.confidence >= conf_threshold or pred.label != "other":
                predictions.append(pred)

    covered_set = {p.label for p in predictions if p.label in EXPECTED_ROOMS}
    missing = [room for room in EXPECTED_ROOMS if room not in covered_set]

    # Bathroom is critical; bedroom expected for multi-bedroom listings
    required = {"bathroom", "kitchen", "living", "exterior"}
    if bedrooms >= 1:
        required.add("bedroom")

    covered_required = len(required & covered_set)
    score = (covered_required / len(required)) * 100 if required else 0.0

    reasons: list[str] = []
    for room in missing:
        if room in required:
            reasons.append(f"No {room} photo detected")
    if not reasons:
        reasons.append("Good room coverage")

    return RoomCoverageResult(
        score=float(score),
        covered=sorted(covered_set),
        missing=missing,
        predictions=predictions,
        reasons=reasons,
    )
