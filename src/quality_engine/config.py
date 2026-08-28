"""Load YAML configuration from the config/ directory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def load_weights() -> dict[str, Any]:
    return _load_yaml("weights.yaml")


@lru_cache
def load_thresholds() -> dict[str, Any]:
    return _load_yaml("thresholds.yaml")


@lru_cache
def load_room_labels() -> dict[str, Any]:
    return _load_yaml("room_labels.yaml")


@lru_cache
def load_models_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


def validate_weights_sum_to_100() -> float:
    weights = load_weights()["weights"]
    total = sum(weights.values())
    if abs(total - 100) > 1e-6:
        raise ValueError(f"Weights must sum to 100, got {total}")
    return total
