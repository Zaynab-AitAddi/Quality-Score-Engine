"""Output schemas for quality scoring results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComponentScores(BaseModel):
    photo_technical: float = 0.0
    photo_aesthetic: float = 0.0
    room_coverage: float = 0.0
    amenity_check: float = 0.0
    text_quality: float = 0.0
    completeness: float = 0.0


class Explanation(BaseModel):
    positive: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class QualityResult(BaseModel):
    listing_id: str
    quality_score: float
    components: ComponentScores
    explanation: Explanation
    version: str = "quality-v1"
    quality_engine_version: str = "1.0.0"
    config_version: str = "quality-config-v1"
    photo_status: str = "ok"
