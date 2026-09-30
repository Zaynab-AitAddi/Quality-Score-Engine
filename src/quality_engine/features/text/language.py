"""Text quality: language coverage, title, description length."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from quality_engine.config import load_thresholds
from quality_engine.schemas.listing import ListingRecord


@dataclass
class TextQualityResult:
    score: float
    languages: dict[str, bool] = field(default_factory=dict)
    word_count: int = 0
    reasons: list[str] = field(default_factory=list)


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
FRENCH_HINTS = re.compile(r"\b(le|la|les|des|une|un|et|dans|pour|avec|chambre|appartement)\b", re.IGNORECASE)
ENGLISH_HINTS = re.compile(r"\b(the|and|with|room|bedroom|apartment|kitchen|pool|wifi)\b", re.IGNORECASE)


def detect_languages(text: str) -> dict[str, bool]:
    text = text or ""
    return {
        "arabic": bool(ARABIC_RE.search(text)),
        "french": bool(FRENCH_HINTS.search(text)),
        "english": bool(ENGLISH_HINTS.search(text)),
    }


def score_text_quality(listing: ListingRecord) -> TextQualityResult:
    thresholds = load_thresholds()["text"]
    combined = f"{listing.name}\n{listing.description}".strip()
    languages = detect_languages(combined)
    words = combined.split()
    word_count = len(words)

    lang_score = sum(languages.values()) / 3 * 100

    min_w = thresholds["min_description_words"]
    good_w = thresholds["good_description_words"]
    exc_w = thresholds["excellent_description_words"]
    if word_count < min_w:
        length_score = 30.0
    elif word_count < good_w:
        length_score = 60.0
    elif word_count < exc_w:
        length_score = 85.0
    else:
        length_score = 95.0

    title = listing.name.strip()
    min_title = thresholds["min_title_length"]
    if not title:
        title_score = 0.0
    elif len(title) < min_title:
        title_score = 40.0
    else:
        title_score = 100.0

    score = 0.4 * lang_score + 0.4 * length_score + 0.2 * title_score
    reasons: list[str] = []

    for lang, present in languages.items():
        if present:
            reasons.append(f"{lang.capitalize()} content detected")
        else:
            reasons.append(f"{lang.capitalize()} description missing")

    if word_count >= good_w:
        reasons.append("Good description length")
    elif word_count >= min_w:
        reasons.append("Description length is moderate")
    else:
        reasons.append("Description is too short")

    if not title:
        reasons.append("Title is empty")
    elif len(title) < min_title:
        reasons.append("Title is very short")

    return TextQualityResult(score=float(score), languages=languages, word_count=word_count, reasons=reasons)
