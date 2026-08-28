"""Input data schemas for listings and photos."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ListingRecord(BaseModel):
    listing_id: str
    vertical: str = ""
    name: str = ""
    description: str = ""
    city: str = ""
    country: str = ""
    capacity: int = 0
    bedrooms: int = 0
    bathrooms: float = 0
    base_price: float = 0
    currency: str = ""
    instant_booking: bool = False
    is_calendar_synced: bool = False
    has_active_ical: bool = False
    pricing_days_180: int = 0
    blocked_days_180: int = 0
    amenities: list[str] = Field(default_factory=list)
    photo_count: int = 0
    property_type: str = ""
    status: str = ""


class PhotoRecord(BaseModel):
    entity_id: str
    file: str
    seq: int = 0
    is_cover: bool = False
    width: int = 0
    height: int = 0
    bytes: int = 0
