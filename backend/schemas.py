from __future__ import annotations

from pydantic import BaseModel


class Aircraft(BaseModel):
    icao24: str
    callsign: str | None = None
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    velocity_ms: float | None = None
    heading_deg: float | None = None
    on_ground: bool = False
    last_contact: int | None = None


class Detection(BaseModel):
    label: str
    confidence: float
    bbox_px: list[float]  # [x1, y1, x2, y2] in pixels
    lat: float | None = None
    lon: float | None = None


class DetectionResult(BaseModel):
    image_id: str
    georeferenced: bool
    counts: dict[str, int]
    detections: list[Detection]
    scene_assessment: str
    sitrep: str
    tile_url: str | None = None
    bounds: list[list[float]] | None = None