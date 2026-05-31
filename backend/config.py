from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OpenAI — scene assessment + SITREP generation ("the rest").
    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o"
    openai_report_model: str = "gpt-4o"

    # OpenSky — live ADS-B aircraft feed.
    opensky_client_id: str = ""
    opensky_client_secret: str = ""
    # Feed bounding box: "min_lat,max_lat,min_lon,max_lon" (default: DC area).
    track_bbox: str = "38.7,39.1,-77.3,-76.8"
    track_poll_seconds: int = 12

    # Detector — runs on the RTX 3080. "auto" -> CUDA when available, else CPU.
    detector_model: str = "rtdetr-l.pt"
    detector_device: str = "auto"
    detection_conf: float = 0.25

    # Storage — PostGIS if DATABASE_URL is set, else a local SQLite file.
    database_url: str = ""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        parts = [float(x) for x in self.track_bbox.split(",")]
        return (parts[0], parts[1], parts[2], parts[3])


@lru_cache
def get_settings() -> Settings:
    return Settings()
