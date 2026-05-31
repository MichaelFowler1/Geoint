from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .config import get_settings
from .schemas import Detection

logger = logging.getLogger("geoint.db")

_SQLITE_PATH = Path("geoint.db")


def _use_postgis() -> bool:
    return bool(get_settings().database_url)


def init_db() -> None:
    if _use_postgis():
        _init_postgis()
    else:
        _init_sqlite()


def _init_sqlite() -> None:
    con = sqlite3.connect(_SQLITE_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            label TEXT,
            confidence REAL,
            lat REAL,
            lon REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.commit()
    con.close()


def _pg_conn():
    import psycopg  # noqa: PLC0415

    return psycopg.connect(get_settings().database_url)


def _init_postgis() -> None:
    try:
        with _pg_conn() as con, con.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    image_id TEXT,
                    label TEXT,
                    confidence REAL,
                    geom geometry(Point, 4326),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            con.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("PostGIS init failed (%s); is the db service up?", exc)


def save_detections(image_id: str, detections: list[Detection]) -> None:
    geo = [d for d in detections if d.lat is not None and d.lon is not None]
    if not geo:
        return
    if _use_postgis():
        try:
            with _pg_conn() as con, con.cursor() as cur:
                for d in geo:
                    cur.execute(
                        "INSERT INTO detections (image_id, label, confidence, geom) "
                        "VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
                        (image_id, d.label, d.confidence, d.lon, d.lat),
                    )
                con.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_detections (postgis) failed: %s", exc)
    else:
        con = sqlite3.connect(_SQLITE_PATH)
        con.executemany(
            "INSERT INTO detections (image_id, label, confidence, lat, lon) "
            "VALUES (?, ?, ?, ?, ?)",
            [(image_id, d.label, d.confidence, d.lat, d.lon) for d in geo],
        )
        con.commit()
        con.close()


def recent_detections(limit: int = 200) -> list[dict]:
    if _use_postgis():
        try:
            with _pg_conn() as con, con.cursor() as cur:
                cur.execute(
                    "SELECT image_id, label, confidence, ST_Y(geom), ST_X(geom) "
                    "FROM detections ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("recent_detections (postgis) failed: %s", exc)
            return []
    else:
        con = sqlite3.connect(_SQLITE_PATH)
        rows = con.execute(
            "SELECT image_id, label, confidence, lat, lon FROM detections "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
    return [
        {
            "image_id": r[0],
            "label": r[1],
            "confidence": r[2],
            "lat": r[3],
            "lon": r[4],
        }
        for r in rows
    ]
