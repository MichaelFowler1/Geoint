from __future__ import annotations

import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import db, detection, reporting
from .config import get_settings
from .schemas import DetectionResult
from .tracking import hub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geoint")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
TILES_DIR = Path(__file__).resolve().parent.parent / "tiles"
TILES_DIR.mkdir(exist_ok=True)

_metrics = {"detections_run": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    await hub.start()
    yield
    await hub.stop()


app = FastAPI(title="GEOINT-COP", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "components": {
            "openai_configured": bool(s.openai_api_key),
            "storage": "postgis" if s.database_url else "sqlite",
            "live_tracking_polls": hub.poll_count,
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    # Minimal Prometheus exposition. Swap for prometheus-fastapi-instrumentator.
    return (
        f"geoint_detections_run_total {_metrics['detections_run']}\n"
        f"geoint_track_polls_total {hub.poll_count}\n"
        f"geoint_live_aircraft {len(hub.latest)}\n"
    )


@app.post("/api/detect", response_model=DetectionResult)
async def detect(
    image: UploadFile = File(...),
    location: str | None = Form(default=None),
):
    image_id = uuid.uuid4().hex[:12]
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        detections = detection.run_detection(tmp_path)  # RTX 3080 / CUDA
    except detection.DetectorUnavailable as exc:
        Path(tmp_path).unlink(missing_ok=True)
        return DetectionResult(
            image_id=image_id,
            georeferenced=False,
            counts={},
            detections=[],
            scene_assessment=str(exc),
            sitrep="Detector not installed - see requirements-ml.txt.",
        )

    georeferenced = detection.georeference(tmp_path, detections)
    bounds = detection.export_web_tile(tmp_path, str(TILES_DIR / f"{image_id}.png"))
    tile_url = f"/tiles/{image_id}.png" if bounds else None
    counts = detection.summarize_counts(detections)
    scene = reporting.scene_assessment(tmp_path)  # OpenAI ("the rest")
    sitrep = reporting.generate_sitrep(counts, scene, location)  # OpenAI

    db.save_detections(image_id, detections)
    _metrics["detections_run"] += 1
    Path(tmp_path).unlink(missing_ok=True)

    return DetectionResult(
        image_id=image_id,
        georeferenced=georeferenced,
        counts=counts,
        detections=detections,
        scene_assessment=scene,
        sitrep=sitrep,
        tile_url=tile_url,
        bounds=bounds,
    )


@app.get("/api/detections/recent")
async def recent() -> list[dict]:
    return db.recent_detections()


@app.websocket("/ws/tracks")
async def ws_tracks(ws: WebSocket):
    await ws.accept()
    await hub.register(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive pings from the client
    except WebSocketDisconnect:
        hub.unregister(ws)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# Static assets (app.js, style.css).
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/tiles", StaticFiles(directory=TILES_DIR), name="tiles")
