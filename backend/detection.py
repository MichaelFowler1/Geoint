from __future__ import annotations

import logging
from collections import Counter

from .config import get_settings
from .schemas import Detection

logger = logging.getLogger("geoint.detection")

_model = None  # cached detector instance


class DetectorUnavailable(RuntimeError):
    """Raised when the ML extras / model are not installed."""


def summarize_counts(detections: list[Detection]) -> dict[str, int]:
    """Pure helper: tally detections by label. (Unit-tested.)"""
    return dict(Counter(d.label for d in detections))


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _load_model():
    global _model
    if _model is not None:
        return _model
    s = get_settings()
    try:
        if "rtdetr" in s.detector_model.lower():
            from ultralytics import RTDETR  # noqa: PLC0415

            _model = RTDETR(s.detector_model)
        else:
            from ultralytics import YOLO  # noqa: PLC0415

            _model = YOLO(s.detector_model)
    except ImportError as exc:
        raise DetectorUnavailable(
            "Detector requires ML extras. Install: pip install -r requirements-ml.txt"
        ) from exc
    logger.info(
        "Loaded detector %s on %s",
        s.detector_model,
        _resolve_device(s.detector_device),
    )
    return _model


def run_detection(image_path: str) -> list[Detection]:
    """Run the object detector (CUDA on the RTX 3080 when available).

    Handles both standard axis-aligned models (COCO) and oriented-bounding-box
    models such as the DOTA-pretrained ``*-obb`` weights, so the detector stays
    swappable via DETECTOR_MODEL.
    """
    s = get_settings()
    model = _load_model()
    device = _resolve_device(s.detector_device)
    results = model.predict(
        image_path, conf=s.detection_conf, device=device, verbose=False
    )
    detections: list[Detection] = []
    for r in results:
        names = r.names
        obb = getattr(r, "obb", None)
        if obb is not None and len(obb) > 0:  # aerial OBB models (DOTA)
            for i in range(len(obb)):
                cls_id = int(obb.cls[i])
                detections.append(
                    Detection(
                        label=names.get(cls_id, str(cls_id)),
                        confidence=float(obb.conf[i]),
                        bbox_px=[float(v) for v in obb.xyxy[i].tolist()],
                    )
                )
        elif r.boxes is not None:  # standard COCO models
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(
                        label=names.get(cls_id, str(cls_id)),
                        confidence=float(box.conf[0]),
                        bbox_px=[float(v) for v in box.xyxy[0].tolist()],
                    )
                )
    return detections


def georeference(image_path: str, detections: list[Detection]) -> bool:
    """Attach lat/lon to each detection via the image's geotransform.

    Returns True if the image carried geospatial metadata (e.g. a GeoTIFF). For a
    plain image chip (DOTA/xView), detections are returned without coordinates.
    """
    try:
        import rasterio  # noqa: PLC0415
        from rasterio.warp import transform as warp_transform  # noqa: PLC0415
    except ImportError:
        logger.info("rasterio not installed; skipping georeferencing")
        return False
    try:
        with rasterio.open(image_path) as ds:
            if ds.crs is None:
                return False
            for det in detections:
                x1, y1, x2, y2 = det.bbox_px
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                world_x, world_y = ds.transform * (cx, cy)
                lon, lat = warp_transform(ds.crs, "EPSG:4326", [world_x], [world_y])
                det.lon, det.lat = lon[0], lat[0]
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Georeferencing failed: %s", exc)
        return False


def export_web_tile(image_path: str, out_png: str) -> list[list[float]] | None:
    """Write a web-displayable PNG of a georeferenced image and return its bounds.

    Returns lat/lon bounds as ``[[south, west], [north, east]]`` for a Leaflet
    image overlay, or ``None`` for non-georeferenced images (no overlay possible).
    """
    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from rasterio.warp import transform_bounds  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with rasterio.open(image_path) as ds:
            if ds.crs is None:
                return None
            west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            rgb = ds.read([1, 2, 3])
        arr = np.transpose(rgb, (1, 2, 0)).astype("uint8")
        img = Image.fromarray(arr, mode="RGB")
        img.thumbnail((2048, 2048))
        img.save(out_png, format="PNG")
        return [[south, west], [north, east]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("web tile export failed: %s", exc)
        return None
