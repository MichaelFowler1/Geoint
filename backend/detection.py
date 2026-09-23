# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

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
    """Load the Hugging Face detector once: (processor, model, device)."""
    global _model
    if _model is not None:
        return _model
    s = get_settings()
    try:
        from transformers import (  # noqa: PLC0415
            AutoImageProcessor,
            AutoModelForObjectDetection,
        )
    except ImportError as exc:
        raise DetectorUnavailable(
            "Detector requires ML extras. Install: pip install -r requirements-ml.txt"
        ) from exc
    device = _resolve_device(s.detector_device)
    # Pinned to a commit so a changed upload can't swap the weights underneath.
    processor = AutoImageProcessor.from_pretrained(
        s.detector_model, revision=s.detector_revision
    )
    model = AutoModelForObjectDetection.from_pretrained(
        s.detector_model, revision=s.detector_revision, use_safetensors=True
    )
    model.to(device).eval()
    _model = (processor, model, device)
    logger.info("Loaded detector %s on %s", s.detector_model, device)
    return _model


def _read_rgb(image_path: str):
    """Open an image as 8-bit RGB. GeoTIFFs go through rasterio (bands 1-3),
    so a 4-band NAIP scene loses its near-infrared band rather than failing."""
    from PIL import Image  # noqa: PLC0415

    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415

        with rasterio.open(image_path) as ds:
            if ds.count >= 3 and ds.dtypes[0] == "uint8":
                rgb = np.transpose(ds.read([1, 2, 3]), (1, 2, 0))
                return Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    except Exception:  # noqa: BLE001, S110 - fall back to PIL for anything else
        pass
    with Image.open(image_path) as img:
        return img.convert("RGB")


def to_detections(
    scores: list[float],
    labels: list[int],
    boxes: list[list[float]],
    id2label: dict[int, str],
) -> list[Detection]:
    """Pure helper: post-processed detector output -> Detection list. (Unit-tested.)"""
    return [
        Detection(
            label=id2label.get(int(cls_id), str(int(cls_id))),
            confidence=float(score),
            bbox_px=[float(v) for v in box],
        )
        for score, cls_id, box in zip(scores, labels, boxes)
    ]


def run_detection(image_path: str) -> list[Detection]:
    """Run the object detector (CUDA on the RTX 3080 when available).

    Any Hugging Face object-detection checkpoint works through DETECTOR_MODEL,
    so the detector stays swappable. Boxes are axis-aligned pixel xyxy.
    """
    import torch  # noqa: PLC0415

    s = get_settings()
    processor, model, device = _load_model()
    image = _read_rgb(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs,
        threshold=s.detection_conf,
        target_sizes=[(image.height, image.width)],
    )[0]
    return to_detections(
        result["scores"].tolist(),
        result["labels"].tolist(),
        result["boxes"].tolist(),
        model.config.id2label,
    )


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
