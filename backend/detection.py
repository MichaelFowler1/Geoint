# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

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
    if s.detector_model.startswith("models/") and not Path(s.detector_model).is_dir():
        raise DetectorUnavailable(
            f"{s.detector_model} is missing. Run: python fetch_detector.py"
        )
    device = _resolve_device(s.detector_device)
    # Pinned to a commit so a changed upload can't swap the weights underneath.
    processor = AutoImageProcessor.from_pretrained(
        s.detector_model, revision=s.detector_revision
    )
    model = AutoModelForObjectDetection.from_pretrained(
        s.detector_model, revision=s.detector_revision, use_safetensors=True
    )
    model.to(device).eval()
    _model = (processor, model, device, _load_calibration(s.detector_model))
    logger.info("Loaded detector %s on %s", s.detector_model, device)
    return _model


def _load_calibration(model_dir: str) -> dict | None:
    """Per-class score calibration shipped beside a local checkpoint, if any.

    ``calibration.json`` maps each class's raw score to the precision seen at
    that score on held-out data, as matching breakpoints. The aerial
    fine-tune needs it: its raw scores rank well but run low (a clear plane
    scores about 0.07), so a threshold on them means little on its own.
    """
    path = Path(model_dir) / "calibration.json"
    if not path.is_file():
        return None
    classes = json.loads(path.read_text(encoding="utf-8"))["classes"]
    return {name: (c["raw"], c["calibrated"]) for name, c in classes.items()}


def calibrate(
    scores: list[float], labels: list[str], table: dict | None
) -> list[float]:
    """Pure helper: raw scores -> calibrated ones, class by class. (Unit-tested.)

    Linear between breakpoints and flat beyond them. A class the table does
    not cover keeps its raw score, and so does everything when there is no
    table.
    """
    if not table:
        return list(scores)
    import numpy as np  # noqa: PLC0415

    out = []
    for score, label in zip(scores, labels):
        if label in table:
            xs, ys = table[label]
            score = float(np.interp(score, xs, ys))
        out.append(score)
    return out


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


def tile_origins(
    width: int, height: int, tile: int, overlap: int
) -> list[tuple[int, int]]:
    """Pure helper: top-left corners of overlapping tiles covering an image.

    The last row and column are pulled back to end on the image edge rather
    than hanging off it. An image no bigger than a tile is one tile. (Unit-tested.)
    """

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        return [*range(0, n - tile, tile - overlap), n - tile]

    return [(x, y) for y in starts(height) for x in starts(width)]


def run_detection(image_path: str) -> list[Detection]:
    """Run the object detector (CUDA on the RTX 3080 when available).

    Any Hugging Face object-detection checkpoint works through DETECTOR_MODEL,
    so the detector stays swappable. Boxes are axis-aligned pixel xyxy.

    A large scene is cut into overlapping DETECTOR_TILE-pixel tiles at full
    resolution, because shrinking a whole 2000px scene to the model's input
    size leaves an airliner a few pixels long. Duplicates where tiles overlap
    are merged by non-maximum suppression.
    """
    import torch  # noqa: PLC0415
    from torchvision.ops import batched_nms  # noqa: PLC0415

    s = get_settings()
    processor, model, device, table = _load_model()
    # With a calibration table the cut is made on calibrated scores, after
    # mapping, so keep every candidate the table can speak for until then.
    raw_floor = min(raw[0] for raw, _ in table.values()) if table else s.detection_conf
    image = _read_rgb(image_path)
    tile = s.detector_tile or max(image.width, image.height)
    origins = tile_origins(image.width, image.height, tile, s.detector_tile_overlap)
    scores, labels, boxes = [], [], []
    for i in range(0, len(origins), 8):
        batch = origins[i : i + 8]
        crops = [image.crop((x, y, x + tile, y + tile)) for x, y in batch]
        # A crop that runs past a small image's edge is padded black by PIL.
        inputs = processor(images=crops, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs,
            threshold=raw_floor,
            target_sizes=[(tile, tile)] * len(crops),
        )
        for (x, y), r in zip(batch, results):
            scores.append(r["scores"].float().cpu())
            labels.append(r["labels"].cpu())
            boxes.append(r["boxes"].float().cpu() + torch.tensor([x, y, x, y]))
    scores, labels, boxes = torch.cat(scores), torch.cat(labels), torch.cat(boxes)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image.width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image.height)
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
    detections = to_detections(
        scores[keep].tolist(),
        labels[keep].tolist(),
        boxes[keep].tolist(),
        model.config.id2label,
    )
    calibrated = calibrate(
        [d.confidence for d in detections], [d.label for d in detections], table
    )
    for d, c in zip(detections, calibrated):
        d.confidence = c
    return [d for d in detections if d.confidence >= s.detection_conf]


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
