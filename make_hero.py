#!/usr/bin/env python3
# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Generate docs/hero.png - the README image.

Runs the project's own detection pipeline (backend.detection: tiled RT-DETR,
then georeferencing) on the bundled public-domain NAIP GeoTIFF and draws the
boxes, class counts, and the scene's geodetic center: overhead imagery ->
GPU detector -> georeferenced detections -> report.

Run:  DETECTOR_MODEL=<aerial checkpoint> python make_hero.py
      (DETECTOR_REVISION too when the checkpoint is on the Hugging Face Hub)
"""

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from backend import detection
from backend.config import get_settings

TIF = "naip_clip.tif"
BG, INK, DIM, ACC = "#070b12", "#d7e2f0", "#6b7d95", "#ffb020"

# --- imagery + georeference ---
with rasterio.open(TIF) as ds:
    rgb = np.transpose(ds.read([1, 2, 3]), (1, 2, 0)).astype("uint8")
    cx, cy = ds.width / 2, ds.height / 2
    wx, wy = ds.transform * (cx, cy)
    lon, lat = (v[0] for v in warp_transform(ds.crs, "EPSG:4326", [wx], [wy]))

# --- real detection, through the same code the API runs ---
dets = detection.run_detection(TIF)
counts = Counter(d.label for d in dets)
device = detection._resolve_device(get_settings().detector_device)

# --- figure ---
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig = plt.figure(figsize=(11, 8.6), facecolor=BG)

fig.text(
    0.05,
    0.960,
    "GEOINT-COP  ·  COMMON OPERATING PICTURE",
    fontsize=16,
    fontweight="bold",
)
n_planes = counts.get("plane", 0)
n_other = len(dets) - n_planes
headline = f"{n_planes} aircraft detected" + (f"  +{n_other} other" if n_other else "")
fig.text(0.95, 0.960, headline, ha="right", fontsize=14, fontweight="bold", color=ACC)
fig.text(
    0.05,
    0.922,
    "Overhead imagery  →  GPU detector (RT-DETR, aerial fine-tune)  →  "
    "georeferenced detections  →  AI SITREP  +  live ADS-B tracks",
    fontsize=10,
    color=DIM,
)
fig.text(
    0.05,
    0.012,
    f"scene center {lat:.4f}°N {abs(lon):.4f}°W   ·   "
    f"NAIP imagery (USDA, public domain)   ·   inference on {device.upper()}",
    fontsize=8.5,
    color=DIM,
)

ax = fig.add_axes([0.07, 0.045, 0.86, 0.855])
ax.imshow(rgb)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor("#1b2740")
    s.set_linewidth(1.5)

for d in dets:
    x1, y1, x2, y2 = d.bbox_px
    color = ACC if d.label == "plane" else "#4cc9f0"
    ax.add_patch(
        Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.4
        )
    )
    if d.label == "plane":
        ax.text(
            x1,
            y1 - 6,
            f"{d.confidence:.0%}",
            fontsize=6.5,
            color=ACC,
            fontweight="bold",
        )

# counts box
lines = [f"{v} × {k}" for k, v in counts.most_common()]
ax.text(
    0.985,
    0.02,
    "DETECTIONS\n" + "\n".join(lines),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color=INK,
    bbox=dict(facecolor=BG, alpha=0.85, edgecolor=ACC, boxstyle="round,pad=0.5"),
)

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/hero.png", dpi=130, facecolor=BG)
print(f"[+] wrote docs/hero.png - {len(dets)} detections: {dict(counts)}")
print(f"    scene center: {lat:.5f}, {lon:.5f}")
