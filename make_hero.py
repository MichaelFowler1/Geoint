#!/usr/bin/env python3
"""
Generate docs/hero.png - the README image.

Runs the project's real detection pipeline (Ultralytics OBB on CUDA) on the
bundled public-domain NAIP GeoTIFF and renders the oriented bounding boxes,
class counts, and the scene's geodetic center - the core of what GEOINT-COP
does: overhead imagery -> GPU detector -> georeferenced detections -> report.

Run:  python make_hero.py            (uses yolo26m-obb.pt, conf 0.20)
"""
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from ultralytics import YOLO

TIF, MODEL, CONF = "naip_clip.tif", "yolo26m-obb.pt", 0.20
IMGSZ = 2048   # infer at native resolution; the 640px default misses small aircraft
BG, INK, DIM, ACC = "#070b12", "#d7e2f0", "#6b7d95", "#ffb020"

# --- imagery + georeference ---
with rasterio.open(TIF) as ds:
    rgb = np.transpose(ds.read([1, 2, 3]), (1, 2, 0)).astype("uint8")
    cx, cy = ds.width / 2, ds.height / 2
    wx, wy = ds.transform * (cx, cy)
    lon, lat = (v[0] for v in warp_transform(ds.crs, "EPSG:4326", [wx], [wy]))

# --- real detection on the GPU ---
model = YOLO(MODEL)
r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, device="cuda", verbose=False)[0]
obb = r.obb
labels = [r.names[int(c)] for c in obb.cls]
counts = Counter(labels)
polys = obb.xyxyxyxy.cpu().numpy()          # (n, 4, 2) oriented corners
confs = obb.conf.cpu().numpy()

# --- figure ---
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig = plt.figure(figsize=(11, 8.6), facecolor=BG)

fig.text(0.05, 0.960, "GEOINT-COP  ·  COMMON OPERATING PICTURE",
         fontsize=16, fontweight="bold")
n_planes = counts.get("plane", 0)
n_other = len(polys) - n_planes
headline = f"{n_planes} aircraft detected" + (f"  +{n_other} vehicles" if n_other else "")
fig.text(0.95, 0.960, headline, ha="right",
         fontsize=14, fontweight="bold", color=ACC)
fig.text(0.05, 0.922, "Overhead imagery  →  GPU detector (RT-DETR / YOLO-OBB)  →  "
                      "georeferenced detections  →  AI SITREP  +  live ADS-B tracks",
         fontsize=10, color=DIM)
fig.text(0.05, 0.012, f"scene center {lat:.4f}°N {abs(lon):.4f}°W   ·   "
                      f"NAIP imagery (USDA, public domain)   ·   inference on CUDA",
         fontsize=8.5, color=DIM)

ax = fig.add_axes([0.07, 0.045, 0.86, 0.855])
ax.imshow(rgb)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor("#1b2740"); s.set_linewidth(1.5)

for poly, conf in zip(polys, confs):
    ax.add_patch(Polygon(poly, closed=True, fill=False,
                         edgecolor=ACC, linewidth=1.6))
    x, y = poly[:, 0].min(), poly[:, 1].min()
    ax.text(x, y - 6, f"{conf:.0%}", fontsize=6.5, color=ACC, fontweight="bold")

# counts box
lines = [f"{v} × {k}" for k, v in counts.most_common()]
ax.text(0.985, 0.02, "DETECTIONS\n" + "\n".join(lines),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
        fontweight="bold", color=INK,
        bbox=dict(facecolor=BG, alpha=0.85, edgecolor=ACC, boxstyle="round,pad=0.5"))

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/hero.png", dpi=130, facecolor=BG)
print(f"[+] wrote docs/hero.png - {len(polys)} detections: {dict(counts)}")
print(f"    scene center: {lat:.5f}, {lon:.5f}")
