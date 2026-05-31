"""Fetch an app-ready NAIP GeoTIFF clip over a point (continental US only).

Usage:
    python fetch_naip.py                    # defaults to LAX
    python fetch_naip.py 40.6413 -73.7781   # any "lat lon" (e.g. JFK)
"""

from __future__ import annotations

import sys

import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

LAT, LON = 33.9416, -118.4085  # LAX terminals (default)
HALF_M = 600  # half-width of the clip in meters (~1.2 km box)
OUT = "naip_clip.tif"

if len(sys.argv) >= 3:
    LAT, LON = float(sys.argv[1]), float(sys.argv[2])

# 1. Find the most recent NAIP tile covering the point.
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
items = sorted(
    catalog.search(
        collections=["naip"],
        intersects={"type": "Point", "coordinates": [LON, LAT]},
    ).items(),
    key=lambda i: i.datetime,
    reverse=True,
)
if not items:
    sys.exit("No NAIP imagery covers that point (NAIP is continental US only).")
item = items[0]
print(f"Using NAIP tile from {item.datetime.date()}")

# 2. Read a window around the point and keep only the RGB bands.
with rasterio.open(item.assets["image"].href) as src:
    tx = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    cx, cy = tx.transform(LON, LAT)
    win = from_bounds(cx - HALF_M, cy - HALF_M, cx + HALF_M, cy + HALF_M, src.transform)
    rgb = src.read([1, 2, 3], window=win)
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=3,
        height=rgb.shape[1],
        width=rgb.shape[2],
        transform=src.window_transform(win),
        photometric="RGB",
    )
    profile.pop("nodata", None)

with rasterio.open(OUT, "w", **profile) as dst:
    dst.write(rgb)
print(f"Saved {OUT}  ({rgb.shape[2]}x{rgb.shape[1]} px) — upload this in the app.")
