# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Download the aerial detector weights from this repo's GitHub release.

Usage:
    python fetch_detector.py        # -> models/rtdetr-dota-v1

The weights are RT-DETR (PekingU/rtdetr_r50vd, Apache-2.0) fine-tuned on DOTA
v1.0, so they carry DOTA's terms: academic, noncommercial use only. See
MODEL_CARD.md inside the download.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://github.com/MichaelFowler1/Geoint/releases/download/"
    "aerial-detector-v1/rtdetr-dota-v1.zip"
)
SHA256 = "bca5878e6933be0f94eedbcc92e30550ec2fae6095ff7cc9351fd702095b5f68"
OUT = Path("models")
ZIP = OUT / "rtdetr-dota-v1.zip"

OUT.mkdir(exist_ok=True)
if not ZIP.exists():
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, ZIP)  # noqa: S310 - fixed https URL
digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
if digest != SHA256:
    ZIP.unlink()
    sys.exit(f"Checksum mismatch ({digest}); deleted the download, try again.")
with zipfile.ZipFile(ZIP) as zf:
    zf.extractall(OUT)
print(
    f"Saved {OUT / 'rtdetr-dota-v1'}; DETECTOR_MODEL=models/rtdetr-dota-v1 is the default."
)
