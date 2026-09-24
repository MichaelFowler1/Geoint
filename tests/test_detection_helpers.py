# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import pytest

from backend.detection import (
    calibrate,
    summarize_counts,
    tile_origins,
    to_detections,
)
from backend.schemas import Detection


def _det(label: str) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_px=[0, 0, 10, 10])


def test_summarize_counts_tallies_by_label():
    dets = [_det("aircraft"), _det("aircraft"), _det("vehicle")]
    assert summarize_counts(dets) == {"aircraft": 2, "vehicle": 1}


def test_summarize_counts_empty():
    assert summarize_counts([]) == {}


def test_to_detections_maps_ids_to_labels_and_keeps_boxes():
    dets = to_detections(
        scores=[0.91, 0.40],
        labels=[4, 99],
        boxes=[[10.5, 20.0, 30.0, 40.25], [0, 0, 5, 5]],
        id2label={4: "airplane"},
    )
    assert [d.label for d in dets] == ["airplane", "99"]
    assert dets[0].confidence == 0.91
    assert dets[0].bbox_px == [10.5, 20.0, 30.0, 40.25]
    assert dets[1].bbox_px == [0.0, 0.0, 5.0, 5.0]


def test_to_detections_empty():
    assert to_detections([], [], [], {}) == []


def test_tile_origins_cover_the_image_and_end_on_its_edges():
    origins = tile_origins(2000, 1500, tile=640, overlap=160)
    xs = sorted({x for x, _ in origins})
    ys = sorted({y for _, y in origins})
    assert xs == [0, 480, 960, 1360]
    assert ys == [0, 480, 860]
    assert len(origins) == len(xs) * len(ys)


def test_tile_origins_small_image_is_one_tile():
    assert tile_origins(500, 300, tile=640, overlap=160) == [(0, 0)]


def test_calibrate_maps_by_class_and_leaves_unknowns_alone():
    table = {"plane": ([0.02, 0.05], [0.1, 0.9])}
    out = calibrate(
        [0.035, 0.08, 0.01, 0.4], ["plane", "plane", "plane", "ship"], table
    )
    assert out[0] == pytest.approx(0.5)  # halfway between breakpoints
    assert out[1] == pytest.approx(0.9)  # flat past the last one
    assert out[2] == pytest.approx(0.1)  # and before the first
    assert out[3] == 0.4  # a class with no table keeps its raw score


def test_calibrate_without_table_is_identity():
    assert calibrate([0.3, 0.7], ["a", "b"], None) == [0.3, 0.7]
