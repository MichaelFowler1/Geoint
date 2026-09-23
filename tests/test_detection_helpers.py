from backend.detection import summarize_counts, to_detections
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
