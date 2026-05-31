from backend.detection import summarize_counts
from backend.schemas import Detection


def _det(label: str) -> Detection:
    return Detection(label=label, confidence=0.9, bbox_px=[0, 0, 10, 10])


def test_summarize_counts_tallies_by_label():
    dets = [_det("aircraft"), _det("aircraft"), _det("vehicle")]
    assert summarize_counts(dets) == {"aircraft": 2, "vehicle": 1}


def test_summarize_counts_empty():
    assert summarize_counts([]) == {}
