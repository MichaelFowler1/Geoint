from backend.schemas import Aircraft, DetectionResult


def test_aircraft_optional_fields():
    a = Aircraft(icao24="abc123")
    assert a.on_ground is False
    assert a.lat is None


def test_detection_result_roundtrip():
    r = DetectionResult(
        image_id="x",
        georeferenced=False,
        counts={"aircraft": 3},
        detections=[],
        scene_assessment="s",
        sitrep="t",
    )
    assert r.counts["aircraft"] == 3
    assert r.model_dump()["image_id"] == "x"
