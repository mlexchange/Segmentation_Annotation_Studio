"""Torch-free tests for infer_jobs: label-map vectorization and preview colour."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage

import infer_jobs


def _classes() -> list[dict]:
    return [
        {"classId": 5, "label": "pore", "color": "#ff0000"},
        {"classId": 9, "label": "void", "color": "#00ff00"},
    ]


def test_vectorize_label_map_maps_channel_index_to_run_classid() -> None:
    """Predicted channel 0 -> run_classes[0]["classId"] (5), not the raw index."""
    label = np.zeros((40, 40), dtype=np.uint8)
    label[5:25, 5:25] = 1  # channel 0 -> classId 5

    shapes = infer_jobs._vectorize_label_map(
        label,
        _classes(),
        min_area=10,
        simplify_tol=0.0,
        run_id="run123456",
        slice_idx=3,
    )

    assert len(shapes) == 1
    assert shapes[0]["classId"] == 5
    assert shapes[0]["kind"] == "polygon"
    assert len(shapes[0]["points"]) >= 6


def test_vectorize_label_map_ignores_background() -> None:
    """Value 0 (below confidence threshold / no prediction) yields no shapes."""
    label = np.zeros((20, 20), dtype=np.uint8)
    shapes = infer_jobs._vectorize_label_map(label, _classes(), 1, 0.0, "run1", 0)
    assert shapes == []


def test_vectorize_label_map_filters_small_components_by_min_area() -> None:
    label = np.zeros((40, 40), dtype=np.uint8)
    label[0:2, 0:2] = 1  # 4px region

    shapes = infer_jobs._vectorize_label_map(label, _classes(), min_area=100, simplify_tol=0.0, run_id="r", slice_idx=0)

    assert shapes == []


def test_vectorize_label_map_handles_a_prediction_that_fills_the_whole_frame() -> None:
    """Regression: a predicted region touching all four edges (e.g. a dominant
    class spanning the whole slice) has no internal 0.5-level transition for
    skimage.measure.find_contours to trace, and previously produced zero
    shapes silently (confirmed against a live HTTP training+inference run —
    every pixel predicted as the sole class yielded 0 regions instead of 1)."""
    label = np.ones((40, 60), dtype=np.uint8)  # entire frame is class 5 (channel 0)

    shapes = infer_jobs._vectorize_label_map(label, _classes(), min_area=10, simplify_tol=0.0, run_id="r", slice_idx=0)

    assert len(shapes) == 1
    assert shapes[0]["classId"] == 5
    xs = shapes[0]["points"][0::2]
    ys = shapes[0]["points"][1::2]
    # The traced polygon should span essentially the full canvas, not collapse to nothing.
    assert max(xs) - min(xs) > 50
    assert max(ys) - min(ys) > 30


def test_vectorize_label_map_handles_multiple_classes_and_components() -> None:
    label = np.zeros((60, 60), dtype=np.uint8)
    label[5:20, 5:20] = 1  # classId 5
    label[30:45, 30:45] = 2  # classId 9

    shapes = infer_jobs._vectorize_label_map(label, _classes(), min_area=10, simplify_tol=0.0, run_id="r", slice_idx=0)

    class_ids = sorted(s["classId"] for s in shapes)
    assert class_ids == [5, 9]


def test_hex_to_rgb_parses_standard_and_short_hex() -> None:
    assert infer_jobs._hex_to_rgb("#ff0000") == (255, 0, 0)
    assert infer_jobs._hex_to_rgb("#0f0") == (0, 255, 0)


def test_hex_to_rgb_falls_back_on_invalid_input() -> None:
    assert infer_jobs._hex_to_rgb(None) == (255, 0, 0)
    assert infer_jobs._hex_to_rgb("not-a-color") == (255, 0, 0)


def test_preview_png_colourises_cached_label_map() -> None:
    label = np.zeros((10, 10), dtype=np.uint8)
    label[2:5, 2:5] = 1  # classId 5, color #ff0000

    buf = io.BytesIO()
    PILImage.fromarray(label, mode="L").save(buf, format="PNG")
    infer_jobs._cache_put(
        "job1",
        {
            "run_id": "run1",
            "classes": _classes(),
            "source": "s",
            "kind": "local",
            "server_uri": None,
            "height": 10,
            "width": 10,
            "label_pngs": {0: buf.getvalue()},
        },
    )

    png = infer_jobs.preview_png("job1", 0)
    rgba = np.asarray(PILImage.open(io.BytesIO(png)))

    assert rgba.shape == (10, 10, 4)
    assert tuple(rgba[3, 3][:3]) == (255, 0, 0)
    assert rgba[0, 0][3] == 0  # background stays transparent


def test_preview_png_raises_404_for_unknown_job() -> None:
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        infer_jobs.preview_png("no-such-job", 0)
    assert exc_info.value.status_code == 404


def test_cache_evicts_oldest_beyond_max_cached_jobs() -> None:
    for i in range(infer_jobs._MAX_CACHED_JOBS + 2):
        infer_jobs._cache_put(f"evict-job-{i}", {"label_pngs": {}, "classes": []})
    assert infer_jobs._cache_get("evict-job-0") is None
    assert infer_jobs._cache_get(f"evict-job-{infer_jobs._MAX_CACHED_JOBS + 1}") is not None
