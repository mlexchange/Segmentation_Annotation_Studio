"""Tests for Pydantic schema models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from schemas import (
    BrushShape,
    BrushStroke,
    DraftPayload,
    EllipseShape,
    ExportRequest,
    ExportSourceItem,
    IngestPreflightRequest,
    PolygonShape,
    RectShape,
    RenderOpts,
)


def test_polygon_shape_roundtrip() -> None:
    """PolygonShape should serialise and deserialise cleanly."""
    s = PolygonShape(
        id="p1",
        classId=1,
        kind="polygon",
        points=[0.0, 0.0, 10.0, 0.0, 5.0, 10.0],
    )
    assert s.kind == "polygon"
    d = s.model_dump()
    assert d["kind"] == "polygon"


def test_brush_shape_has_strokes() -> None:
    """BrushShape should store stroke list with correct attributes."""
    stroke = BrushStroke(points=[0.0, 0.0, 10.0, 10.0], radius=5.0, mode="paint")
    s = BrushShape(id="b1", classId=2, kind="brush", strokes=[stroke])
    assert len(s.strokes) == 1
    assert s.strokes[0].mode == "paint"


def test_render_opts_defaults() -> None:
    """RenderOpts should have sensible defaults without explicit values."""
    opts = RenderOpts()
    assert opts.norm == "global"
    assert opts.scale == "linear"
    assert opts.cmap == "gray"


def test_rect_shape_fields() -> None:
    """RectShape should expose geometry fields correctly."""
    r = RectShape(id="r1", classId=3, kind="rectangle", x=10.0, y=20.0, w=100.0, h=50.0)
    assert r.kind == "rectangle"
    assert r.w == 100.0


def test_ellipse_shape_fields() -> None:
    """EllipseShape should expose centre and radii correctly."""
    e = EllipseShape(id="e1", classId=4, kind="ellipse", cx=50.0, cy=50.0, rx=30.0, ry=20.0)
    assert e.kind == "ellipse"
    assert e.rx == 30.0


def test_brush_stroke_erase_mode() -> None:
    """BrushStroke should accept 'erase' mode."""
    stroke = BrushStroke(points=[5.0, 5.0], radius=3.0, mode="erase")
    assert stroke.mode == "erase"


def test_ingest_preflight_request_carries_a_whole_folder() -> None:
    """A folder's worth of filenames must fit — as query params it 431'd."""
    names = [f"20260221_135217_petiole22_{i:05d}.tiff" for i in range(690)]
    req = IngestPreflightRequest(container_path="browse/ds", names=names, server_uri=None)
    assert len(req.names) == 690
    assert req.names[0].endswith("_00000.tiff")


def test_ingest_preflight_request_names_default_to_empty() -> None:
    """Omitting names is allowed — used to only ask for a free container name."""
    req = IngestPreflightRequest(container_path="browse/ds")
    assert req.names == []
    assert req.server_uri is None


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (RectShape, {"id": "r", "classId": 0, "kind": "rectangle", "x": 0, "y": 0, "w": -1, "h": 1}),
        (EllipseShape, {"id": "e", "classId": 0, "kind": "ellipse", "cx": 0, "cy": 0, "rx": 1, "ry": -1}),
        (PolygonShape, {"id": "p", "classId": 0, "kind": "polygon", "points": [0, 0, 1, 1]}),
        (PolygonShape, {"id": "p", "classId": 0, "kind": "polygon", "points": [0, 0, 1, 0, 1]}),
        (BrushStroke, {"points": [0, 0], "radius": 0}),
        (BrushStroke, {"points": [0, 0, 1], "radius": 1}),
    ],
)
def test_shape_geometry_rejects_invalid_normalized_or_pair_data(model, kwargs) -> None:
    """The API must enforce the documented normalized image-coordinate model."""
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_shape_geometry_rejects_non_finite_coordinates() -> None:
    """NaN/Infinity must not enter rasterization or persisted annotation JSON."""
    with pytest.raises(ValidationError):
        RectShape(id="r", classId=0, kind="rectangle", x=math.inf, y=0, w=1, h=1)


def test_draft_validates_shape_union_instead_of_accepting_raw_dicts() -> None:
    """Unknown or malformed shape dictionaries are rejected at the API boundary."""
    with pytest.raises(ValidationError):
        DraftPayload(slices={"0": [{"id": "x", "classId": 0, "kind": "shell"}]})
    with pytest.raises(ValidationError):
        DraftPayload(
            slices={
                "0": [
                    {
                        "id": "r",
                        "classId": 0,
                        "kind": "rectangle",
                        "x": 0,
                        "y": 0,
                        "w": -1,
                        "h": 1,
                    }
                ]
            }
        )


def test_polygon_holes_and_vector_erase_strokes_remain_supported() -> None:
    """Strict validation must preserve legitimate frontend polygon extensions."""
    payload = DraftPayload(
        classes=[{"classId": 0, "label": "region", "color": "#112233"}],
        slices={
            "0": [
                {
                    "id": "p",
                    "classId": 0,
                    "kind": "polygon",
                    "points": [0, 0, 10, 0, 10, 10],
                    "holes": [[2, 2, 4, 2, 4, 4]],
                    "erased": [{"points": [5, 5, 6, 6], "radius": 1}],
                }
            ]
        }
    )
    shape = payload.slices["0"][0]
    assert shape["holes"] == [[2.0, 2.0, 4.0, 2.0, 4.0, 4.0]]
    assert shape["erased"][0]["radius"] == 1.0


@pytest.mark.parametrize("split", ["../outside", "/tmp/owned", "training", ""])
def test_export_rejects_unrecognized_or_path_like_splits(split: str) -> None:
    """Split names become directories/ZIP prefixes and must be a fixed enum."""
    with pytest.raises(ValidationError):
        ExportSourceItem(kind="local", source="image.npy", split_by_slice={"0": split})


@pytest.mark.parametrize("name", ["../outside", "/tmp/owned", "a/b", "a\\b", ".", ".."])
def test_export_rejects_unsafe_dataset_names(name: str) -> None:
    """A dataset name is one filename component, never a client-provided path."""
    with pytest.raises(ValidationError):
        ExportRequest(dataset_name=name)


@pytest.mark.parametrize(
    "opts",
    [
        {"vmin_pct": -1, "vmax_pct": 99},
        {"vmin_pct": 1, "vmax_pct": 101},
        {"vmin_pct": 50, "vmax_pct": 50},
        {"vmin_pct": 80, "vmax_pct": 20},
    ],
)
def test_render_percentiles_are_bounded_and_ordered(opts: dict[str, float]) -> None:
    """Invalid percentile windows should fail before numerical rendering."""
    with pytest.raises(ValidationError):
        RenderOpts(**opts)


@pytest.mark.parametrize(
    "auto_split",
    [
        {"ratios": [0.8, 0.2], "seed": 1},
        {"ratios": [0.8, 0.2, 0.2], "seed": 1},
        {"ratios": [-0.1, 0.5, 0.6], "seed": 1},
    ],
)
def test_auto_split_requires_three_nonnegative_ratios_summing_to_one(auto_split: dict) -> None:
    """Split math must be deterministic and reject malformed ratio payloads."""
    with pytest.raises(ValidationError):
        ExportRequest(auto_split=auto_split)


def test_payload_rejects_duplicate_class_ids_and_unknown_shape_references() -> None:
    """Taxonomy IDs are unique and every shape references a declared class."""
    duplicate_classes = [
        {"classId": 1, "label": "one", "color": "#112233"},
        {"classId": 1, "label": "two", "color": "#445566"},
    ]
    with pytest.raises(ValidationError):
        DraftPayload(classes=duplicate_classes)

    with pytest.raises(ValidationError):
        DraftPayload(
            classes=[{"classId": 1, "label": "one", "color": "#112233"}],
            slices={
                "0": [
                    {
                        "id": "r",
                        "classId": 2,
                        "kind": "rectangle",
                        "x": 0,
                        "y": 0,
                        "w": 1,
                        "h": 1,
                    }
                ]
            },
        )
