"""Tests for Pydantic schema models."""

from __future__ import annotations

import pytest

from schemas import BrushShape, BrushStroke, EllipseShape, PolygonShape, RectShape, RenderOpts


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
