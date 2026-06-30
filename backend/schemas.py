"""Pydantic models for the SAM3 Annotation Studio API.

Shapes
------
* :class:`PolygonShape`  — closed polygon defined by flat point list
* :class:`RectShape`     — axis-aligned bounding rectangle
* :class:`EllipseShape`  — ellipse defined by centre + radii
* :class:`BrushShape`    — freehand brush strokes (paint / erase)

Other models
------------
* :class:`AnnotationClass`  — class label + colour + visibility
* :class:`RenderOpts`       — normalisation / colour-map settings
* :class:`ExportRequest`    — body for the export endpoint
* :class:`DraftPayload`     — autosave / restore payload
* :class:`ImageMeta`        — shape / dtype info for an opened image
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BrushStroke(BaseModel):
    """A single continuous stroke made by the brush tool.

    Attributes:
        points: Flat list of (x, y) pairs: [x0, y0, x1, y1, ...].
        radius: Brush radius in pixels.
        mode: Whether the stroke paints (adds) or erases pixels.
    """

    points: list[float]
    radius: float
    mode: Literal["paint", "erase"] = "paint"


class PolygonShape(BaseModel):
    """A closed polygon annotation.

    Attributes:
        id: Unique shape identifier.
        classId: Integer class index.
        kind: Discriminator literal.
        points: Flat list of (x, y) vertex pairs: [x0, y0, x1, y1, ...].
    """

    id: str
    classId: int
    kind: Literal["polygon"]
    points: list[float]


class RectShape(BaseModel):
    """An axis-aligned bounding-box annotation.

    Attributes:
        id: Unique shape identifier.
        classId: Integer class index.
        kind: Discriminator literal.
        x: Left edge (pixels).
        y: Top edge (pixels).
        w: Width (pixels).
        h: Height (pixels).
    """

    id: str
    classId: int
    kind: Literal["rectangle"]
    x: float
    y: float
    w: float
    h: float


class EllipseShape(BaseModel):
    """An ellipse annotation.

    Attributes:
        id: Unique shape identifier.
        classId: Integer class index.
        kind: Discriminator literal.
        cx: Centre x (pixels).
        cy: Centre y (pixels).
        rx: Semi-axis along x (pixels).
        ry: Semi-axis along y (pixels).
    """

    id: str
    classId: int
    kind: Literal["ellipse"]
    cx: float
    cy: float
    rx: float
    ry: float


class BrushShape(BaseModel):
    """A collection of freehand brush strokes forming one annotation region.

    Attributes:
        id: Unique shape identifier.
        classId: Integer class index.
        kind: Discriminator literal.
        strokes: Ordered list of individual strokes.
    """

    id: str
    classId: int
    kind: Literal["brush"]
    strokes: list[BrushStroke]


Shape = PolygonShape | RectShape | EllipseShape | BrushShape


class AnnotationClass(BaseModel):
    """A labelled class used for annotating images.

    Attributes:
        classId: Integer class index (must be unique within a session).
        label: Human-readable class name.
        color: CSS colour string (e.g. ``"#ff0000"``).
        isVisible: Whether the class layer is rendered in the UI.
    """

    classId: int
    label: str
    color: str
    isVisible: bool = True


class RenderOpts(BaseModel):
    """Image rendering / normalisation options.

    Attributes:
        norm: Normalisation scope — ``"slice"`` uses per-slice percentiles,
            ``"global"`` samples the whole volume.
        scale: Intensity transform applied before normalisation.
        vmin_pct: Lower percentile for contrast clipping (0–100).
        vmax_pct: Upper percentile for contrast clipping (0–100).
        cmap: Colour map for grayscale images.
    """

    norm: Literal["slice", "global"] = "global"
    scale: Literal["linear", "log", "symlog"] = "linear"
    vmin_pct: float = 1.0
    vmax_pct: float = 99.0
    cmap: Literal["gray", "viridis"] = "gray"


class ExportSourceItem(BaseModel):
    """A single annotated sample within a multi-source export.

    Attributes:
        kind: Source kind — ``"tiled"`` or ``"local"``.
        source: Tiled path or local relative path to the image data.
        server_uri: Tiled server URI (required when kind is ``"tiled"``).
        slices: Mapping of slice key → list of serialised shape dicts.
        split_by_slice: Mapping of slice key → dataset split name.
        negative_slices: Slice keys included as negative (unannotated) examples.
    """

    kind: Literal["tiled", "local"]
    source: str
    server_uri: str | None = None
    slices: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    split_by_slice: dict[str, str] = Field(default_factory=dict)
    negative_slices: list[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    """Request body for the dataset export endpoint.

    Attributes:
        dataset_name: Optional name for the output folder; auto-derived if absent.
        kind: Source kind — ``"tiled"`` or ``"local"`` (single-source mode).
        source: Tiled path or local relative path (single-source mode).
        server_uri: Tiled server URI (single-source mode, tiled only).
        sources: List of annotated samples for multi-source export (overrides
            kind/source/slices when provided).
        mode: Conflict resolution when output already exists.
        dry_run: If ``True``, compute the export plan but do not write files.
        render: Render options applied to exported PNG tiles.
        classes: Annotation classes present in this export.
        slices: Mapping of slice key → list of serialised shape dicts (single-source).
        split_by_slice: Mapping of slice key → dataset split name (single-source).
        auto_split: Auto-split configuration (ratios + seed).
        negative_slices: Slice keys included as negative examples (single-source).
    """

    dataset_name: str | None = None
    kind: Literal["tiled", "local"] = "tiled"
    source: str = ""
    server_uri: str | None = None
    sources: list[ExportSourceItem] = Field(default_factory=list)
    mode: Literal["fail", "overwrite", "merge"] = "merge"
    dry_run: bool = False
    # Include the (costly) polygon copy in COCO segmentation_poly. RLE is always
    # written and is exact; polygons are opt-in for external viewers.
    include_polygons: bool = False
    render: RenderOpts = Field(default_factory=RenderOpts)
    classes: list[AnnotationClass] = Field(default_factory=list)
    slices: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    split_by_slice: dict[str, str] = Field(default_factory=dict)
    auto_split: dict[str, Any] = Field(
        default_factory=lambda: {"ratios": [0.8, 0.1, 0.1], "seed": 1234}
    )
    negative_slices: list[str] = Field(default_factory=list)


class DraftPayload(BaseModel):
    """Autosave payload persisted between sessions.

    Attributes:
        classes: Annotation classes defined in the session.
        slices: Mapping of slice key → list of serialised shape dicts.
        split_by_slice: Mapping of slice key → dataset split name.
        negative_slices: Slice keys marked as negative examples.
    """

    classes: list[AnnotationClass] = Field(default_factory=list)
    slices: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    split_by_slice: dict[str, str] = Field(default_factory=dict)
    negative_slices: list[str] = Field(default_factory=list)


class SaveVersionRequest(BaseModel):
    """Explicit save request with optional version metadata.

    Attributes:
        payload: Current annotation session state.
        annotated_by: Name or identifier of the person who annotated.
        notes: Free-text notes for this version.
        thumbnail_base64: Optional PNG from the save-modal preview (skips re-render).
    """

    payload: DraftPayload
    annotated_by: str = ""
    notes: str = ""
    thumbnail_base64: str | None = None


class ImageMeta(BaseModel):
    """Shape and dtype metadata for an opened image source.

    Attributes:
        n_slices: Number of slices (1 for 2-D images).
        height: Image height in pixels.
        width: Image width in pixels.
        dtype: NumPy dtype string (e.g. ``"float32"``).
        is_rgb: ``True`` if the array has a colour channel dimension.
        value_range: ``[min, max]`` of the first slice.
    """

    n_slices: int
    height: int
    width: int
    dtype: str
    is_rgb: bool
    value_range: list[float]
