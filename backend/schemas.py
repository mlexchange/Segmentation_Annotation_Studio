"""Pydantic models for the Segmentation Annotation Studio API.

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
    # Who produced this annotation. Stamped into the output folder name, COCO
    # info, and manifest.json so downloads are self-identifying for external
    # inter-annotator-agreement analysis.
    annotator: str = ""
    mode: Literal["fail", "overwrite", "merge"] = "merge"
    # Export target format: COCO-for-SAM3 (default) or DINOv3/Lightly semantic-seg
    # (per-split images/ + masks/ with matching filename stems + classes.json).
    format: Literal["coco_sam3", "lightly_dinov3"] = "coco_sam3"
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


class MeasureRequest(BaseModel):
    """Request body for per-region intensity measurement.

    Attributes:
        slice_index: Zero-based slice to sample.
        shapes: Serialised shape dicts; their union defines the measured region.
    """

    slice_index: int = 0
    shapes: list[dict[str, Any]] = Field(default_factory=list)


class IngestPreflightRequest(BaseModel):
    """Request body for the pre-upload duplicate check.

    Sent as a POST body rather than query params because a dropped folder can
    hold hundreds of filenames — as a query string that exceeds the HTTP
    header size limit and the request is rejected with 431 before routing.

    Attributes:
        container_path: Target container, e.g. ``browse/myset``.
        names: Original filenames about to be uploaded (may be empty to only
            ask for a suggested free container name).
        server_uri: Target Tiled server URI; ``None`` uses the default server.
    """

    container_path: str
    names: list[str] = Field(default_factory=list)
    server_uri: str | None = None


class ZarrRegisterRequest(BaseModel):
    """Request body for registering an on-disk Zarr volume with Tiled.

    No bytes are uploaded: the volume stays where it is and Tiled reads it in
    place, so this carries a server-side path rather than file content.

    Attributes:
        path: Absolute path to the ``.zarr`` directory on the server.
        container_path: Target container, e.g. ``browse``.
        description: Optional keyword(s) stored on the node (same treatment as
            the drag-and-drop ingest, so the volume is filterable in Browse).
        on_conflict: ``"fail"``, ``"replace"`` or ``"skip"`` when the key exists.
        server_uri: Target Tiled server URI; ``None`` uses the default server.
    """

    path: str
    container_path: str = "browse"
    description: str = ""
    on_conflict: str = "fail"
    server_uri: str | None = None


class TiffStackRegisterRequest(ZarrRegisterRequest):
    """Request body for registering a TIFF directory as a 3-D volume.

    Same fields as :class:`ZarrRegisterRequest` — ``path`` is a directory of 2-D
    TIFF slices rather than a ``.zarr`` store. Kept as its own type so the two
    endpoints document themselves and can diverge without a breaking change.

    Unlike the Zarr path this does real work: the full-resolution slices are
    registered in place, but the downsampled pyramid levels the 3-D viewer
    actually renders have to be computed, so registration runs as a job.
    """


class DenoiseBakeRequest(BaseModel):
    """Request body to denoise a whole volume into a new Tiled dataset.

    The Annotate preview is non-destructive — it changes what you see, not the
    data, and exports still use the original pixels. This is the other half: it
    writes a denoised copy as a first-class dataset you can open, annotate and
    export.

    Attributes:
        source: Tiled path of the volume to denoise.
        server_uri: Tiled server URI; ``None`` uses the default server.
        method: A ``denoise.ALL_METHODS`` entry other than ``"none"``.
        strength: 0..1, mapped onto the method's native parameter.
        target_path: Destination Tiled path; defaults to ``<source>_denoised``, a
            sibling so it lands next to its source in Browse. Must sit beneath
            the configured ingest root.
        description: Optional comma-separated tags, treated exactly as ingest
            treats them — searchable in Browse.
    """

    source: str
    server_uri: str | None = None
    method: str
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    target_path: str | None = None
    description: str = ""
    run_id: str | None = None
    """Saved denoiser run, for the trained-model path. Not yet wired up here."""


class VolumeBuildRequest(BaseModel):
    """Request body for building a 3-D volume from a stack already in Tiled.

    Deliberately minimal: the slices are already in the catalog, so the only
    thing needed is which dataset. No source path, because requiring one would
    mean asking the user to re-supply data the app already holds.

    Attributes:
        source: Tiled path of the per-slice dataset.
        kind: Source kind; only ``"tiled"`` has a catalog to register into.
        container_path: Where to place the volume sidecar; defaults to the
            dataset's own parent container.
        server_uri: Target Tiled server URI; ``None`` uses the default server.
    """

    source: str
    kind: str = "tiled"
    container_path: str | None = None
    server_uri: str | None = None


class GuideClass(BaseModel):
    """One class entry in an annotation guide.

    Attributes:
        label: Human-readable class name (matches an annotation class label).
        color: CSS colour string used for this class.
        description: Free-text guidance on what the class is and how it looks.
        exampleCrops: Base64 data-URL PNG crops illustrating the class.
    """

    label: str
    color: str
    description: str = ""
    exampleCrops: list[str] = Field(default_factory=list)


class GuidePayload(BaseModel):
    """A project lead's annotation guide for a dataset.

    Attributes:
        classes: Ordered guide entries, one per class.
        notes: Optional overall notes for the annotation task.
    """

    classes: list[GuideClass] = Field(default_factory=list)
    notes: str = ""


class ImageMeta(BaseModel):
    """Shape and dtype metadata for an opened image source.

    Attributes:
        n_slices: Number of slices (1 for 2-D images).
        height: Image height in pixels.
        width: Image width in pixels.
        dtype: NumPy dtype string (e.g. ``"float32"``).
        is_rgb: ``True`` if the array has a colour channel dimension.
        value_range: ``[min, max]`` of the first slice.
        keywords: Dataset tags stored at ingest; each is pre-created as an
            annotation class in the Annotate tab.
        level_key: For a multiscale Zarr volume, the pyramid level being read
            (e.g. ``"scale2"``). ``height``/``width``/``n_slices`` always
            describe the FINEST level, since annotations are stored in
            full-resolution coordinates; these fields describe what is actually
            being displayed underneath them.
        level_index: Index of that level, finest = 0.
        level_count: Number of levels in the pyramid.
        level_height / level_width / level_n_slices: The open level's own shape.
        z_downsample: Finest-z / level-z. When > 1 the level can only address
            every f-th full-resolution slice.
    """

    n_slices: int
    height: int
    width: int
    dtype: str
    is_rgb: bool
    value_range: list[float]
    keywords: list[str] = Field(default_factory=list)
    level_key: str | None = None
    level_index: int | None = None
    level_count: int | None = None
    level_height: int | None = None
    level_width: int | None = None
    level_n_slices: int | None = None
    z_downsample: float | None = None
