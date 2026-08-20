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

import math
from typing import Annotated, Any, Literal, Self, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


MAX_POINT_VALUES = 200_000
MAX_STROKES_PER_SHAPE = 10_000
MAX_SHAPES_PER_SLICE = 10_000
MAX_SLICES_PER_DOCUMENT = 100_000
MAX_CLASSES_PER_DOCUMENT = 1_000

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
SliceKey = Annotated[str, Field(pattern=r"^(0|[1-9]\d*)$", max_length=16)]
SplitName = Literal["train", "valid", "test", "auto"]
ShapePayloadList = Annotated[list[dict[str, Any]], Field(max_length=MAX_SHAPES_PER_SLICE)]
SlicesPayload = Annotated[
    dict[SliceKey, ShapePayloadList],
    Field(max_length=MAX_SLICES_PER_DOCUMENT),
]
SplitPayload = Annotated[
    dict[SliceKey, SplitName],
    Field(max_length=MAX_SLICES_PER_DOCUMENT),
]
NegativeSlices = Annotated[list[SliceKey], Field(max_length=MAX_SLICES_PER_DOCUMENT)]


class StrictModel(BaseModel):
    """Base model for untrusted API payloads with fail-closed field handling."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _validate_point_pairs(
    points: list[float],
    *,
    minimum_values: int,
    field_name: str,
) -> list[float]:
    """Validate a bounded flat ``[x, y, ...]`` coordinate list."""
    if len(points) < minimum_values:
        raise ValueError(f"{field_name} requires at least {minimum_values // 2} points")
    if len(points) % 2:
        raise ValueError(f"{field_name} must contain complete x/y pairs")
    return points


class EraseStroke(StrictModel):
    """A vector-shape carve-out represented as a positive-radius stroke."""

    points: Annotated[list[FiniteFloat], Field(max_length=MAX_POINT_VALUES)]
    radius: PositiveFiniteFloat

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: list[float]) -> list[float]:
        """Require at least one complete finite coordinate pair."""
        return _validate_point_pairs(points, minimum_values=2, field_name="stroke points")


class BrushStroke(EraseStroke):
    """A single continuous stroke made by the brush tool.

    Attributes:
        points: Flat list of (x, y) pairs: [x0, y0, x1, y1, ...].
        radius: Brush radius in pixels.
        mode: Whether the stroke paints (adds) or erases pixels.
    """

    mode: Literal["paint", "erase"] = "paint"


class BaseShape(StrictModel):
    """Fields shared by every persisted annotation shape."""

    id: Annotated[str, Field(min_length=1, max_length=128)]
    classId: Annotated[int, Field(ge=0, le=2_147_483_647)]
    erased: Annotated[list[EraseStroke], Field(max_length=MAX_STROKES_PER_SHAPE)] = Field(
        default_factory=list
    )


class PolygonShape(BaseShape):
    """A closed polygon annotation.

    Attributes:
        id: Unique shape identifier.
        classId: Integer class index.
        kind: Discriminator literal.
        points: Flat list of (x, y) vertex pairs: [x0, y0, x1, y1, ...].
    """

    kind: Literal["polygon"]
    points: Annotated[list[FiniteFloat], Field(max_length=MAX_POINT_VALUES)]
    holes: Annotated[
        list[Annotated[list[FiniteFloat], Field(max_length=MAX_POINT_VALUES)]],
        Field(max_length=1_000),
    ] = Field(default_factory=list)

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: list[float]) -> list[float]:
        """Require a polygon to contain at least three complete vertices."""
        return _validate_point_pairs(points, minimum_values=6, field_name="polygon points")

    @field_validator("holes")
    @classmethod
    def validate_holes(cls, holes: list[list[float]]) -> list[list[float]]:
        """Apply the same three-vertex rule to every inner ring."""
        for hole in holes:
            _validate_point_pairs(hole, minimum_values=6, field_name="polygon hole")
        return holes


class RectShape(BaseShape):
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

    kind: Literal["rectangle"]
    x: FiniteFloat
    y: FiniteFloat
    w: NonNegativeFiniteFloat
    h: NonNegativeFiniteFloat


class EllipseShape(BaseShape):
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

    kind: Literal["ellipse"]
    cx: FiniteFloat
    cy: FiniteFloat
    rx: NonNegativeFiniteFloat
    ry: NonNegativeFiniteFloat


class BrushShape(BaseShape):
    """A collection of freehand brush strokes forming one annotation region.

    Attributes:
        id: Unique shape identifier.
        classId: Integer class index.
        kind: Discriminator literal.
        strokes: Ordered list of individual strokes.
    """

    kind: Literal["brush"]
    strokes: Annotated[list[BrushStroke], Field(max_length=MAX_STROKES_PER_SHAPE)]


Shape = Annotated[
    PolygonShape | RectShape | EllipseShape | BrushShape,
    Field(discriminator="kind"),
]
_SHAPE_ADAPTER: TypeAdapter[Shape] = TypeAdapter(Shape)


def _validated_shape_map(value: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Validate shape dictionaries while preserving the mapping interface internally."""
    validated: dict[str, list[dict[str, Any]]] = {}
    for slice_key, shapes in value.items():
        validated[slice_key] = [
            _SHAPE_ADAPTER.validate_python(shape).model_dump(exclude_none=True)
            for shape in shapes
        ]
    return validated


class AnnotationClass(StrictModel):
    """A labelled class used for annotating images.

    Attributes:
        classId: Integer class index (must be unique within a session).
        label: Human-readable class name.
        color: CSS colour string (e.g. ``"#ff0000"``).
        isVisible: Whether the class layer is rendered in the UI.
    """

    classId: Annotated[int, Field(ge=0, le=2_147_483_647)]
    label: Annotated[str, Field(min_length=1, max_length=128)]
    color: Annotated[
        str,
        Field(pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", max_length=9),
    ]
    isVisible: bool = True


class RenderOpts(StrictModel):
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
    vmin_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] = 1.0
    vmax_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] = 99.0
    cmap: Literal["gray", "viridis"] = "gray"

    @model_validator(mode="after")
    def validate_percentile_window(self) -> Self:
        """Require a non-empty, ascending percentile window."""
        if self.vmin_pct >= self.vmax_pct:
            raise ValueError("vmin_pct must be less than vmax_pct")
        return self


class SlicePayloadModel(StrictModel):
    """Shared, bounded slice payload with validated shape dictionaries."""

    slices: SlicesPayload = Field(default_factory=dict)
    split_by_slice: SplitPayload = Field(default_factory=dict)
    negative_slices: NegativeSlices = Field(default_factory=list)

    @field_validator("slices")
    @classmethod
    def validate_shapes(
        cls, value: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Enforce the discriminated shape schema at every API boundary."""
        return _validated_shape_map(value)

    @field_validator("negative_slices")
    @classmethod
    def validate_negative_slices(cls, value: list[str]) -> list[str]:
        """Reject duplicate negative-slice entries."""
        if len(value) != len(set(value)):
            raise ValueError("negative_slices must be unique")
        return value


class AutoSplit(StrictModel):
    """Validated deterministic train/valid/test ratio configuration."""

    ratios: tuple[
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)],
    ] = (0.8, 0.1, 0.1)
    seed: int = 1234

    @model_validator(mode="after")
    def validate_ratio_sum(self) -> Self:
        """Require train/valid/test ratios to sum to one."""
        if not math.isclose(sum(self.ratios), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("auto_split ratios must sum to 1")
        return self


def _validate_taxonomy(
    classes: list[AnnotationClass],
    shape_maps: list[dict[str, list[dict[str, Any]]]],
) -> None:
    """Require unique class IDs/names and valid shape-to-class references."""
    class_ids = [item.classId for item in classes]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("classId values must be unique")

    normalized_names = [item.label.casefold() for item in classes]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("class labels must be unique")

    referenced_ids = {
        int(shape["classId"])
        for shape_map in shape_maps
        for shapes in shape_map.values()
        for shape in shapes
    }
    unknown_ids = referenced_ids.difference(class_ids)
    if unknown_ids:
        raise ValueError(f"shapes reference undefined classId values: {sorted(unknown_ids)}")


class ExportSourceItem(SlicePayloadModel):
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
    source: Annotated[str, Field(min_length=1, max_length=4_096)]
    server_uri: Annotated[str, Field(max_length=2_048)] | None = None


class ExportRequest(SlicePayloadModel):
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

    dataset_name: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    kind: Literal["tiled", "local"] = "tiled"
    source: Annotated[str, Field(max_length=4_096)] = ""
    server_uri: Annotated[str, Field(max_length=2_048)] | None = None
    sources: Annotated[list[ExportSourceItem], Field(max_length=1_000)] = Field(default_factory=list)
    # Who produced this annotation. Stamped into the output folder name, COCO
    # info, and manifest.json so downloads are self-identifying for external
    # inter-annotator-agreement analysis.
    annotator: Annotated[str, Field(max_length=128)] = ""
    mode: Literal["fail", "overwrite", "merge"] = "merge"
    # Export target format: COCO-for-SAM3 (default) or DINOv3/Lightly semantic-seg
    # (per-split images/ + masks/ with matching filename stems + classes.json).
    format: Literal["coco_sam3", "lightly_dinov3"] = "coco_sam3"
    dry_run: bool = False
    # Include the (costly) polygon copy in COCO segmentation_poly. RLE is always
    # written and is exact; polygons are opt-in for external viewers.
    include_polygons: bool = False
    render: RenderOpts = Field(default_factory=RenderOpts)
    classes: Annotated[list[AnnotationClass], Field(max_length=MAX_CLASSES_PER_DOCUMENT)] = Field(
        default_factory=list
    )
    auto_split: dict[str, Any] = Field(
        default_factory=lambda: {"ratios": [0.8, 0.1, 0.1], "seed": 1234}
    )

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str | None) -> str | None:
        """Accept one portable filename component, never a path."""
        if value is None:
            return None
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("dataset_name must be a single safe filename component")
        return value

    @field_validator("auto_split")
    @classmethod
    def validate_auto_split(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate ratios and return the legacy mapping expected by export code."""
        return AutoSplit.model_validate(value).model_dump(mode="python")

    @model_validator(mode="after")
    def validate_export_taxonomy(self) -> Self:
        """Validate class uniqueness/references across every exported source."""
        shape_maps = [self.slices, *(item.slices for item in self.sources)]
        _validate_taxonomy(self.classes, shape_maps)
        if self.format == "lightly_dinov3" and len(self.classes) > 255:
            raise ValueError("Lightly export supports at most 255 classes")
        return self


class DraftPayload(SlicePayloadModel):
    """Autosave payload persisted between sessions.

    Attributes:
        classes: Annotation classes defined in the session.
        slices: Mapping of slice key → list of serialised shape dicts.
        split_by_slice: Mapping of slice key → dataset split name.
        negative_slices: Slice keys marked as negative examples.
    """

    classes: Annotated[list[AnnotationClass], Field(max_length=MAX_CLASSES_PER_DOCUMENT)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_draft_taxonomy(self) -> Self:
        """Require unique classes and valid class references before persistence."""
        _validate_taxonomy(self.classes, [self.slices])
        return self


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


class MeasureRequest(StrictModel):
    """Request body for per-region intensity measurement.

    Attributes:
        slice_index: Zero-based slice to sample.
        shapes: Serialised shape dicts; their union defines the measured region.
    """

    slice_index: Annotated[int, Field(ge=0)] = 0
    shapes: ShapePayloadList = Field(default_factory=list)

    @field_validator("shapes")
    @classmethod
    def validate_measure_shapes(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply the same shape validation used for draft/export payloads."""
        return [
            _SHAPE_ADAPTER.validate_python(shape).model_dump(exclude_none=True)
            for shape in value
        ]


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
        grouping: How images map onto samples (see ``ingest.sample_name_for``);
            echoed back as a per-sample preview so the UI can show the split
            before any bytes are uploaded. A ``Literal`` (matching
            ``ingest.GROUPING_MODES``) so a bogus value 422s here rather than
            surfacing as a 502 from inside ``preflight``'s broad except clause.
    """

    container_path: str
    names: list[str] = Field(default_factory=list)
    server_uri: str | None = None
    grouping: Literal["prefix", "per_image", "single"] = "prefix"


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
    """

    n_slices: int
    height: int
    width: int
    dtype: str
    is_rgb: bool
    value_range: list[float]
    keywords: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model fine-tuning / inference (Train tab)
#
# Three model families share one training/inference pipeline (data prep, job
# registry, mask<->polygon vectorization, Tiled write-back):
#   * "dinov3_lora"    — LoRA fine-tune of a pretrained DINOv3 ViT backbone
#                        (backend/dino_runtime.py, backend/dino_train.py).
#   * "dlsia_tunet"    — dlsia's tunable U-Net trained from scratch, no
#                        pretrained checkpoint (backend/dlsia_runtime.py).
#   * "dlsia_denoiser" — the same dlsia TUNet architecture trained
#                        self-supervised (Noise2Noise/Noise2Void) as a
#                        single-channel regression denoiser instead of a
#                        multi-class segmenter (backend/denoise_runtime.py,
#                        backend/denoise_train.py). Selected via
#                        ``TrainRequest.task == "denoising"``.
# ---------------------------------------------------------------------------

DinoArch = Literal["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16"]
ModelFamily = Literal["dinov3_lora", "dlsia_tunet", "dlsia_denoiser"]


def _validate_safe_filename_component(value: str, *, field_name: str) -> str:
    """Reject anything that isn't a single portable filename component.

    Shared by checkpoint/run_name/run_id fields, which are all joined onto a
    server-owned directory path — never accept a separator or traversal token.
    """
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} must be a single safe filename component")
    return value


class DinoHyperParams(StrictModel):
    """Bounded training hyperparameters for a DINOv3 LoRA fine-tune.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate for the LoRA + head parameters (only trainable ones).
        lora_rank: LoRA adapter rank.
        lora_alpha: LoRA scaling factor.
        batch_size: Training batch size.
        image_size: Square side length images/labels are letterboxed to; must
            be a multiple of 16 (DINOv3's patch size). With ``tiling`` on this
            is the tile window instead of a whole-image rescale target.
        seed: Seed for shuffling and augmentation.
        flip_augment: Whether to randomly flip image+label together.
        tiling: Train on native-resolution ``image_size`` windows cut from each
            slice, instead of rescaling the whole slice down to ``image_size``
            (see :mod:`tiling`). Recorded on the run, so inference reproduces
            whichever geometry the run was trained with.
    """

    epochs: Annotated[int, Field(ge=1, le=500)] = 30
    lr: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 1e-4
    lora_rank: Annotated[int, Field(ge=1, le=64)] = 8
    lora_alpha: Annotated[float, Field(gt=0, le=256, allow_inf_nan=False)] = 16.0
    batch_size: Annotated[int, Field(ge=1, le=16)] = 2
    # Tile count scales with 1/size², so a larger window means far fewer forward
    # passes per slice and more context in each one. 512 also satisfies both
    # families' divisors, keeping the default the same across them.
    image_size: Annotated[int, Field(ge=224, le=1024)] = 512
    seed: int = 1234
    flip_augment: bool = True
    tiling: bool = True

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: int) -> int:
        """Require a multiple of 16 so the patch grid divides evenly."""
        if value % 16 != 0:
            raise ValueError("image_size must be a multiple of 16 (DINOv3 patch size)")
        return value


class TunetHyperParams(StrictModel):
    """Bounded training hyperparameters for a dlsia TUNet trained from scratch.

    Attributes:
        epochs: Number of training epochs (from-scratch training typically
            needs more epochs than fine-tuning a pretrained backbone).
        lr: Learning rate for the whole network.
        depth: U-Net depth (encoder/decoder stages).
        base_channels: Channel count of the first conv stage.
        growth_rate: Channel growth factor per depth level.
        batch_size: Training batch size.
        image_size: Square side length images/labels are letterboxed to.
            dlsia's TUNet fixes its layer sizes to ``image_shape`` at
            construction, so train and inference images must match exactly.
            With ``tiling`` on this is the tile window — every window is exactly
            this size, so the constraint still holds.
        seed: Seed for shuffling and augmentation.
        flip_augment: Whether to randomly flip image+label together.
        tiling: Train on native-resolution ``image_size`` windows cut from each
            slice, instead of rescaling the whole slice down to ``image_size``
            (see :mod:`tiling`). Recorded on the run, so inference reproduces
            whichever geometry the run was trained with.
    """

    epochs: Annotated[int, Field(ge=1, le=1000)] = 60
    lr: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 1e-3
    depth: Annotated[int, Field(ge=2, le=6)] = 4
    base_channels: Annotated[int, Field(ge=1, le=128)] = 8
    growth_rate: Annotated[float, Field(gt=0, le=4, allow_inf_nan=False)] = 1.5
    batch_size: Annotated[int, Field(ge=1, le=32)] = 4
    image_size: Annotated[int, Field(ge=64, le=2048)] = 512
    seed: int = 1234
    flip_augment: bool = True
    tiling: bool = True

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: int) -> int:
        """A TUNet at ``depth`` stages needs the side divisible by 2**depth
        so every downsample/upsample step lands on a whole-pixel size."""
        if value % 64 != 0:
            raise ValueError("image_size must be a multiple of 64 (covers depth up to 6)")
        return value


class DinoV3LoraConfig(StrictModel):
    """Model-family config: LoRA fine-tune of a pretrained DINOv3 backbone.

    Attributes:
        model_family: Discriminator literal.
        arch: DINOv3 ViT backbone architecture.
        checkpoint: Filename of a pretrained checkpoint under the configured
            models directory — a single safe component, never a path.
        hyperparams: Training hyperparameters.
    """

    model_family: Literal["dinov3_lora"] = "dinov3_lora"
    arch: DinoArch
    checkpoint: Annotated[str, Field(min_length=1, max_length=255)]
    hyperparams: DinoHyperParams = Field(default_factory=DinoHyperParams)

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint(cls, value: str) -> str:
        return _validate_safe_filename_component(value, field_name="checkpoint")


class DlsiaTunetConfig(StrictModel):
    """Model-family config: dlsia tunable U-Net trained from scratch.

    Attributes:
        model_family: Discriminator literal.
        hyperparams: Training hyperparameters.
    """

    model_family: Literal["dlsia_tunet"] = "dlsia_tunet"
    hyperparams: TunetHyperParams = Field(default_factory=TunetHyperParams)


class DlsiaDenoiserConfig(StrictModel):
    """Model-family config: a single-channel image denoiser trained
    self-supervised, instead of a multi-class segmenter.

    Two architectures share this family. Keeping them under one
    ``model_family`` is deliberate: the frontend partitions runs with
    ``isSegmentationRun == !isDenoiserRun`` (``lib/runCompatibility.ts``), so a
    *new* family value would be silently treated as segmentation and offered in
    the fine-tune / apply / inference pickers, which key off a class list a
    denoiser does not have. An ``architecture`` discriminator inside the family
    avoids that entirely.

    :class:`TunetHyperParams` is reused verbatim rather than defining a parallel
    hyperparameter class — ``depth``/``base_channels`` mean downsampling levels
    and first-conv width for both architectures. Note it is the SAME class
    object the segmentation family uses, so architecture-specific knobs
    (``ae_compression``) live here on the config, not there.

    Attributes:
        model_family: Discriminator literal.
        architecture: Which network to build — ``"tunet"`` (dlsia TUNet with
            ``in_channels=1, out_channels=1``; see :mod:`denoise_runtime`) or
            ``"cnn_ae"`` (a plain convolutional autoencoder with an explicit
            latent bottleneck and NO skip connections; see
            :mod:`autoencoder_runtime`). Defaults to ``"tunet"`` so runs saved
            before this field existed keep their meaning.
        hyperparams: Training hyperparameters (reuses the segmentation
            family's TUNet knobs — depth/base_channels/growth_rate/etc.).
        ae_compression: How much the ``"cnn_ae"`` bottleneck compresses, as a
            ratio of input values to latent values. Higher removes more noise
            but also discards more real detail. Only meaningful for
            ``"cnn_ae"``.
        training_scheme: Self-supervised training objective — ``"n2n"``
            (Noise2Noise: paired noisy/noisy training), ``"n2v"``
            (Noise2Void: blind-spot training from single noisy images), or
            ``"ae"`` (pure self-reconstruction: target IS the input, and the
            bottleneck is what forces noise out).

            ``"ae"`` is valid ONLY with ``architecture="cnn_ae"``, enforced
            below. On a skip-connected network like TUNet, training on
            ``target == input`` makes ``f(x) = x`` trivially learnable: it
            converges to copying the input, removes no noise whatsoever, and
            still reports a falling loss. Without skips, the bottleneck cannot
            pass the input through unchanged, so reconstruction becomes a real
            denoising objective — noise is precisely the part that will not fit
            through. That pairing is a correctness constraint, not a
            convenience.
    """

    model_family: Literal["dlsia_denoiser"] = "dlsia_denoiser"
    architecture: Literal["tunet", "cnn_ae"] = "tunet"
    hyperparams: TunetHyperParams = Field(default_factory=TunetHyperParams)
    training_scheme: Literal["n2n", "n2v", "ae"]
    ae_compression: Annotated[int, Field(ge=4, le=64)] = 16

    @model_validator(mode="after")
    def _check_scheme_matches_architecture(self) -> "DlsiaDenoiserConfig":
        """Keep scheme and architecture to the combinations that make sense.

        ``ae`` + ``tunet`` is the identity-collapse footgun described above and
        must be impossible to request. The reverse (``cnn_ae`` with a masking or
        paired scheme) is not unsound in principle, just untested here, so it is
        refused rather than silently shipped.
        """
        if self.training_scheme == "ae" and self.architecture != "cnn_ae":
            raise ValueError(
                "training_scheme='ae' (pure self-reconstruction) requires "
                "architecture='cnn_ae'. On a skip-connected network it would just learn to "
                "copy its input and remove no noise."
            )
        if self.architecture == "cnn_ae" and self.training_scheme != "ae":
            raise ValueError(
                "architecture='cnn_ae' is only supported with training_scheme='ae'; "
                f"got {self.training_scheme!r}."
            )
        return self


ModelConfig = Annotated[
    Union[DinoV3LoraConfig, DlsiaTunetConfig, DlsiaDenoiserConfig],
    Field(discriminator="model_family"),
]


class BatchProbeRequest(StrictModel):
    """Request body to measure the largest batch size a model config can fit.

    Deliberately not a :class:`TrainRequest`: the probe feeds synthetic tensors, so
    it needs no sources, and requiring them would force the caller to invent data
    just to ask a question about memory.

    Attributes:
        model: Model-family configuration to size (patch size and the per-family
            hyperparameters come from its ``hyperparams``).
        n_classes: Segmentation head output channels. Affects memory only
            marginally, so it defaults to a typical value — meaning a batch size
            can be estimated before any classes have been defined.
    """

    model: ModelConfig
    n_classes: Annotated[int, Field(ge=1, le=MAX_CLASSES_PER_DOCUMENT)] = 2


class TrainRequest(StrictModel):
    """Request body to start a fine-tuning job for any model family.

    Attributes:
        task: What the trained model is for — ``"segmentation"`` (the
            original behaviour, requires at least one class) or
            ``"denoising"`` (a self-supervised Noise2Noise/Noise2Void
            denoiser, which has no class taxonomy at all). Defaults to
            ``"segmentation"`` so every pre-existing request body (which
            never sent this field) keeps behaving exactly as before.
        sources: Annotated samples to train on (each with its own slices/splits).
        classes: Annotation classes/taxonomy shared across every source.
            Required (at least one) when ``task == "segmentation"``; may be
            empty when ``task == "denoising"``, since a denoiser has nothing
            to classify.
        render: Render options used to rasterise training images.
        denoise: Optional denoising applied to the model's INPUT pixels. Recorded
            on the run and reapplied automatically at inference, so train and
            predict can never disagree — see :class:`DenoiseTrainOpts`. Ignored
            when ``task == "denoising"``: a denoiser learns to remove noise, so
            pre-cleaning its input would defeat the point.
        auto_split: Auto-split configuration for slices without an explicit split.
        model: Model-family configuration (discriminated on ``model_family``).
        run_name: Optional human-readable label for the resulting run.
        resume_from_run_id: Continue fine-tuning from this saved run's weights
            instead of starting from the pretrained base (DINOv3) or from
            scratch (TUNet). The saved run's architecture-defining settings
            win over anything sent in ``model`` — a resume that changed them
            could not load the saved weights at all — so the server overrides
            them rather than trusting the client to echo them back. Result is
            always a NEW run; the parent is never modified.
    """

    task: Literal["segmentation", "denoising"] = "segmentation"
    sources: Annotated[list[ExportSourceItem], Field(min_length=1, max_length=1_000)]
    classes: Annotated[list[AnnotationClass], Field(max_length=MAX_CLASSES_PER_DOCUMENT)]
    render: RenderOpts = Field(default_factory=RenderOpts)
    denoise: DenoiseTrainOpts | None = None
    auto_split: dict[str, Any] = Field(
        default_factory=lambda: {"ratios": [0.8, 0.1, 0.1], "seed": 1234}
    )
    model: ModelConfig
    run_name: Annotated[str, Field(max_length=128)] | None = None
    resume_from_run_id: Annotated[str, Field(min_length=1, max_length=255)] | None = None

    @field_validator("run_name")
    @classmethod
    def validate_run_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_safe_filename_component(value, field_name="run_name")

    @field_validator("resume_from_run_id")
    @classmethod
    def validate_resume_from_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_safe_filename_component(value, field_name="resume_from_run_id")

    @field_validator("auto_split")
    @classmethod
    def validate_auto_split(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate ratios and return the legacy mapping expected by export code."""
        return AutoSplit.model_validate(value).model_dump(mode="python")

    @model_validator(mode="after")
    def validate_train_taxonomy(self) -> Self:
        """Validate class uniqueness/references and the training-specific class cap.

        255 (not the export path's usual per-format check) because the label
        map trained against is always 0-indexed classes with 255 reserved as
        the ignore index — see coco_export.lightly_classes_map / build_export_plan.

        ``classes`` used to carry a hard ``Field(min_length=1)`` — fine while
        every trained model was a classifier, but a self-supervised denoiser
        has no classes at all. That unconditional constraint is replaced by
        this task-aware check: still required (and still validated/capped)
        for ``"segmentation"``, but allowed empty for ``"denoising"``.
        """
        if self.task == "segmentation" and len(self.classes) < 1:
            raise ValueError("Segmentation training requires at least one class")
        shape_maps = [item.slices for item in self.sources]
        _validate_taxonomy(self.classes, shape_maps)
        if len(self.classes) > 255:
            raise ValueError("Training supports at most 255 classes (0-indexed label map + 255 ignore)")
        return self


class InferRequest(StrictModel):
    """Request body to run inference with a saved fine-tuned run.

    Attributes:
        run_id: Identifier of a previously trained run (see dino_runtime.list_runs).
        kind: Source kind — ``"tiled"`` or ``"local"``.
        source: Tiled path or local relative path to run inference on.
        server_uri: Tiled server URI (kind == "tiled" only).
        slice_indices: Zero-based slice indices to run inference on.
        render: Render options; ``None`` reuses the run's stored render options.
        min_area: Minimum connected-component pixel area kept per predicted region.
        simplify_tol: Polygon simplification tolerance (pixels).
        min_confidence: Softmax confidence below which a pixel is treated as
            background (no class) — training never sees an explicit background
            class, since unannotated pixels are the ignore index, not a label.
    """

    run_id: Annotated[str, Field(min_length=1, max_length=255)]
    kind: Literal["tiled", "local"]
    source: Annotated[str, Field(min_length=1, max_length=4_096)]
    server_uri: Annotated[str, Field(max_length=2_048)] | None = None
    slice_indices: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1, max_length=2_000)]
    render: RenderOpts | None = None
    min_area: Annotated[int, Field(ge=0, le=1_000_000)] = 64
    simplify_tol: Annotated[float, Field(ge=0, le=50, allow_inf_nan=False)] = 1.5
    min_confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validate_safe_filename_component(value, field_name="run_id")

    @field_validator("slice_indices")
    @classmethod
    def validate_slice_indices(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("slice_indices must be unique")
        return value


class DenoiseTrainOpts(StrictModel):
    """Denoising applied to a model's INPUT pixels, at training and inference alike.

    Distinct from the Annotate tab's denoise preview, which is display-only and
    never reaches a model. When this is set on a training request it is recorded
    on the saved run, and inference reads it back off the run rather than off the
    request — the same rule ``tiling`` already follows, and for the same reason:
    a model must see the same pixel distribution it was trained on. Letting the
    two be chosen independently would produce a silent distribution shift with
    no error, just quietly worse predictions.

    Attributes:
        method: A ``denoise.ALL_METHODS`` entry other than ``"none"``/``"model"``
            (a learned denoiser as a preprocessor for another model is not
            supported — it would need its own run and GPU pass per slice).
        strength: 0..1, mapped onto the method's native parameter.
    """

    method: Annotated[str, Field(min_length=1, max_length=64)]
    strength: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5


class DenoiseBakeRequest(StrictModel):
    """Request body to denoise a whole volume into a new Tiled dataset.

    Attributes:
        source: Tiled path of the volume to denoise.
        server_uri: Tiled server URI.
        method: A ``denoise.ALL_METHODS`` entry other than ``"none"``.
        strength: 0..1, mapped onto the method's native parameter.
        target_path: Destination Tiled path; defaults to
            ``<source>_denoised`` (a sibling, so it lands next to its source in
            Browse). Must sit beneath the configured ingest root.
        description: Optional comma-separated tags, treated exactly as ingest
            treats them — searchable in Browse and pre-created as annotation
            classes.
    """

    source: Annotated[str, Field(min_length=1, max_length=4_096)]
    server_uri: Annotated[str, Field(max_length=2_048)] | None = None
    method: Annotated[str, Field(min_length=1, max_length=64)]
    strength: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5
    target_path: Annotated[str, Field(max_length=4_096)] | None = None
    description: Annotated[str, Field(max_length=2_048)] = ""
    run_id: Annotated[str, Field(max_length=256)] | None = None
    """Saved denoiser run to apply when ``method == "model"``; ignored otherwise."""


class MasksFromTiledRequest(StrictModel):
    """Request body to read a previously-written ``<stem>__masks`` container
    back into vectorized shapes for the Annotate canvas (see
    ``tiled_mask_sync.run_masks_readback_job``).

    Attributes:
        kind: Always ``"tiled"`` — masks are only ever written for Tiled
            sources (see ``tiled_mask_sync``'s module docstring), so a local
            source has nothing to read back.
        source: Tiled path whose ``<stem>__masks`` sibling to read.
        server_uri: Tiled server URI.
        min_area: Minimum connected-component pixel area kept per region.
        simplify_tol: Polygon simplification tolerance (pixels).
    """

    kind: Literal["tiled"]
    source: Annotated[str, Field(min_length=1, max_length=4_096)]
    server_uri: Annotated[str, Field(max_length=2_048)] | None = None
    # Same defaults as InferRequest, so masks loaded back match what "Import
    # as annotations" would have produced from the same underlying label maps.
    min_area: Annotated[int, Field(ge=0, le=1_000_000)] = 64
    simplify_tol: Annotated[float, Field(ge=0, le=50, allow_inf_nan=False)] = 1.5
