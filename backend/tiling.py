"""Patch-based ("tiled") training and inference for images larger than the model input.

Without tiling, a slice is letterbox-rescaled to a single ``image_size`` square
(see :func:`train_common.letterbox`), so a 4096px slice at the default 512 is
downsampled 8× before the model ever sees it — and DINOv3 then predicts on a
grid 16× coarser still. Fine structure is unresolvable.

Tiling instead cuts ``window``-sized windows at **native resolution**, runs the
model per window, and recombines. Training patches come out exactly
``window``x``window``, which :func:`train_common.letterbox` passes through
untouched, so the existing training loop, flip augmentation and validation work
unchanged. Inference blends overlapping windows into a full-resolution
prediction.

Geometry comes from :mod:`qlty` (``NCYXQuilt``), which is already a dlsia
dependency and written by dlsia's author. Two of its properties matter here:

* Windows overlap, and a window's outermost ring is the least reliable part of
  its prediction (least surrounding context), so blending down-weights that
  ring and training masks it out of the loss entirely.
* The last window in each axis is *clamped* to sit inside the image rather than
  the image being zero-padded, so edge windows simply overlap more.

Softmax is applied **after** recombining, never per window: averaging softmaxed
patches is not the softmax of averaged logits (qlty's own docs call this out).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from train_common import IGNORE_INDEX

logger = logging.getLogger(__name__)

# Fraction of the window shared by consecutive windows. At 25% every interior
# pixel gets several predictions to average, which hides seams, while the window
# count only grows ~1/(1-f)² — 4/3× per axis here.
OVERLAP_FRACTION = 0.25
# The outermost ``window // BORDER_DIVISOR`` ring of each window is the
# down-weighted/ignored border. At 25% overlap this ring is exactly one half
# of the overlap (`border_for(w) == (w - step_for(w)) // 2`), so a
# down-weighted edge is always covered by a neighbour's full-weight interior,
# which extends the other half of the way across the overlap.
BORDER_DIVISOR = 8
# Weight given to border pixels when blending (interior is 1.0). Small but
# non-zero, so a border pixel still contributes where nothing else covers it.
BORDER_WEIGHT = 0.1
# Windows per forward pass during inference. Keeps device memory bounded
# independently of how many windows the image yields.
INFER_TILE_BATCH = 4


def step_for(window: int) -> int:
    """Stride between consecutive windows."""
    return max(1, window - int(round(window * OVERLAP_FRACTION)))


def border_for(window: int) -> int:
    """Width of the down-weighted/ignored ring at each window edge."""
    return max(1, window // BORDER_DIVISOR)


def tile_origins(dim: int, window: int, step: int) -> list[int]:
    """Start offsets of every window along one axis.

    Mirrors qlty's own grid, including its clamped final window
    (``min(i * step, dim - window)``) — so a streamed traversal lands on exactly
    the windows ``NCYXQuilt.unstitch`` would produce. Requires ``dim >= window``
    (pad first, see :func:`pad_to_min`).
    """
    if dim < window:
        raise ValueError(f"dim {dim} smaller than window {window}; pad first")
    full_steps = (dim - window) // step
    count = full_steps + 2 if dim > full_steps * step + window else full_steps + 1
    return [min(i * step, dim - window) for i in range(count)]


def pad_to_min(arr: np.ndarray, min_h: int, min_w: int, fill: int) -> np.ndarray:
    """Bottom/right-pad *arr* so it is at least ``min_h`` x ``min_w``.

    An image smaller than one window can't be tiled at all; padding to the
    window (rather than upscaling) keeps every real pixel at native scale. The
    padding is cropped back off after stitching, and for labels *fill* should be
    :data:`train_common.IGNORE_INDEX` so it never reaches the loss.
    """
    h, w = arr.shape[:2]
    pad_h, pad_w = max(0, min_h - h), max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_width, mode="constant", constant_values=fill)


def _quilt(height: int, width: int, window: int) -> Any:
    """``NCYXQuilt`` for one image's geometry, with this module's overlap/border."""
    from qlty.qlty2D import NCYXQuilt  # noqa: PLC0415 — optional dependency

    step = step_for(window)
    return NCYXQuilt(
        Y=height,
        X=width,
        window=(window, window),
        step=(step, step),
        border=border_for(window),
        border_weight=BORDER_WEIGHT,
    )


def qlty_available() -> bool:
    """True if :mod:`qlty` can be imported (ships with dlsia; see pyproject's ml extra)."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("qlty") is not None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _tile_pair(rgb: np.ndarray, label: np.ndarray, window: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut one full-resolution ``(rgb, label)`` pair into window-sized patches.

    Patches whose labels are entirely unannotated are dropped — with sparse
    annotations most of an image is :data:`train_common.IGNORE_INDEX` and such
    patches contribute nothing to the loss.
    """
    import torch  # noqa: PLC0415
    from qlty.cleanup import weed_sparse_classification_training_pairs_2D  # noqa: PLC0415

    img = pad_to_min(rgb, window, window, fill=0)
    lbl = pad_to_min(label, window, window, fill=IGNORE_INDEX)
    quilt = _quilt(img.shape[0], img.shape[1], window)

    # uint8 throughout: 4× less memory than float32 for the patch stack, and
    # IGNORE_INDEX (255) is representable. Conversion to float happens per batch
    # inside the training loop's to_tensor_fn.
    img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)
    lbl_t = torch.from_numpy(np.ascontiguousarray(lbl)).unsqueeze(0)

    def _cut_and_weed(*, mask_borders: bool) -> tuple[Any, Any]:
        """Cut patches, then drop those with no labelled pixel left.

        With *mask_borders*, each patch's border ring is blanked in the target so
        the loss only supervises interiors — where the model has full context —
        and only interior pixels count towards keeping a patch (``weed`` masks
        its validity check by the same border tensor).
        """
        patches_in, patches_out = quilt.unstitch_data_pair(
            img_t, lbl_t, missing_label=IGNORE_INDEX if mask_borders else None
        )
        border = quilt.border_tensor() if mask_borders else torch.ones(quilt.window)
        return weed_sparse_classification_training_pairs_2D(
            patches_in, patches_out, missing_label=IGNORE_INDEX, border_tensor=border
        )[:2]

    kept_in, kept_out = _cut_and_weed(mask_borders=True)

    # An annotation lying entirely within the image's outermost ring falls in no
    # patch's interior, so border-masking would discard the slice's only labels.
    # Keep those patches whole rather than silently training on nothing.
    if len(kept_in) == 0 and bool((lbl != IGNORE_INDEX).any()):
        kept_in, kept_out = _cut_and_weed(mask_borders=False)
        logger.info("Tiling: slice annotated only near its edge — kept %d unmasked patch(es)", len(kept_in))

    return [
        (
            np.ascontiguousarray(kept_in[i].numpy().transpose(1, 2, 0)),
            np.ascontiguousarray(kept_out[i].numpy()),
        )
        for i in range(len(kept_in))
    ]


# Share of training patches held out for validation when there are no validation
# slices — matches the 0.1 "valid" ratio of the default auto-split.
VAL_HOLDOUT_FRACTION = 0.1
# Below this many patches, holding any out costs more training signal than the
# metric is worth.
VAL_HOLDOUT_MIN_PATCHES = 8


def holdout_val_patches(
    datasets: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    seed: int,
    fraction: float = VAL_HOLDOUT_FRACTION,
    min_patches: int = VAL_HOLDOUT_MIN_PATCHES,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], int]:
    """Move a deterministic share of training patches into an empty val split.

    Splits are assigned per *slice* before tiling, so annotating one or two
    slices puts them all in train and the run reports no ``val_loss``/``val_miou``
    at all. Tiling turns those slices into many patches, which makes a holdout
    possible — so a small, seeded share becomes validation data.

    Only fires when validation is otherwise **empty**: an explicit or auto-assigned
    validation slice is always left alone.

    Caveat worth surfacing to the user: held-out patches come from the same
    image(s) as the ones trained on, so the resulting mIoU is optimistic compared
    with validating on a genuinely unseen slice.

    Returns:
        ``(datasets, n_held_out)`` — the input unchanged with ``0`` when the
        holdout doesn't apply.
    """
    train = datasets.get("train") or []
    if datasets.get("val") or len(train) < min_patches:
        return datasets, 0

    n_val = max(1, int(round(len(train) * fraction)))
    if n_val >= len(train):
        return datasets, 0

    order = np.random.default_rng(seed).permutation(len(train))
    val_idx = {int(i) for i in order[:n_val]}
    return (
        {
            **datasets,
            "train": [p for i, p in enumerate(train) if i not in val_idx],
            "val": [train[i] for i in sorted(val_idx)],
        },
        n_val,
    )


def tile_datasets(
    datasets: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    window: int,
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]] | None:
    """Replace each full-resolution pair with its window-sized patches.

    Takes and returns :func:`train_common.prepare_datasets`' shape, so it drops
    straight into the training pipeline.

    Checks *cancel_cb* (if given) between slices — tiling a large annotated set
    can itself take a while, and previously ran to completion uncancellably with
    no progress shown beyond one summary line per split at the very end. Returns
    ``None`` if cancelled partway, mirroring :func:`predict_label_map_tiled`'s
    cancellation contract so callers check for it the same way.
    """
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for split, pairs in datasets.items():
        patches: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (rgb, label) in enumerate(pairs):
            if cancel_cb is not None and cancel_cb():
                return None
            before = len(patches)
            patches.extend(_tile_pair(rgb, label, window))
            if progress_cb is not None:
                progress_cb(f"{split} slice {i + 1}/{len(pairs)}: {len(patches) - before} patch(es)")
        out[split] = patches
        if progress_cb is not None:
            progress_cb(f"{split}: {len(pairs)} slice(s) → {len(patches)} patch(es) of {window}px")
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _blend_tiled_forward(
    rgb: np.ndarray,
    *,
    forward_fn: Callable[[Any], Any],
    to_tensor_fn: Callable[[np.ndarray], Any],
    window: int,
    device: str,
    cancel_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> Any | None:
    """Blend per-window ``forward_fn`` output into a full-resolution canvas.

    This is the shared core behind :func:`predict_label_map_tiled`
    (classification: softmax/argmax/confidence-thresholding layered on top) and
    :func:`denoise_image_tiled` (regression: the blended canvas IS the result,
    no further post-processing) — anything that needs qlty's tiled-window
    blending but differs only in what it does with the blended output.
    *forward_fn* is fully generic here: nothing classification- or
    regression-specific lives in this function, only the tiling/accumulation
    machinery.

    Windows are forwarded in small batches, so peak DEVICE memory depends on
    :data:`INFER_TILE_BATCH` rather than on the image size — only one batch of
    windows is ever on the GPU/MPS device at once. The weighted output canvas
    they accumulate into lives on the host instead, precisely so it can scale
    with image size x channel count without competing for device memory; for a
    very large image or channel count that host allocation is still the actual
    memory ceiling here, just not a device one. The accumulation is
    arithmetically the same weighted mean ``NCYXQuilt.stitch`` computes.

    Returns a ``(C, orig_h, orig_w)`` float32 CPU tensor — *C* is whatever
    ``forward_fn`` emits per window (n_classes for segmentation logits, 1 for a
    single-channel denoiser) — or ``None`` if *cancel_cb* asked to stop partway.
    """
    import torch  # noqa: PLC0415

    orig_h, orig_w = rgb.shape[:2]
    padded = pad_to_min(rgb, window, window, fill=0)
    height, width = padded.shape[:2]

    quilt = _quilt(height, width, window)
    weight = quilt.weight  # (window, window): 1.0 interior, BORDER_WEIGHT ring
    step = step_for(window)
    origins = [(y, x) for y in tile_origins(height, window, step) for x in tile_origins(width, window, step)]

    image = to_tensor_fn(padded)  # (3, H, W) float, CPU
    canvas: Any = None  # (channels, H, W), allocated once the channel count is known
    norm = torch.zeros((height, width), dtype=torch.float32)

    if progress_cb is not None:
        progress_cb(f"{len(origins)} window(s) of {window}px @ {step}px step")

    for batch_start in range(0, len(origins), INFER_TILE_BATCH):
        if cancel_cb is not None and cancel_cb():
            return None
        batch_origins = origins[batch_start : batch_start + INFER_TILE_BATCH]  # noqa: E203
        batch = torch.stack([image[:, y : y + window, x : x + window] for y, x in batch_origins])  # noqa: E203

        out = forward_fn(batch.to(device)).detach().to("cpu", torch.float32)
        if canvas is None:
            canvas = torch.zeros((out.shape[1], height, width), dtype=torch.float32)

        for i, (y, x) in enumerate(batch_origins):
            canvas[:, y : y + window, x : x + window] += out[i] * weight  # noqa: E203
            norm[y : y + window, x : x + window] += weight  # noqa: E203

    if canvas is None:  # unreachable for a real image (always ≥1 window)
        raise RuntimeError("Tiled inference produced no windows")

    # Every pixel is covered by ≥1 window, but clamp anyway so a zero can never
    # turn into inf/NaN and poison whatever runs on top of this.
    blended = canvas / norm.clamp(min=1e-8)
    return blended[:, :orig_h, :orig_w]  # drop any padding


def predict_label_map_tiled(
    rgb: np.ndarray,
    *,
    forward_fn: Callable[[Any], Any],
    to_tensor_fn: Callable[[np.ndarray], Any],
    window: int,
    min_confidence: float,
    device: str,
    cancel_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """Predict a full-resolution label map by blending per-window predictions.

    The blending itself — streamed accumulation into a weighted canvas,
    arithmetically identical to ``NCYXQuilt.stitch`` — lives in
    :func:`_blend_tiled_forward`, shared with :func:`denoise_image_tiled`. This
    function only adds what's specific to classification: softmax is applied
    **after** recombining, never per window (averaging softmaxed patches is
    not the softmax of averaged logits — qlty's own docs call this out), then
    argmax and confidence-thresholding turn it into a label map.

    Returns a ``(H, W)`` uint8 map using the pipeline's convention —
    ``0`` = below ``min_confidence`` (background), ``1..n`` = class index + 1 —
    or ``None`` if *cancel_cb* asked to stop partway.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    blended = _blend_tiled_forward(
        rgb,
        forward_fn=forward_fn,
        to_tensor_fn=to_tensor_fn,
        window=window,
        device=device,
        cancel_cb=cancel_cb,
        progress_cb=progress_cb,
    )
    if blended is None:
        return None

    probs = F.softmax(blended, dim=0)  # after stitching — never per window
    confidence, pred_class = probs.max(dim=0)
    label = torch.where(
        confidence >= min_confidence,
        (pred_class + 1).to(torch.uint8),
        torch.zeros_like(pred_class, dtype=torch.uint8),
    )
    return label.numpy()


def denoise_image_tiled(
    rgb: np.ndarray,
    *,
    forward_fn: Callable[[Any], Any],
    to_tensor_fn: Callable[[np.ndarray], Any],
    window: int,
    device: str,
    cancel_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """Denoise a full-resolution image by blending per-window regression output.

    Same tiled-window blending as :func:`predict_label_map_tiled` — see
    :func:`_blend_tiled_forward`, which both share — but *forward_fn* here is a
    regression model (continuous-valued output, e.g. a trained denoiser)
    rather than a classifier, so blending is the END of the pipeline: no
    softmax, argmax or confidence-thresholding, all of which are meaningless
    on continuous output. The blended canvas IS the denoised image.

    Used both for a live single-slice preview and (by a caller driving it once
    per slice of a volume) a whole-volume "apply trained denoiser" bake job.

    Returns a float32 array at *rgb*'s original resolution — ``(H, W)`` when
    *forward_fn* emits a single channel (the expected case for a denoiser),
    else ``(C, H, W)`` — or ``None`` if *cancel_cb* asked to stop partway.
    """
    blended = _blend_tiled_forward(
        rgb,
        forward_fn=forward_fn,
        to_tensor_fn=to_tensor_fn,
        window=window,
        device=device,
        cancel_cb=cancel_cb,
        progress_cb=progress_cb,
    )
    if blended is None:
        return None
    out = blended.numpy()
    return out[0] if out.shape[0] == 1 else out
