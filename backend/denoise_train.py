"""Self-supervised denoiser training: data sampling, patching, and the loop.

This is the training-time counterpart to :mod:`denoise_runtime` (which only
builds/saves/loads the single-channel TUNet) and the denoiser's answer to
:func:`train_common.run_training_loop`.

Why a separate loop instead of extending ``run_training_loop``
--------------------------------------------------------------
``train_common.run_training_loop`` is not family-generic in the way its name
suggests — it is *segmentation*-generic. Three of its assumptions are
hardcoded and all three are wrong for regression:

1. ``nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)`` — a classification loss
   over class logits.
2. Its ``_prep`` casts the target with ``.astype(np.int64)`` — class indices,
   not intensities.
3. ``_evaluate`` does ``logits.argmax(dim=1)`` + :func:`train_common.compute_miou`.

So this module deliberately duplicates the epoch/batch/scheduler/cancel
skeleton rather than growing a ``task=`` switch through the middle of a
function every existing segmentation run depends on. Everything *around* the
loss — device handling, the ``on_batch``/``on_epoch`` cooperative-cancel
protocol (return ``True`` to stop), AdamW + cosine annealing, the returned
metrics dict shape — is kept identical, so :mod:`train_jobs` drives this the
same way it drives the segmentation loop.

Three self-supervised schemes
-----------------------------
None needs a single annotation; all three learn from the raw slices themselves.
The first two are architecture-agnostic; the third depends on the architecture
having no skip connections.

* **Noise2Noise** (``"n2n"``) — train slice *i* to predict slice *i+stride*.
  Adjacent slices of a tomographic/microscopy volume share structure but carry
  independent noise realizations, so under MSE the optimum is the conditional
  mean, i.e. the shared (clean) structure: the noise cannot be predicted and
  averages out.
* **Noise2Void** (``"n2v"``) — single slices, no pairing. A small fraction of
  pixels is *masked* by overwriting each with a random neighbour's value, and
  the loss is evaluated **only** at those coordinates, against the original
  values. Because the model never sees a masked pixel's own value, it cannot
  learn the identity function — it has to infer the value from context, which
  is exactly the denoising task.
* **Autoencoder** (``"ae"``) — single slices, pure self-reconstruction: the
  target IS the input, under plain MSE. That only works on an architecture with
  no skip connections and a narrow latent bottleneck (``"cnn_ae"``; see
  :mod:`autoencoder_runtime`), which is why ``schemas.DlsiaDenoiserConfig``
  refuses this scheme on TUNet. Given those, the input physically cannot pass
  through unchanged, and noise — the high-entropy part that does not fit through
  the bottleneck — is what gets dropped. Requires no synthetic corruption, which
  is the point: adding fake noise to already-noisy data trains the model on the
  wrong noise distribution. The honest tradeoff is that a bottleneck discards
  fine real detail along with the noise, so this tends to blur more than n2v.

Intensity normalisation
-----------------------
The denoiser is single-channel and trains on raw intensity, not on the
3-channel RGB tiles the segmentation path renders. Slices are mapped to uint8
through :func:`images.normalize_scalar_unit` — the same intensity pipeline
behind the 2-D canvas and the 3-D volume view — so training data matches what
the user actually sees. The percentile bounds are forced **volume-global**
even when the request asks for per-slice normalisation: a per-slice mapping
would put a Noise2Noise input and its target on two different intensity
scales, which would show up as a constant structural error the network cannot
fix and would happily waste capacity trying to. The requested ``scale``
transform (linear/log/symlog) is preserved.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Slice offset between a Noise2Noise input and its target. 1 = immediately
# adjacent, which is the strongest structural correspondence available; larger
# strides trade structural similarity for a bit more independence. Not a
# `TunetHyperParams` field (that schema is shared with the segmentation TUNet
# and is out of scope here), so it is a sampler argument with this default.
DEFAULT_PAIR_STRIDE = 1

# Share of pixels blanked per patch for Noise2Void. The N2V paper's usable
# range is ~0.5-2%: too few and each patch supervises almost nothing (the loss
# gets noisy and training crawls), too many and the masked pixels start
# destroying the very context the model needs to inpaint them from.
DEFAULT_N2V_MASK_FRACTION = 0.015
# Side of the square window a masked pixel's replacement value is drawn from.
DEFAULT_N2V_NEIGHBOURHOOD = 5
# Bounded resampling attempts when a drawn donor is unusable (it is the masked
# pixel itself, or another masked pixel). See :func:`n2v_mask_and_replace`.
_N2V_MAX_RESAMPLE = 8


# ---------------------------------------------------------------------------
# Raw slice sampling (annotation-free — deliberately NOT prepare_datasets)
# ---------------------------------------------------------------------------


def _render_opts(render: Any) -> dict[str, Any]:
    """Coerce a ``RenderOpts`` model (or plain dict) to the dict form
    :mod:`images` expects, with the normalisation scope pinned to
    ``"global"`` (see this module's docstring)."""
    opts = render.model_dump() if hasattr(render, "model_dump") else dict(render or {})
    return {**opts, "norm": "global"}


def selected_slice_indices(item: Any, n_slices: int) -> list[int]:
    """Which raw slice indices of *item* the denoiser should train on.

    The Learned Denoiser panel has no annotations to send, so it reuses
    ``ExportSourceItem.slices`` purely to name the in-scope slice indices, with
    an empty shape list for each (there is no other field on that schema for
    "just these indices"). An empty/absent mapping means the whole volume.

    Returns a sorted, deduplicated, in-bounds list.
    """
    keys = getattr(item, "slices", None) or {}
    indices: set[int] = set()
    for key in keys:
        try:
            idx = int(key)
        except (TypeError, ValueError):
            logger.warning("Denoiser scope: ignoring non-integer slice key %r", key)
            continue
        if 0 <= idx < n_slices:
            indices.add(idx)
    return sorted(indices) if indices else list(range(n_slices))


def _slice_to_gray_uint8(
    node: Any,
    meta: dict[str, Any],
    idx: int,
    opts: dict[str, Any],
    global_range: tuple[float, float] | None,
) -> np.ndarray:
    """Read slice *idx* and map it to a ``(H, W)`` uint8 grayscale array.

    Matches ``images.render_slice``'s grayscale branch exactly (same
    ``normalize_scalar_unit`` call, same ``round()`` quantisation), so the
    denoiser trains on the intensities the viewer displays. A colour source is
    collapsed to a single channel first — the network has ``in_channels=1``.
    """
    import arrays as arrays_mod  # noqa: PLC0415 — avoid a hard import-time cycle
    import images as images_mod  # noqa: PLC0415

    arr = np.asarray(arrays_mod.read_slice(node, meta, idx))
    if arr.ndim == 3:
        arr = arr[:, :, :3].astype(np.float64).mean(axis=2)
    unit = images_mod.normalize_scalar_unit(arr, opts, global_range)
    return (unit * 255.0).round().astype(np.uint8)


def load_source_slices(
    item: Any,
    render: Any,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Read every in-scope slice of one source as uint8 grayscale.

    Returns ``(indices, {index: (H, W) uint8})``.

    Each slice is read **exactly once** and shared by every pair that
    references it. That is deliberately not the "fetch a contiguous pair in one
    round-trip" optimisation (``node[i:i+2]``) the shape of a Noise2Noise
    sampler first suggests: with ``stride=1`` every interior slice belongs to
    two pairs (as input of one and target of the previous), and with
    ``both_directions`` to four, so a per-pair range read would fetch each
    slice 2-4x. Reading the scope once into a dict is strictly fewer
    round-trips than any per-pair batching, at the same peak memory the
    segmentation path already accepts (uint8 grayscale here vs. its uint8 RGB —
    a third the bytes per slice).
    """
    import arrays as arrays_mod  # noqa: PLC0415
    import images as images_mod  # noqa: PLC0415

    node = arrays_mod.resolve_array(item.source, item.kind, item.server_uri)
    meta = arrays_mod.array_shape_meta(node)
    indices = selected_slice_indices(item, meta["n_slices"])

    opts = _render_opts(render)
    global_range = images_mod._sample_global_stats(node, meta)

    slices: dict[int, np.ndarray] = {}
    for n_done, idx in enumerate(indices, start=1):
        slices[idx] = _slice_to_gray_uint8(node, meta, idx, opts, global_range)
        if progress_cb is not None and (n_done % 10 == 0 or n_done == len(indices)):
            progress_cb(f"{item.source}: read {n_done}/{len(indices)} slice(s)")
    return indices, slices


def noise2noise_pairs(
    indices: Sequence[int],
    *,
    stride: int = DEFAULT_PAIR_STRIDE,
    both_directions: bool = True,
) -> list[tuple[int, int]]:
    """Index pairs ``(input_index, target_index)`` for Noise2Noise.

    A pair is emitted only when **both** of its indices are in *indices*, which
    is what keeps it inside the volume at either end: the last in-scope slice
    simply has no partner rather than being clamped onto itself (a
    self-pair would be a perfect identity target and would train the network to
    do nothing).

    With *both_directions*, each adjacency also contributes its reverse pair.
    That is free extra data, not a duplicate: MSE is symmetric in value but the
    two directions are different (input, target) assignments, so the network
    sees each slice as an input as well as a target.
    """
    if stride < 1:
        raise ValueError(f"Noise2Noise pair stride must be >= 1, got {stride}")
    available = {int(i) for i in indices}
    pairs: list[tuple[int, int]] = []
    for i in sorted(available):
        j = i + stride
        if j in available:
            pairs.append((i, j))
            if both_directions:
                pairs.append((j, i))
    return pairs


def prepare_noise2noise_datasets(
    sources: Iterable[Any],
    render: Any,
    *,
    stride: int = DEFAULT_PAIR_STRIDE,
    both_directions: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build ``(input_slice, target_slice)`` pairs from adjacent raw slices.

    Annotation-free by construction: unlike
    :func:`train_common.prepare_datasets`, nothing here goes through
    ``coco_export.build_export_plan`` — a denoiser has no classes to rasterise
    and would be blocked entirely by that path's "no annotated slices" outcome.

    Returns :func:`train_common.prepare_datasets`' dict shape
    (``{"train": [(input, target), ...], "val": [...]}``), except both arrays
    are ``(H, W)`` uint8 grayscale rather than ``(rgb_hwc, label_hw)``.
    Everything lands in ``"train"``; the validation split is taken afterwards
    by :func:`tiling.holdout_val_patches`, at patch granularity, which is the
    same seeded holdout the tiled segmentation path already uses (and the only
    thing that works for a one- or two-slice scope).
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for item in sources:
        indices, slices = load_source_slices(item, render, progress_cb)
        index_pairs = noise2noise_pairs(indices, stride=stride, both_directions=both_directions)
        if not index_pairs:
            logger.warning(
                "Noise2Noise: source %r has no slice %d apart in scope — contributed no pairs",
                getattr(item, "source", "?"),
                stride,
            )
        pairs.extend((slices[i], slices[j]) for i, j in index_pairs)
        if progress_cb is not None:
            progress_cb(f"{item.source}: {len(indices)} slice(s) → {len(index_pairs)} Noise2Noise pair(s)")

    if not pairs:
        raise ValueError(
            "Noise2Noise needs at least two slices that are "
            f"{stride} apart within the selected scope — none were found. "
            "Widen the slice range, or train with Noise2Void instead."
        )
    return {"train": pairs, "val": []}


def prepare_noise2void_datasets(
    sources: Iterable[Any],
    render: Any,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build single-slice training items for Noise2Void.

    Returns the same ``{"train": [(input, target), ...], "val": []}`` shape as
    :func:`prepare_noise2noise_datasets` so patching and the loop are shared,
    with the slice paired **with itself**: the blind-spot masking that makes
    the two differ is applied per patch, per epoch, inside
    :func:`run_denoise_training_loop` (a fresh mask each time a patch is seen,
    which is the point — one fixed mask would supervise only ~1.5% of pixels
    ever). The same array object is used on both sides; nothing downstream
    mutates it in place.
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for item in sources:
        indices, slices = load_source_slices(item, render, progress_cb)
        pairs.extend((slices[i], slices[i]) for i in indices)
        if progress_cb is not None:
            progress_cb(f"{item.source}: {len(indices)} Noise2Void slice(s)")

    if not pairs:
        raise ValueError("Noise2Void needs at least one slice in the selected scope — none were found.")
    return {"train": pairs, "val": []}


# ---------------------------------------------------------------------------
# Patch extraction (qlty geometry, WITHOUT the sparse-annotation weeding)
# ---------------------------------------------------------------------------


def tile_denoise_pair(inp: np.ndarray, tgt: np.ndarray, window: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut one ``(input, target)`` grayscale pair into ``window``-sized patches.

    Deliberately **not** :func:`tiling._tile_pair`. That one runs
    ``qlty.cleanup.weed_sparse_classification_training_pairs_2D``, which drops
    every patch containing no labelled pixel. That is right for sparse
    segmentation annotations and completely wrong here: a denoiser is trained
    on raw pixels, so every patch is valid training data and a weeded run would
    silently discard the entire dataset (an all-zero "label" is, to the weeder,
    an unlabelled patch).

    Input and target are unstitched **independently through the same
    ``NCYXQuilt``**, so patch *k* of one is the exact same window of the image
    as patch *k* of the other — spatial alignment comes from shared geometry
    rather than from trusting a paired helper. The quilt is
    :func:`tiling._quilt`, the same geometry
    :func:`tiling.denoise_image_tiled` uses at inference time, so the model is
    trained on and applied to identically-shaped windows.

    Note that an image smaller than one window is zero-padded up to it (see
    :func:`tiling.pad_to_min`); the padded region is constant 0 in both input
    and target, so it is trivially satisfiable rather than misleading — but it
    is real loss mass, so tiling an undersized image is worth avoiding.
    """
    import torch  # noqa: PLC0415

    import tiling  # noqa: PLC0415 — optional (qlty) dependency

    if inp.shape != tgt.shape:
        raise ValueError(f"Denoiser input/target shapes differ: {inp.shape} vs {tgt.shape}")

    img = tiling.pad_to_min(inp, window, window, fill=0)
    tar = tiling.pad_to_min(tgt, window, window, fill=0)
    quilt = tiling._quilt(img.shape[0], img.shape[1], window)

    # (1, 1, H, W) — one image, one channel. uint8 all the way to the batch's
    # to_tensor_fn, same as the segmentation patch cache: 4x less memory than
    # float32 for what can be tens of thousands of patches.
    patches_in = quilt.unstitch(torch.from_numpy(np.ascontiguousarray(img))[None, None])
    patches_tgt = quilt.unstitch(torch.from_numpy(np.ascontiguousarray(tar))[None, None])

    return [
        (
            np.ascontiguousarray(patches_in[k, 0].numpy()),
            np.ascontiguousarray(patches_tgt[k, 0].numpy()),
        )
        for k in range(len(patches_in))
    ]


def tile_denoise_datasets(
    datasets: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    window: int,
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]] | None:
    """Replace each full-resolution pair with its window-sized patches.

    Mirrors :func:`tiling.tile_datasets`' contract exactly — same in/out dict
    shape, same "return ``None`` if *cancel_cb* asked to stop partway" — but
    routes through :func:`tile_denoise_pair` (no weeding, single channel).
    """
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for split, pairs in datasets.items():
        patches: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (inp, tgt) in enumerate(pairs):
            if cancel_cb is not None and cancel_cb():
                return None
            before = len(patches)
            patches.extend(tile_denoise_pair(inp, tgt, window))
            if progress_cb is not None:
                progress_cb(f"{split} item {i + 1}/{len(pairs)}: {len(patches) - before} patch(es)")
        out[split] = patches
        if progress_cb is not None:
            progress_cb(f"{split}: {len(pairs)} item(s) → {len(patches)} patch(es) of {window}px")
    return out


# ---------------------------------------------------------------------------
# Geometry for the non-tiled path
# ---------------------------------------------------------------------------


def letterbox_denoise_pair(inp: np.ndarray, tgt: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize-keep-aspect + pad an ``(input, target)`` grayscale pair to
    ``size`` x ``size``.

    :func:`train_common.letterbox` cannot be used here: it pads the *label*
    with :data:`train_common.IGNORE_INDEX` (255) and nearest-resizes it as
    class indices. For a regression target 255 is not "ignore", it is
    *maximum brightness* — a bright frame the network would be trained to
    reproduce. This variant treats both sides as images: identical bilinear
    resize, identical zero padding, so they stay pixel-aligned.

    With ``tiling=True`` (the default) every patch already arrives exactly
    ``size`` x ``size`` and this short-circuits, exactly as
    :func:`train_common.letterbox` does. The non-tiled path is handled
    explicitly rather than left to rely on that short-circuit never being
    missed — but it is still the worse option for a denoiser, because
    resampling a noisy image correlates neighbouring pixels and so weakens the
    per-pixel noise independence both schemes are built on. The caller logs a
    warning; see :func:`run_denoise_training_loop`.
    """
    from PIL import Image as PILImage  # noqa: PLC0415

    if inp.shape != tgt.shape:
        raise ValueError(f"Denoiser input/target shapes differ: {inp.shape} vs {tgt.shape}")
    h, w = inp.shape[:2]
    if (h, w) == (size, size):
        return inp, tgt

    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - nh) // 2, (size - nw) // 2

    out: list[np.ndarray] = []
    for arr in (inp, tgt):
        resized = np.asarray(PILImage.fromarray(arr).resize((nw, nh), PILImage.Resampling.BILINEAR))
        canvas = np.zeros((size, size), dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = resized  # noqa: E203
        out.append(canvas)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Noise2Void blind-spot masking
# ---------------------------------------------------------------------------


def _mirror_offset(centre: np.ndarray, offset: np.ndarray, n: int) -> np.ndarray:
    """``centre + offset`` along one axis, folded back inside ``[0, n)`` by
    flipping the offset's sign rather than by clamping the index.

    Both of the obvious alternatives silently break the blind spot at the image
    border, by mapping some neighbour offset back onto the pixel itself:

    * clipping — ``centre=0, offset=-1`` clips to ``0``;
    * index reflection (``-idx``) — ``centre=1, offset=-2`` reflects ``-1`` to
      ``1``.

    Flipping the offset instead gives ``centre - offset``, which differs from
    ``centre`` for every non-zero offset. It stays in range whenever the axis
    is at least ``2 * radius + 1`` long (true of any real patch); the final
    clip is a guard for degenerately small arrays, where the caller's
    donor-is-not-the-centre resampling catches the leftover case.
    """
    raw = centre + offset
    folded = np.where((raw < 0) | (raw >= n), centre - offset, raw)
    return np.clip(folded, 0, n - 1)


def n2v_mask_and_replace(
    patch: np.ndarray,
    rng: np.random.Generator,
    *,
    fraction: float = DEFAULT_N2V_MASK_FRACTION,
    neighbourhood: int = DEFAULT_N2V_NEIGHBOURHOOD,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Noise2Void masking to one ``(H, W)`` patch.

    Returns ``(masked_patch, mask)`` — a **copy** of *patch* in which each
    selected pixel has been overwritten with the value of a random neighbour
    from a ``neighbourhood`` x ``neighbourhood`` window, and the boolean mask
    of the selected coordinates.

    The blind spot is the whole point, and it is enforced twice over: a donor
    is never the masked pixel itself, and never another masked pixel. Without
    the second condition a masked pixel's original value could still reach the
    input by being copied into some *other* masked pixel's position — a leak
    the model's receptive field is wide enough to exploit, at which point it
    learns the identity function and denoises nothing.

    The count is exact (``round(H * W * fraction)``, at least 1) rather than a
    per-pixel coin flip. Beyond making the fraction testable, it guarantees a
    non-empty mask: ``dlsia``'s ``MSELossMasked`` divides by ``masks.sum()``,
    so an all-false mask would return NaN and poison the run.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"n2v mask fraction must be in (0, 1), got {fraction}")
    if neighbourhood < 3 or neighbourhood % 2 == 0:
        raise ValueError(f"n2v neighbourhood must be an odd size >= 3, got {neighbourhood}")

    h, w = patch.shape[:2]
    n_pixels = h * w
    n_masked = min(max(1, int(round(n_pixels * fraction))), n_pixels - 1)

    flat = rng.choice(n_pixels, size=n_masked, replace=False, shuffle=False)
    ys, xs = np.divmod(flat, w)
    mask = np.zeros((h, w), dtype=bool)
    mask[ys, xs] = True

    radius = neighbourhood // 2
    donor_y = np.empty(n_masked, dtype=np.int64)
    donor_x = np.empty(n_masked, dtype=np.int64)
    # Positions in ys/xs still without a usable donor. An explicit index array
    # rather than a boolean `todo[todo] = ...`, which would index an array with
    # itself while writing to it.
    pending = np.arange(n_masked)

    for _ in range(_N2V_MAX_RESAMPLE):
        if pending.size == 0:
            break
        dy = rng.integers(-radius, radius + 1, size=pending.size)
        dx = rng.integers(-radius, radius + 1, size=pending.size)
        cand_y = _mirror_offset(ys[pending], dy, h)
        cand_x = _mirror_offset(xs[pending], dx, w)
        donor_y[pending] = cand_y
        donor_x[pending] = cand_x
        # Unusable if it landed on the pixel itself, or on another masked pixel.
        unusable = ((cand_y == ys[pending]) & (cand_x == xs[pending])) | mask[cand_y, cand_x]
        pending = pending[unusable]

    if pending.size:
        # Degenerate only at absurd mask fractions, where a 5x5 window can be
        # entirely masked. Fall back to a uniformly-drawn unmasked pixel from
        # anywhere in the patch: a worse donor (no locality) but one that keeps
        # the blind-spot guarantee, which correctness depends on and locality
        # does not. `fraction < 1` guarantees the candidate set is non-empty.
        unmasked = np.flatnonzero(~mask.ravel())
        picked = rng.choice(unmasked, size=pending.size, replace=True)
        donor_y[pending], donor_x[pending] = np.divmod(picked, w)
        logger.debug("n2v: %d pixel(s) fell back to a non-local donor", pending.size)

    masked = patch.copy()
    masked[ys, xs] = patch[donor_y, donor_x]
    return masked, mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Key of the returned validation metric. Named for what it actually measures —
# Pearson correlation between the prediction and the (still noisy) target — so
# it can't be read as an image-quality score. It is NOT PSNR and must not be
# presented as one: for Noise2Noise the target is a different noisy slice, and
# for Noise2Void it is the original noisy pixel, so perfect agreement with it
# would mean the model had learned to reproduce noise. Rising early then
# plateauing is the convergence signal to read it for.
VAL_METRIC_KEY = "val_noisy_target_pearson"


def _pearson(pred: "Any", target: "Any") -> float:
    """dlsia's regression metric, coerced to a plain finite float.

    ``torch.corrcoef`` returns NaN when either input has zero variance (a
    constant patch — an all-padding window, or a model that has collapsed to a
    constant early in training). Reporting NaN would propagate into
    ``metrics.json`` and render as a broken value in the runs list, so it
    degrades to 0.0, which is also the honest reading: no measured agreement.
    """
    from dlsia.core.train_scripts import regression_metrics  # noqa: PLC0415

    if target.numel() < 2:
        return 0.0
    value = float(regression_metrics(pred, target))
    return value if math.isfinite(value) else 0.0


def _prep_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    indices: Sequence[int],
    *,
    image_size: int,
    scheme: str,
    to_tensor_fn: Callable[[Any], Any],
    rng: np.random.Generator,
    flip_augment: bool,
    mask_fraction: float,
    neighbourhood: int,
) -> tuple[Any, Any, Any]:
    """Assemble one batch as ``(inputs, targets, mask_or_None)`` CPU tensors.

    For ``"n2v"`` the mask is drawn fresh here, per patch and per epoch, so a
    patch seen ten times supervises ten different pixel subsets. ``"ae"`` needs no
    per-epoch randomisation — its objective is fixed (reconstruct this patch),
    and the bottleneck, not a fresh perturbation, is what stops it cheating.
    """
    import torch  # noqa: PLC0415

    imgs, tgts, masks = [], [], []
    for i in indices:
        inp, tgt = pairs[int(i)]
        inp, tgt = letterbox_denoise_pair(inp, tgt, image_size)
        flip = bool(flip_augment and rng.random() < 0.5)
        if flip:
            inp = np.ascontiguousarray(inp[:, ::-1])
        if scheme == "ae":
            # Pure self-reconstruction: the target IS the input. Safe only
            # because the 'cnn_ae' architecture has no skip connections, so the
            # bottleneck cannot pass the input through unchanged (the schema
            # refuses this scheme on TUNet for exactly that reason). Like n2v,
            # an ae item is a slice paired with itself, so the pair's `tgt` is
            # ignored — the target must be the same flip `inp` just took.
            imgs.append(to_tensor_fn(inp))
            tgts.append(to_tensor_fn(inp))
        elif scheme == "n2v":
            # Target is the ORIGINAL patch; the model input is the masked copy.
            # `tgt` is ignored — an n2v item is a slice paired with itself, and
            # the target has to be the same flip of it that `inp` just took.
            masked, mask = n2v_mask_and_replace(
                inp, rng, fraction=mask_fraction, neighbourhood=neighbourhood
            )
            imgs.append(to_tensor_fn(masked))
            tgts.append(to_tensor_fn(inp))
            masks.append(torch.from_numpy(mask).unsqueeze(0))
        else:
            if flip:
                tgt = np.ascontiguousarray(tgt[:, ::-1])
            imgs.append(to_tensor_fn(inp))
            tgts.append(to_tensor_fn(tgt))

    batch_mask = torch.stack(masks) if masks else None
    return torch.stack(imgs), torch.stack(tgts), batch_mask


def _denoise_loss(pred: Any, target: Any, mask: Any, criterion: Any) -> Any:
    """Masked MSE for Noise2Void, plain MSE for Noise2Noise."""
    return criterion(pred, target) if mask is None else criterion(pred, target, mask)


def evaluate_denoise(
    val_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    image_size: int,
    scheme: str,
    to_tensor_fn: Callable[[Any], Any],
    forward_fn: Callable[[Any], Any],
    device: str,
    criterion: Any,
    batch_size: int,
    seed: int,
    mask_fraction: float,
    neighbourhood: int,
) -> tuple[float, float]:
    """Average validation loss and target-correlation over *val_pairs*.

    Mirrors :func:`train_common._evaluate`'s batching and its per-sample
    weighting of a batch-mean loss, with mIoU replaced by
    :data:`VAL_METRIC_KEY`'s Pearson correlation — and, for Noise2Void, the
    correlation restricted to the masked coordinates, which are the only
    positions the model was ever asked to predict.

    The mask RNG is re-seeded from *seed* on every call, so epoch-to-epoch
    changes in the metric come from the model rather than from a different
    random subset of pixels being scored each time.
    """
    import torch  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    total_loss = 0.0
    total_corr = 0.0
    n = len(val_pairs)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk_idx = range(start, min(start + batch_size, n))
            imgs, tgts, mask = _prep_batch(
                val_pairs,
                list(chunk_idx),
                image_size=image_size,
                scheme=scheme,
                to_tensor_fn=to_tensor_fn,
                rng=rng,
                flip_augment=False,
                mask_fraction=mask_fraction,
                neighbourhood=neighbourhood,
            )
            imgs, tgts = imgs.to(device), tgts.to(device)
            mask = mask.to(device) if mask is not None else None

            pred = forward_fn(imgs)
            chunk_n = len(chunk_idx)
            total_loss += float(_denoise_loss(pred, tgts, mask, criterion).item()) * chunk_n
            if mask is None:
                total_corr += _pearson(pred, tgts) * chunk_n
            else:
                total_corr += _pearson(pred[mask], tgts[mask]) * chunk_n
    return total_loss / n, total_corr / n


def run_denoise_training_loop(
    *,
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    val_pairs: list[tuple[np.ndarray, np.ndarray]],
    image_size: int,
    training_scheme: str,
    epochs: int,
    batch_size: int,
    seed: int,
    flip_augment: bool,
    to_tensor_fn: Callable[[Any], Any],
    forward_fn: Callable[[Any], Any],
    trainable_params: list[Any],
    lr: float,
    device: str,
    n2v_mask_fraction: float = DEFAULT_N2V_MASK_FRACTION,
    n2v_neighbourhood: int = DEFAULT_N2V_NEIGHBOURHOOD,
    make_optimizer_fn: Callable[[list[Any], float], Any] | None = None,
    on_batch: Callable[[], bool] | None = None,
    on_epoch: Callable[[int, float, float | None, float | None], bool] | None = None,
    set_train_mode: Callable[[bool], None] | None = None,
) -> dict[str, Any]:
    """Epoch/batch loop for the self-supervised denoiser families.

    The regression counterpart of :func:`train_common.run_training_loop`, with
    the same optimizer (AdamW + cosine annealing over ``epochs``), the same
    cooperative-cancel protocol (``on_batch``/``on_epoch`` return ``True`` to
    stop; a cancelled run still returns its partial metrics so the caller can
    save a checkpoint), and the same ``set_train_mode`` handling around
    validation — TUNet uses BatchNorm2d, whose running stats must not be
    updated while validating.

    ``training_scheme`` selects the objective:

    * ``"n2n"`` — plain ``nn.MSELoss`` between the input slice's prediction and
      the paired adjacent slice.
    * ``"n2v"`` — ``dlsia.core.custom_losses.MSELossMasked`` evaluated only at
      the freshly-masked blind-spot coordinates (see
      :func:`n2v_mask_and_replace`). dlsia's implementation is used rather than
      a hand-rolled ``(err * mask).sum() / mask.sum()`` so the normalisation
      convention matches the rest of the dlsia stack.

    Returns :func:`train_common.run_training_loop`'s dict shape —
    ``{"epochs_completed", "final_train_loss", "final_val_loss", "cancelled"}``
    — with mIoU replaced by :data:`VAL_METRIC_KEY`. Read that field as a
    convergence signal, never as image quality: both schemes' targets are
    themselves noisy.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    if not train_pairs:
        raise ValueError("No training data: the denoiser needs at least one slice in scope")
    if training_scheme not in {"n2n", "n2v", "ae"}:
        raise ValueError(f"Unknown denoiser training scheme: {training_scheme!r}")

    sample_shape = train_pairs[0][0].shape[:2]
    if sample_shape != (image_size, image_size):
        # Not fatal — letterbox_denoise_pair handles it, and it is the expected
        # state with tiling disabled — but resampling noisy data undermines the
        # per-pixel noise independence both schemes assume, so say so once.
        logger.warning(
            "Denoiser training on %s items that are not already %dpx: they will be letterboxed "
            "(bilinear resize + zero pad), which correlates neighbouring noise. Enable tiling to "
            "train on native-resolution windows instead.",
            sample_shape,
            image_size,
        )

    if training_scheme == "n2v":
        from dlsia.core.custom_losses import MSELossMasked  # noqa: PLC0415

        criterion: Any = MSELossMasked()
    else:
        criterion = nn.MSELoss()

    rng = np.random.default_rng(seed)
    optimizer = (make_optimizer_fn or (lambda params, lr_: torch.optim.AdamW(params, lr=lr_, weight_decay=0.01)))(
        trainable_params, lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    cancelled = False
    final_train_loss = 0.0
    final_val_loss: float | None = None
    val_metric: float | None = None
    epochs_completed = 0

    for epoch in range(epochs):
        order = rng.permutation(len(train_pairs))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]  # noqa: E203
            imgs, tgts, mask = _prep_batch(
                train_pairs,
                batch_idx,
                image_size=image_size,
                scheme=training_scheme,
                to_tensor_fn=to_tensor_fn,
                rng=rng,
                flip_augment=flip_augment,
                mask_fraction=n2v_mask_fraction,
                neighbourhood=n2v_neighbourhood,
            )
            imgs, tgts = imgs.to(device), tgts.to(device)
            mask = mask.to(device) if mask is not None else None

            pred = forward_fn(imgs)
            loss = _denoise_loss(pred, tgts, mask, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            n_batches += 1
            if on_batch is not None and on_batch():
                cancelled = True
                break
        scheduler.step()
        final_train_loss = epoch_loss / max(1, n_batches)
        epochs_completed = epoch + 1

        if val_pairs and not cancelled:
            if set_train_mode is not None:
                set_train_mode(False)
            final_val_loss, val_metric = evaluate_denoise(
                val_pairs,
                image_size=image_size,
                scheme=training_scheme,
                to_tensor_fn=to_tensor_fn,
                forward_fn=forward_fn,
                device=device,
                criterion=criterion,
                batch_size=batch_size,
                seed=seed,
                mask_fraction=n2v_mask_fraction,
                neighbourhood=n2v_neighbourhood,
            )
            if set_train_mode is not None:
                set_train_mode(True)

        if on_epoch is not None and on_epoch(epochs_completed, final_train_loss, final_val_loss, val_metric):
            cancelled = True
        if cancelled:
            break

    return {
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        VAL_METRIC_KEY: val_metric,
        "cancelled": cancelled,
    }
