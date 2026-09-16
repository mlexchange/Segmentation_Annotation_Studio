"""Expose a directory of TIFF slices as a streamable 3-D Zarr volume.

Why this exists
---------------
Tiled 0.2.12 mounts a Zarr v2 router at ``/zarr/v2``: every array node is
already readable as a Zarr store, with ``.zarray`` synthesized from the Tiled
structure and ``/{i.j.k}`` serving a chunk. The 3-D viewer streams from that
directly — no export, no second endpoint.

Two things stop a TIFF stack from working that way today:

1. :mod:`ingest` writes one 2-D array **per file**, so ``/zarr/v2/<stack>`` is a
   *group of N 2-D arrays* — not a volume. That layout is load-bearing for the
   2-D canvas (``arrays._stack_keys`` / ``read_slice``) and for per-frame Browse
   metadata, so it stays exactly as it is. This module registers a 3-D **view
   alongside** it; nothing here changes how slices are read.
2. Tiled's ``.zattrs`` returns ``metadata["attributes"]`` verbatim, so OME-NGFF
   ``multiscales`` appears only if something writes it. This module writes it.

Layout produced
---------------
The same OME-NGFF shape :mod:`zarr_source` already documents, so both paths look
identical to the viewer. The volume is a **sidecar** of the dataset, keyed with a
``__volume`` suffix in the style of the existing ``__v_thumbs`` and ``__masks``
siblings, because the per-slice container occupies the unsuffixed key::

    <stem>/                # untouched: the per-slice 2-D nodes Annotate reads
    <stem>__volume/        # container; .zattrs carries "multiscales"
        scale0             # (N, H, W) — the TIFF files, registered IN PLACE
        pyramid/scale1     # downsampled, in an on-disk Zarr sidecar
        pyramid/scale2     # ...

Why the coarse levels go to disk rather than into Tiled
-------------------------------------------------------
Writing them with ``write_array`` would be less machinery, but Tiled 0.2.12's
``/zarr/v2`` chunk route only works for **externally-managed** arrays: for one it
writes itself, ``entry.read(slice=<tuple>)`` reaches ``NDSlice.__getitem__`` with
a tuple and raises ``TypeError: tuple indices must be integers or slices``. The
``/api/v1`` block route serves the same array fine, so this is specific to the
Zarr façade — and the Zarr façade is the whole point here. Measured on a real
604 x 2560 x 2560 stack: ``scale0`` (external TIFF sequence) served a 19 MB chunk
in 0.2 s while every ``write_array`` level returned HTTP 500.

Writing a normal Zarr store and registering it in place sidesteps that, and has
the same shape as :mod:`zarr_source`'s path — one fewer special case, not one
more. Revisit if the upstream route is fixed.

Why a pyramid is not optional
-----------------------------
The renderer only lists levels that fit ``maxTextureDimension3D`` (commonly
2048) and its voxel budget. A single-level ``multiscales`` over a
2000x3232x3232 stack therefore offers *nothing renderable*, which looks
identical to a broken viewer. The coarse levels are the ones actually drawn;
``scale0`` is never fully read for the 3-D view, and earns its place by making
full-resolution ROI work possible later.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import HTTPException

import ingest as ingest_mod
from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger("tiff_stack_source")

TIFF_SUFFIXES: tuple[str, ...] = (".tif", ".tiff")

# Longest edge (voxels) the finest GENERATED level may have. 2048 matches
# WebGPU's guaranteed-minimum `maxTextureDimension3D` — the viewer picks
# whichever registered level is the coarsest that still fits the actual
# device's limit, so a level above what a given GPU can take is simply
# skipped, never a hard failure. Raising this raises fidelity but also
# memory/time to build it: this module assembles each generated level fully
# in RAM before writing it, so cost scales with actual voxel count
# (nx * ny * nz, not nx**3 — z is normally far smaller than the in-plane
# edges for a tomography stack).
TARGET_DIM = 2048

# How many levels to generate below scale0. Three gives the renderer a choice of
# detail without the cost growing: each is 1/8 the voxels of the one above.
GENERATED_LEVELS = 3

#: Key of the sub-group holding the generated levels, inside the volume node.
PYRAMID_KEY = "pyramid"


def pyramid_cache_root() -> Path:
    """Directory the generated Zarr pyramids are written to.

    Deliberately *not* next to the source TIFFs: beamline reconstruction
    directories are routinely read-only or on shared storage, and failing to
    register a volume because the source's parent cannot be written to would be a
    confusing way to discover that. Override with ``VOLUME_CACHE_DIR``.
    """
    default = Path(__file__).resolve().parent.parent / ".tiled" / "volumes"
    return Path(os.getenv("VOLUME_CACHE_DIR", str(default))).expanduser().resolve()


def _resolve_dir(raw: str) -> Path:
    """Expand and validate a user-supplied absolute path to a TIFF directory.

    Raises:
        HTTPException: 4xx with a user-facing message, so the UI never has to
            surface a traceback.
    """
    if not (raw or "").strip():
        raise HTTPException(400, "Enter the path to a directory of TIFF slices.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, f"Path must be absolute: {raw!r}")
    if not path.exists():
        raise HTTPException(404, f"No such path: {path}")
    if not path.is_dir():
        raise HTTPException(422, f"Not a directory: {path}")
    return path.resolve()


def tiff_files(path: Path) -> list[Path]:
    """Sorted TIFF files directly inside *path*.

    Sorted lexically, which is slice order for the zero-padded names tomography
    reconstruction writes. Non-padded names (``img_2`` before ``img_10``) would
    sort wrong — :func:`inspect_tiff_stack` rejects those rather than silently
    building a shuffled volume.
    """
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES
    )


def _zero_padding_is_consistent(files: list[Path]) -> bool:
    """True when lexical order is numeric order.

    Only meaningful when the names carry numbers at all; a set of names with no
    digits has nothing to get wrong, so it passes.
    """
    import re

    numbers: list[int] = []
    for f in files:
        match = re.search(r"(\d+)(?!.*\d)", f.stem)
        if not match:
            return True  # no numbering scheme to violate
        numbers.append(int(match.group(1)))
    return numbers == sorted(numbers)


def pyramid_plan(
    shape: tuple[int, int, int],
    target_dim: int = TARGET_DIM,
    levels: int = GENERATED_LEVELS,
) -> list[dict[str, Any]]:
    """Downsample factors and shapes for the levels to generate below ``scale0``.

    Factors are **per axis**, and each is the smallest power of two bringing that
    axis to ``target_dim`` or below. Powers of two keep the block mean exact and
    the coordinate transforms simple.

    Per-axis rather than one shared factor because tomography stacks are often
    strongly anisotropic — 4 slices of 4096x4096 is an ordinary shape. A single
    factor chosen for the wide axes would ask for 4//16 = 0 slices, so *every*
    level would be dropped as degenerate and the volume would end up with no
    renderable level at all: indistinguishable, from the outside, from a broken
    viewer. An axis therefore stops halving once it reaches 1.

    Args:
        shape: ``(n_slices, height, width)`` of the full-resolution stack.
        target_dim: Longest edge allowed for the finest generated level.
        levels: How many levels to generate.

    Returns:
        One dict per level with ``path``, ``factor`` (a 3-list, z/y/x) and
        ``shape``. Empty when the source is already at or below ``target_dim`` on
        every axis — there is then nothing to generate and ``scale0`` alone is
        renderable.
    """
    if any(s <= 0 for s in shape):
        return []
    if max(shape) <= target_dim:
        return []

    def axis_factor(size: int, cap: int) -> int:
        f = 1
        # `size // (f * 2) >= 1` is the guard that keeps a short axis alive.
        while size / f > cap and size // (f * 2) >= 1:
            f *= 2
        return f

    factors = [axis_factor(s, target_dim) for s in shape]

    plan: list[dict[str, Any]] = []
    for index in range(levels):
        level_factors = [
            # Halve again only where the axis can still spare it; an axis already
            # at 1 stays at 1 rather than dragging the level into degeneracy.
            f * 2 if index and shape[axis] // (f * 2) >= 1 else f
            for axis, f in enumerate(factors)
        ]
        factors = level_factors
        level_shape = [shape[axis] // f for axis, f in enumerate(factors)]
        if min(level_shape) < 1:
            break
        # A level identical to the one above adds bytes and no detail.
        if plan and level_shape == plan[-1]["shape"]:
            break
        plan.append(
            {
                "path": f"scale{len(plan) + 1}",
                "factor": list(factors),
                "shape": level_shape,
            }
        )
    return plan


def multiscales_metadata(
    name: str,
    plan: list[dict[str, Any]],
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    include_scale0: bool = True,
) -> dict[str, Any]:
    """OME-NGFF ``multiscales`` for the generated levels, and optionally ``scale0``.

    Returned under an ``attributes`` key because Tiled's ``.zattrs`` route
    returns ``metadata["attributes"]`` verbatim — put it anywhere else and the
    viewer sees an empty attribute set and reports "missing multiscales".

    Args:
        include_scale0: True when the full-resolution data is registered in place
            as a ``scale0`` sibling (the TIFF-directory path). False when the
            volume was built from data already in Tiled, where full resolution
            stays in the per-slice nodes and is not duplicated — the viewer never
            loads a level that large anyway.
    """
    datasets = []
    if include_scale0:
        datasets.append(
            {
                "path": "scale0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(v) for v in voxel_size]}
                ],
            }
        )
    for level in plan:
        factors = level["factor"]
        datasets.append(
            {
                # Generated levels live in the sidecar sub-group, so the path the
                # viewer follows is nested. Tiled serves nested groups fine — the
                # Zarr volumes `zarr_source` registers use `scale0/image`.
                "path": f"{PYRAMID_KEY}/{level['path']}",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [float(v) * float(f) for v, f in zip(voxel_size, factors)],
                    }
                ],
            }
        )
    return {
        "attributes": {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": name,
                    "axes": [
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": datasets,
                }
            ]
        }
    }


def block_mean(block: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Mean-downsample a ``(k, H, W)`` block to ``(H//fy, W//fx)``.

    Averages across the whole block in z as well as over each ``fy x fx`` tile,
    so one output voxel is the mean of every input voxel it covers. Trailing
    rows/columns that do not fill a tile are dropped rather than partially
    averaged — a partial tile would be brighter or darker than its neighbours
    purely because of where the edge fell.

    Computed in float32 regardless of input dtype (uint16 sums overflow fast),
    then cast back by the caller.
    """
    if fy < 1 or fx < 1:
        raise ValueError("downsample factors must be >= 1")
    h = (block.shape[1] // fy) * fy
    w = (block.shape[2] // fx) * fx
    trimmed = block[:, :h, :w].astype(np.float32, copy=False)
    return trimmed.reshape(trimmed.shape[0], h // fy, fy, w // fx, fx).mean(axis=(0, 2, 4))


def inspect_tiff_stack(raw_path: str) -> dict[str, Any]:
    """Describe a TIFF directory and the pyramid that would be built for it.

    Reads only the first file, so this is cheap enough to call on every keystroke
    in a path field.

    Raises:
        HTTPException: 4xx with a user-facing message for anything unusable.
    """
    import tifffile

    path = _resolve_dir(raw_path)
    files = tiff_files(path)
    if not files:
        raise HTTPException(422, f"No .tif/.tiff files directly inside {path.name!r}.")
    if len(files) < 2:
        raise HTTPException(
            422,
            f"{path.name!r} holds a single image, not a volume. The 3-D view needs a stack.",
        )
    if not _zero_padding_is_consistent(files):
        raise HTTPException(
            422,
            f"{path.name!r} has inconsistently numbered files (e.g. 'img_2' next to "
            "'img_10'), so filename order is not slice order. Zero-pad the numbers "
            "and try again — registering as-is would build a shuffled volume.",
        )

    try:
        first = tifffile.imread(str(files[0]))
    except Exception as exc:  # noqa: BLE001 — surface as a clean 422
        raise HTTPException(422, f"Could not read {files[0].name!r}: {exc}") from exc
    if first.ndim != 2:
        raise HTTPException(
            422,
            f"{files[0].name!r} is {first.ndim}-D; this path expects one 2-D slice per file.",
        )

    shape = (len(files), int(first.shape[0]), int(first.shape[1]))
    plan = pyramid_plan(shape)
    return {
        "name": path.name,
        "path": str(path),
        "n_slices": shape[0],
        "height": shape[1],
        "width": shape[2],
        "dtype": str(first.dtype),
        "full_shape": list(shape),
        "pyramid_plan": plan,
        # Surfaced so the UI can say "this will take a while" honestly: every
        # source slice is read once to build the levels.
        "slices_to_read": shape[0] if plan else 0,
    }


def _build_level(
    files: list[Path],
    factor: list[int],
    out_shape: list[int],
    dtype: np.dtype,
    on_slice: Callable[[], None] | None = None,
) -> np.ndarray:
    """Read the source stack and mean-downsample it by the per-axis *factor*.

    Reads at most ``factor[0]`` slices at a time, so peak memory is set by the
    output level plus a handful of input slices — never by the whole stack,
    which can be tens of GB.
    """
    import tifffile

    fz, fy, fx = (int(f) for f in factor)
    out = np.empty(out_shape, dtype=np.float32)
    for z in range(out_shape[0]):
        block = []
        for k in range(fz):
            index = z * fz + k
            if index >= len(files):
                break
            block.append(tifffile.imread(str(files[index])))
            if on_slice:
                on_slice()
        out[z] = block_mean(np.stack(block), fy, fx)
    return _cast_like(out, dtype)


def downsample_array(volume: np.ndarray, factor: list[int]) -> np.ndarray:
    """Mean-downsample an in-memory volume by the per-axis *factor*.

    Used to build each coarse level from the level above rather than from the
    source. Because the factors are powers of two, averaging an average over
    matching block sizes gives the same result as averaging the source directly,
    while reading every source slice **once** instead of once per level. On a
    604 x 2560 x 2560 float32 stack that is the difference between ~16 GB of
    reads and ~48 GB.

    (The two agree exactly only where the divisions are exact; a trailing partial
    tile is dropped at each step, so the very last row/column of a level may
    differ from a direct downsample. That is a sub-voxel edge effect on a preview
    level, not something a viewer can show.)
    """
    fz, fy, fx = (int(f) for f in factor)
    z = (volume.shape[0] // fz) * fz
    y = (volume.shape[1] // fy) * fy
    x = (volume.shape[2] // fx) * fx
    trimmed = volume[:z, :y, :x].astype(np.float32, copy=False)
    return trimmed.reshape(z // fz, fz, y // fy, fy, x // fx, fx).mean(axis=(1, 3, 5))


def write_pyramid_store(key: str, levels: dict[str, np.ndarray]) -> Path:
    """Write the generated levels as a Zarr v2 group and return its path.

    One store per dataset, under :func:`pyramid_cache_root`, replacing any
    previous build for the same key — a re-registration should not accumulate
    stale copies of a volume.

    Written as Zarr **v2** to match what Tiled's ``/zarr/v2`` façade and the
    viewer's OME-NGFF reader both speak.
    """
    import zarr

    root = pyramid_cache_root() / key
    root.mkdir(parents=True, exist_ok=True)
    store_path = root / f"{PYRAMID_KEY}.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    group = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    for name, array in levels.items():
        group.create_array(
            name,
            shape=array.shape,
            dtype=array.dtype,
            # One chunk per output slice: matches how the viewer streams, and
            # keeps any single request small.
            chunks=(1, *array.shape[1:]),
        )[:] = array
    return store_path


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float working array back to the source dtype, rounding integers.

    Keeps the rendered volume in the same intensity units as the 2-D canvas
    instead of silently becoming float — and rounds rather than truncating, so a
    mean of 1.5 does not become 1.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


#: Suffix marking the 3-D volume node as a sidecar of the dataset it describes.
#: Matches the existing ``__v_thumbs`` / ``__masks`` convention.
VOLUME_SUFFIX = "__volume"


def registered_key(path: Path) -> str:
    """The node key for *path*'s 3-D volume.

    Suffixed, because the 2-D per-slice container that Annotate reads is keyed on
    the very same directory name. Without the suffix the volume would land on top
    of it — and since that container holds ingested data, the collision check
    below would (correctly) refuse, making the 3-D view impossible for exactly
    the datasets it is for. A sidecar key lets the two coexist, which is the
    whole design: this is additive, and the 2-D read path is untouched.

    The stem comes from Tiled's own helper rather than assuming ``path.name`` —
    the same care :mod:`zarr_source` takes, and for the same reason: a key
    mismatch makes the collision check inspect a node that is not the one about
    to be created, and Tiled then resolves the real collision itself, deep inside
    registration where its only remedy is deletion.
    """
    from tiled.client.register import Settings

    return f"{Settings.init().key_from_filename(path.name)}{VOLUME_SUFFIX}"


def preflight_tiff_stack(
    server_uri: str | None, raw_path: str, container_path: str = "browse"
) -> dict[str, Any]:
    """Report whether registering *raw_path* would collide, changing nothing."""
    path = _resolve_dir(raw_path)
    key = registered_key(path)
    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    parts = [p for p in container_path.strip("/").split("/") if p]
    target = ingest_mod._walk(client, parts)
    existing = None
    if target is not None and key in ingest_mod._child_keys(target):
        node = target[key]
        meta: dict[str, Any] = {}
        try:
            meta = dict(getattr(node, "metadata", {}) or {})
        except Exception:  # noqa: BLE001 — best-effort description
            pass
        children: list[str] = []
        try:
            children = list(node)
        except Exception:  # noqa: BLE001
            pass
        existing = {
            "child_count": len(children),
            # Only a previous registration by THIS module is safe to replace:
            # dropping it removes catalog rows and the generated levels, never
            # the source TIFFs. Anything else may be internally-managed data,
            # where deleting the node deletes the files.
            "external": meta.get("source_format") == "tiff-stack-3d",
            "sample_name": meta.get("sample_name") or "",
        }
    return {"key": key, "exists": existing is not None, "existing": existing}


def register_tiff_stack(
    server_uri: str | None,
    raw_path: str,
    container_path: str = "browse",
    description: str = "",
    on_conflict: str = "fail",
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Register a TIFF directory as a 3-D multiscale volume, copying no slices.

    ``scale0`` points at the TIFF files where they already are. Only the
    generated (small) levels are written into Tiled.

    Args:
        server_uri: Connected Tiled server URI.
        raw_path: Absolute path to the directory of TIFF slices.
        container_path: Slash-separated target container (e.g. ``browse``).
        description: Optional keyword(s), as ingest stores them, so the volume is
            filterable in Browse.
        on_conflict: ``"fail"``, ``"replace"`` or ``"skip"`` when the key exists.
        progress: Optional ``(message, done, total)`` callback for the job UI.

    Returns:
        The inspection result plus the registered ``key`` and ``tiled_path``.
    """
    info = inspect_tiff_stack(raw_path)
    path = Path(info["path"])
    files = tiff_files(path)
    plan: list[dict[str, Any]] = info["pyramid_plan"]
    description = (description or "").strip()
    keywords = ingest_mod.parse_keywords(description)
    if on_conflict not in ingest_mod.ON_CONFLICT_MODES:
        on_conflict = "fail"

    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    parts = [p for p in container_path.strip("/").split("/") if p]
    target = ingest_mod._ensure_container(client, parts)

    key = registered_key(path)
    if key in ingest_mod._child_keys(target):
        existing = preflight_tiff_stack(server_uri, raw_path, container_path)["existing"]
        if on_conflict == "skip":
            return {**info, "key": key, "tiled_path": "/".join([*parts, key]), "skipped": True}
        if on_conflict == "replace":
            if not (existing or {}).get("external"):
                raise HTTPException(
                    409,
                    f"{key!r} already exists in {container_path!r} and does not hold a "
                    "registered TIFF volume. Replacing it could delete uploaded data — "
                    "load into a different container, or remove that dataset yourself first.",
                )
            target.delete_contents(key, recursive=True, external_only=False)
        else:
            raise HTTPException(
                409,
                f"{key!r} already exists in {container_path!r}. Choose Replace or a "
                "different destination.",
            )

    volume = target.create_container(key=key, metadata={})

    # scale0: the files themselves, registered in place as ONE 3-D array.
    # `register_image_sequence` builds a TiffSequenceAdapter over the sorted list
    # and stores external Assets — shape (N, H, W), one chunk per slice, which is
    # exactly the granularity a streaming volume viewer wants.
    from tiled.client.register import Settings, register_image_sequence

    if progress:
        progress("Registering full-resolution slices", 0, info["slices_to_read"])
    try:
        asyncio.run(register_image_sequence(volume, "scale0", files, Settings.init()))
    except Exception as exc:  # noqa: BLE001 — classified for the UI
        logger.warning("tiff sequence registration failed for %s: %s", path, exc)
        raise HTTPException(502, ingest_mod._classify_error(exc)["message"]) from exc

    if ingest_mod._walk(volume, ["scale0"]) is None:
        # register_image_sequence logs and swallows adapter errors, so a missing
        # node here is the signal that registration did not actually happen.
        raise HTTPException(
            502,
            f"Tiled did not register the slices of {key!r}. Check the backend log for "
            "the adapter error, and that the Tiled server can read this path.",
        )

    # Generated levels, built as a cascade: the finest from the TIFF files, each
    # coarser one from the level above (already in memory, and tiny). Reading the
    # source once instead of once per level is the difference between ~16 GB and
    # ~48 GB of I/O on a 604 x 2560 x 2560 float32 stack.
    dtype = np.dtype(info["dtype"])
    total_reads = info["slices_to_read"]
    done = 0
    previous: np.ndarray | None = None
    previous_factor: list[int] | None = None
    built: dict[str, np.ndarray] = {}

    for level in plan:
        label = f"Building {level['path']}"
        if progress:
            progress(label, done, total_reads)

        if previous is None or previous_factor is None:
            def _tick() -> None:
                nonlocal done
                done += 1
                # Throttled: one update per 25 slices is smooth enough for a
                # progress bar and keeps the job lock uncontended.
                if progress and done % 25 == 0:
                    progress(label, done, total_reads)

            working = _build_level(files, level["factor"], level["shape"], dtype, _tick).astype(
                np.float32, copy=False
            )
        else:
            relative = [
                int(level["factor"][axis] // previous_factor[axis]) for axis in range(3)
            ]
            working = downsample_array(previous, relative)

        built[level["path"]] = _cast_like(working, dtype)
        previous, previous_factor = working, list(level["factor"])

    # Write the generated levels as a plain on-disk Zarr store and register it in
    # place, exactly as an externally-supplied Zarr volume would be — see this
    # module's docstring for why `write_array` cannot be used here.
    if built:
        if progress:
            progress("Writing pyramid", total_reads, total_reads)
        sidecar = write_pyramid_store(key, built)
        from tiled.client.register import register_single_item

        try:
            asyncio.run(
                register_single_item(
                    volume, sidecar, is_directory=True, settings=Settings.init()
                )
            )
        except Exception as exc:  # noqa: BLE001 — classified for the UI
            logger.warning("pyramid registration failed for %s: %s", sidecar, exc)
            raise HTTPException(502, ingest_mod._classify_error(exc)["message"]) from exc

        # A node existing is not enough. When the store sits outside Tiled's
        # `readable_storage`, registration still creates the node — it just has
        # no children, and every chunk request then 500s at read time, far from
        # the cause. Check for contents, and name the likely fix.
        pyramid_node = ingest_mod._walk(volume, [PYRAMID_KEY])
        if pyramid_node is None or not list(pyramid_node):
            raise HTTPException(
                502,
                f"Tiled registered no levels for {key!r} from {sidecar}. The usual "
                "cause is that this path is not in the Tiled server's "
                "`readable_storage` (see tiled/config.yml) — add it, or point "
                "VOLUME_CACHE_DIR somewhere already readable.",
            )

    meta: dict[str, Any] = {
        "sample_name": key,
        "n_images": info["n_slices"],
        "source_format": "tiff-stack-3d",
        "tiff_dir": str(path),
        "full_shape": info["full_shape"],
        "pyramid_plan": plan,
        **multiscales_metadata(key, plan),
    }
    if description:
        meta["description"] = description
    if keywords:
        meta["keywords"] = keywords
    try:
        volume.update_metadata(metadata=meta)
    except Exception as exc:  # noqa: BLE001 — best-effort; the data is registered
        logger.warning("could not set metadata on %s: %s", key, exc)

    if progress:
        progress("Done", total_reads, total_reads)

    # Spread `info` FIRST so its filesystem "path" cannot shadow the Tiled path
    # the caller needs to open the dataset.
    return {
        **info,
        "key": key,
        "tiled_path": "/".join([*parts, key]),
        "skipped": False,
    }
