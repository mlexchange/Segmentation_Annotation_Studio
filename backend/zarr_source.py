"""Register on-disk Zarr volumes with Tiled, without copying any data.

The drag-and-drop ingest path (:mod:`ingest`) streams image files through the
browser and writes each one into Tiled. That cannot express a tomography volume:
these datasets are 7-56 GB, so the bytes must stay where they are.

Tiled can serve a Zarr store in place — ``.zarr`` maps to ``application/x-zarr``
and ``ZarrGroupAdapter`` reads chunks lazily off disk — so "loading" a volume is
really just *registering* a path. A full-resolution 2560x2560 slice of a
(690, 2560, 2560) float32 store comes back over Tiled's HTTP layer in under
0.2 s, which is comfortably interactive.

Layout
------
The supported stores are OME-NGFF-style multiscale groups::

    <name>.zarr/
        .zattrs          # "multiscales" -> datasets[].path + coordinateTransformations
        scale0/image     # (z, y, x) float32, full resolution
        scale1/image     # downsampled
        ...

After registration the Tiled path of one level is
``<container>/<name>/scale0/image``, which :mod:`arrays` resolves as an
``(N, H, W)`` slice stack with no further plumbing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException

import ingest as ingest_mod
from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger("zarr_source")

# Group-level metadata files that mark a directory as a Zarr store. v2 uses
# `.zgroup`/`.zarray`; v3 uses a single `zarr.json`.
_ZARR_MARKERS: tuple[str, ...] = (".zgroup", ".zarray", "zarr.json")


def _is_zarr_dir(path: Path) -> bool:
    """True if *path* looks like the root of a Zarr store."""
    return path.is_dir() and any((path / marker).exists() for marker in _ZARR_MARKERS)


def _resolve_path(raw: str) -> Path:
    """Expand and validate a user-supplied absolute path to a Zarr store.

    Raises:
        HTTPException: 400/404/422 with a message aimed at the user, so the UI
            never has to surface a traceback.
    """
    if not (raw or "").strip():
        raise HTTPException(400, "Enter the path to a .zarr directory.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, f"Path must be absolute: {raw!r}")
    if not path.exists():
        raise HTTPException(404, f"No such path: {path}")
    if path.is_file():
        # The common near-miss: a zipped store. Zarr can read these via ZipStore,
        # but Tiled registers a directory asset, so say so plainly.
        if path.suffix.lower() == ".zip":
            raise HTTPException(
                422,
                "Zipped Zarr archives are not supported — unzip it first and "
                "point at the resulting .zarr directory.",
            )
        raise HTTPException(422, f"Not a directory: {path}")
    if not _is_zarr_dir(path):
        raise HTTPException(
            422,
            f"{path.name!r} is not a Zarr store (no .zgroup/.zarray/zarr.json "
            "at its root).",
        )
    return path.resolve()


def _multiscale_datasets(path: Path) -> list[dict[str, Any]] | None:
    """Read OME-NGFF ``multiscales`` from ``.zattrs``, if present.

    Returns the ``datasets`` list (each with ``path`` and
    ``coordinateTransformations``), or None when the store is not multiscale.
    """
    attrs_file = path / ".zattrs"
    if not attrs_file.exists():
        return None
    try:
        attrs = json.loads(attrs_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return None
    datasets = multiscales[0].get("datasets")
    return datasets if isinstance(datasets, list) and datasets else None


def _scale_vector(dataset: dict[str, Any]) -> list[float] | None:
    """Pull the ``scale`` transform (voxel size per axis) out of a dataset entry."""
    for transform in dataset.get("coordinateTransformations") or []:
        if transform.get("type") == "scale":
            scale = transform.get("scale")
            if isinstance(scale, list) and len(scale) == 3:
                return [float(v) for v in scale]
    return None


def _walk_arrays(group: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    """Depth-first list of ``(path, array)`` for every 3-D array in *group*."""
    import zarr

    found: list[tuple[str, Any]] = []
    if depth > 3:
        return found
    for key in group.keys():
        try:
            child = group[key]
        except Exception:  # noqa: BLE001 — a broken child shouldn't sink the scan
            continue
        child_path = f"{prefix}/{key}" if prefix else key
        if isinstance(child, zarr.Array):
            if child.ndim == 3:
                found.append((child_path, child))
        else:
            found.extend(_walk_arrays(child, child_path, depth + 1))
    return found


def inspect_zarr(raw_path: str) -> dict[str, Any]:
    """Describe a Zarr store's resolution pyramid without registering anything.

    Args:
        raw_path: Absolute path to a ``.zarr`` directory on the server.

    Returns:
        Dict with ``name``, ``levels`` (finest first) and voxel metadata. Each
        level carries the array's Tiled sub-path, shape, dtype, and its
        downsample factor relative to the finest level — the factor the UI needs
        to explain that a coarse level addresses only every f-th slice.

    Raises:
        HTTPException: 4xx with a user-facing message for anything unusable.
    """
    import zarr

    path = _resolve_path(raw_path)
    try:
        group = zarr.open_group(str(path), mode="r")
    except Exception as exc:  # noqa: BLE001 — surface as a clean 422
        raise HTTPException(422, f"Could not open {path.name!r} as Zarr: {exc}") from exc

    # Prefer the declared multiscales order; fall back to discovering 3-D arrays.
    datasets = _multiscale_datasets(path)
    entries: list[tuple[str, Any, list[float] | None]] = []
    if datasets:
        for dataset in datasets:
            key = str(dataset.get("path") or "").strip("/")
            if not key:
                continue
            try:
                arr = group[key]
            except Exception:  # noqa: BLE001 — declared but missing; skip it
                logger.warning("multiscales lists %r but it is not readable", key)
                continue
            if getattr(arr, "ndim", 0) == 3:
                entries.append((key, arr, _scale_vector(dataset)))
    if not entries:
        entries = [(key, arr, None) for key, arr in _walk_arrays(group)]

    if not entries:
        raise HTTPException(
            422,
            f"{path.name!r} contains no 3-D arrays to annotate. (An empty Zarr "
            "group has nothing to load.)",
        )

    # Finest first, by voxel count.
    entries.sort(key=lambda e: -(e[1].shape[0] * e[1].shape[1] * e[1].shape[2]))
    base_shape = entries[0][1].shape
    base_scale = entries[0][2]

    levels: list[dict[str, Any]] = []
    for key, arr, scale in entries:
        shape = [int(v) for v in arr.shape]
        levels.append(
            {
                "path": key,
                "shape": shape,
                "dtype": str(arr.dtype),
                "n_slices": shape[0],
                "height": shape[1],
                "width": shape[2],
                # How many finest-level voxels one voxel here spans, per axis.
                "downsample": [
                    round(base_shape[i] / shape[i], 4) if shape[i] else 1.0 for i in range(3)
                ],
            }
        )

    voxel_size = base_scale[0] if base_scale else None
    return {
        "name": path.name,
        "path": str(path),
        "levels": levels,
        "full_shape": [int(v) for v in base_shape],
        "dtype": str(entries[0][1].dtype),
        # Finest-level voxel size (z, y, x) and its unit, for the Measurement panel.
        "voxel_size": base_scale,
        "voxel_unit": _voxel_unit(path),
        "pixel_size": voxel_size,
    }


def _voxel_unit(path: Path) -> str | None:
    """Axis unit declared in the store's ``multiscales`` axes (e.g. micrometer)."""
    attrs_file = path / ".zattrs"
    if not attrs_file.exists():
        return None
    try:
        axes = json.loads(attrs_file.read_text())["multiscales"][0]["axes"]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    for axis in axes:
        unit = axis.get("unit")
        if unit:
            return str(unit)
    return None


def registered_key(path: Path) -> str:
    """The node key Tiled will use for *path*.

    Tiled strips the extension, so ``foo.zarr`` registers as ``foo``. Deriving
    this with Tiled's own helper (rather than assuming ``path.name``) keeps the
    collision check below looking at the key that will actually be created — a
    mismatch here means the check passes and Tiled then hits the collision
    itself, deep inside registration.
    """
    from tiled.client.register import Settings

    return Settings.init().key_from_filename(path.name)


def _existing_node_info(node: Any) -> dict[str, Any]:
    """Describe what already occupies a key, for the conflict prompt.

    ``external`` distinguishes a previous Zarr registration (safe to drop — it
    only removes catalog rows) from internally-managed data such as an uploaded
    image stack, where deleting the node also deletes the files.
    """
    children: list[str] = []
    try:
        children = list(node)
    except Exception:  # noqa: BLE001 — best-effort description
        pass
    meta = {}
    try:
        meta = dict(getattr(node, "metadata", {}) or {})
    except Exception:  # noqa: BLE001
        pass
    return {
        "child_count": len(children),
        "external": meta.get("source_format") == "zarr",
        "sample_name": meta.get("sample_name") or "",
        "n_images": meta.get("n_images"),
    }


def preflight_zarr(
    server_uri: str | None, raw_path: str, container_path: str = "browse"
) -> dict[str, Any]:
    """Report whether registering *raw_path* would collide, without changing anything.

    Returns the derived ``key``, whether it ``exists``, and — when it does — what
    is there, so the UI can warn before offering Replace.
    """
    path = _resolve_path(raw_path)
    key = registered_key(path)
    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    parts = [p for p in container_path.strip("/").split("/") if p]
    target = ingest_mod._walk(client, parts)
    existing = None
    if target is not None and key in ingest_mod._child_keys(target):
        existing = _existing_node_info(target[key])
    return {"key": key, "exists": existing is not None, "existing": existing}


def register_zarr(
    server_uri: str | None,
    raw_path: str,
    container_path: str,
    description: str = "",
    on_conflict: str = "fail",
) -> dict[str, Any]:
    """Register a Zarr store into a Tiled container, copying no data.

    Args:
        server_uri: Connected Tiled server URI.
        raw_path: Absolute path to the ``.zarr`` directory.
        container_path: Slash-separated target container (e.g. ``browse``).
        description: Optional keyword(s) stored on the node, exactly as ingest
            does, so the volume is filterable in Browse.
        on_conflict: ``"fail"``, ``"replace"`` or ``"skip"`` when the key exists.

    Returns:
        Dict with the registered ``key``, its Tiled ``path``, and the inspection
        result (so the caller can offer the level picker without a second call).
    """
    info = inspect_zarr(raw_path)
    path = Path(info["path"])
    description = (description or "").strip()
    keywords = ingest_mod.parse_keywords(description)
    if on_conflict not in ingest_mod.ON_CONFLICT_MODES:
        on_conflict = "fail"

    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    parts = [p for p in container_path.strip("/").split("/") if p]
    target = ingest_mod._ensure_container(client, parts)

    # Tiled derives the key from the filename (dropping ".zarr"), so resolve the
    # collision against THAT key. Getting this wrong lets registration proceed
    # until Tiled hits the collision internally, where it can only offer to
    # delete external assets and aborts on anything else.
    key = registered_key(path)
    if key in ingest_mod._child_keys(target):
        existing = _existing_node_info(target[key])
        if on_conflict == "skip":
            return {**info, "key": key, "tiled_path": "/".join([*parts, key]), "skipped": True}
        if on_conflict == "replace":
            if not existing["external"]:
                # The occupant is internally-managed data (e.g. an uploaded image
                # stack that happens to share this name). Deleting it would delete
                # the files themselves, which is not what "replace this Zarr"
                # should ever mean — make the user rename or remove it explicitly.
                raise HTTPException(
                    409,
                    f"{key!r} already exists in {container_path!r} and holds uploaded "
                    f"data ({existing['child_count']} items), not a registered Zarr. "
                    "Replacing it would delete that data — load into a different "
                    "container, or delete that dataset yourself first.",
                )
            # A previous Zarr registration: dropping it removes catalog rows only,
            # never the store on disk.
            target.delete_contents(key, recursive=True, external_only=False)
        else:
            raise HTTPException(
                409,
                f"{key!r} already exists in {container_path!r}. Choose Replace or "
                "a different destination.",
            )

    # Tiled's own single-item registration: it resolves .zarr -> application/x-zarr
    # and stores an external Asset pointing at the directory. Nothing is copied.
    from tiled.client.register import Settings, register_single_item

    try:
        asyncio.run(
            register_single_item(target, path, is_directory=True, settings=Settings.init())
        )
    except Exception as exc:  # noqa: BLE001 — classified for the UI below
        logger.warning("zarr registration failed for %s: %s", path, exc)
        raise HTTPException(502, ingest_mod._classify_error(exc)["message"]) from exc

    node = ingest_mod._walk(target, [key])
    if node is None:
        # register_single_item logs and swallows adapter errors, returning None —
        # so a missing node here is the signal that registration did not happen.
        raise HTTPException(
            502,
            f"Tiled did not register {key!r}. Check the backend log for the "
            "adapter error, and that the Tiled server can read this path.",
        )

    # Same metadata shape the dropzone writes, so Browse treats this like any
    # other sample, plus the pyramid description the level picker needs.
    meta: dict[str, Any] = {
        "sample_name": key,
        "n_images": info["levels"][0]["n_slices"],
        "source_format": "zarr",
        "zarr_path": str(path),
        "zarr_levels": info["levels"],
        "full_shape": info["full_shape"],
    }
    if info.get("voxel_size"):
        meta["voxel_size"] = info["voxel_size"]
    if info.get("voxel_unit"):
        meta["voxel_unit"] = info["voxel_unit"]
    if description:
        meta["description"] = description
    if keywords:
        meta["keywords"] = keywords
    try:
        node.update_metadata(metadata=meta)
    except Exception as exc:  # noqa: BLE001 — best-effort; the data is registered
        logger.warning("could not set metadata on %s: %s", key, exc)

    # Spread `info` FIRST: it carries the filesystem path under "path", which
    # must not shadow the Tiled path the caller needs to open the dataset.
    return {
        **info,
        "key": key,
        "tiled_path": "/".join([*parts, key]),
        "skipped": False,
    }
