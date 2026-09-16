"""Background ingest of uploaded image files into a Tiled container.

The Connect page lets the user drag-and-drop a folder of images; those bytes
are streamed to temp files by the API layer, then this module's worker copies
each file into the connected Tiled server via ``write_array`` (works for remote
servers too — Tiled uploads the array over HTTP).

Each file becomes its OWN node with metadata so the result is browsable in the
metadata-driven Browse UI (which needs fields with >=2 distinct values):

* ``image_number``    — zero-padded STRING index parsed from the filename, so it
  stays a string filter (see ``browse_helpers._typed_query_value``).
* ``size``            — ``"H x W"`` from the array shape.
* ``original_filename`` — the uploaded filename.
* ``sample_name``     — the filename stem (the node key); a friendlier label.
* ``description``     — optional user-supplied keyword(s) entered on Connect; the
  same value is written to every node in the batch so a single drop is filterable
  (these keys are made facet-eligible with one distinct value, see
  ``browse_helpers._SINGLE_VALUE_FACET_RAW_KEYS``).
* ``keywords``        — the ``description`` split on commas into a LIST of tags.
  Each tag becomes an individually-searchable Browse value (see
  ``browse_helpers``) and is pre-created as an annotation class for the dataset
  in the Annotate tab.

Job state is held in-memory (lost on restart) — acceptable for a localhost tool.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger("ingest")

IMAGE_EXTS: frozenset[str] = frozenset({".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"})
# Trailing digits of the filename stem (e.g. "..._petiole22_00042" -> "00042").
# Anchored at the end so a leading date like "20260221_..." is not mistaken for
# the frame index.
_FRAME_RE = re.compile(r"(\d+)$")
_MIN_PAD = 5
# Leading HTTP status in a Tiled ClientError message ("409: <path> <url>").
_STATUS_RE = re.compile(r"^\s*(\d{3}):")
_URL_RE = re.compile(r"https?://\S+")
# Trailing "_N" on a container name, so suggestions bump instead of stacking.
_SUFFIX_RE = re.compile(r"^(.*?)_(\d+)$")

# What to do when a target node key already exists (see ``run_ingest_job``).
ON_CONFLICT_MODES: frozenset[str] = frozenset({"fail", "replace", "skip"})

# job_id -> {state, total, done, failed, skipped, errors[], container_path, server_uri}
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def new_job(total: int, server_uri: str | None, container_path: str) -> str:
    """Register a new ingest job and return its id."""
    jid = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[jid] = {
            "state": "pending",
            "total": total,
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "container_path": container_path,
            "server_uri": server_uri,
        }
    return jid


def get_job(jid: str) -> dict | None:
    """Return a snapshot copy of the job, or ``None`` if unknown."""
    with _jobs_lock:
        job = _jobs.get(jid)
        return dict(job) if job else None


def _update(jid: str, **kw: Any) -> None:
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def _bump(
    jid: str,
    *,
    done: int = 0,
    failed: int = 0,
    skipped: int = 0,
    error: dict[str, str] | None = None,
) -> None:
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            return
        job["done"] += done
        job["failed"] += failed
        job["skipped"] += skipped
        if error:
            job["errors"].append(error)


def _read_array(path: Path) -> np.ndarray:
    """Read a supported image file into a NumPy array."""
    import tifffile
    from PIL import Image as PILImage

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    if suffix == ".npy":
        return np.load(str(path), allow_pickle=False)
    if suffix in (".png", ".jpg", ".jpeg"):
        return np.asarray(PILImage.open(str(path)))
    raise ValueError(f"unsupported extension {suffix!r}")


def validate_container_path(container_path: str) -> list[str]:
    """Return safe Tiled key segments beneath the configured ingest root.

    A write confinement check, not a formatting one: callers take a destination
    from the client, so without this a request could name any container in the
    catalog — including one holding unrelated data. Rejects traversal segments,
    control characters, and anything that is not a strict descendant of
    ``TILED_INGEST_ROOT`` (default ``browse``).
    """
    raw = (container_path or "").strip()
    if not raw or raw != raw.strip("/"):
        raise ValueError("container_path must be a relative canonical Tiled path")
    parts = raw.split("/")
    if any(
        not part
        or part in {".", ".."}
        or len(part) > 128
        or any(ord(char) < 32 for char in part)
        for part in parts
    ):
        raise ValueError("container_path contains an unsafe segment")

    root_parts = [
        part for part in os.getenv("TILED_INGEST_ROOT", "browse").strip("/").split("/") if part
    ]
    # `len(parts) <= len(root_parts)` rejects the root itself: a dataset must be
    # written *inside* it, never over it.
    if not root_parts or parts[: len(root_parts)] != root_parts or len(parts) <= len(root_parts):
        raise ValueError("container_path must be a child of the configured ingest root")
    return parts


def _ensure_container(client: Any, parts: list[str]) -> Any:
    """Navigate to ``client[parts...]``, creating containers as needed.

    Only a genuine ``KeyError`` means "missing" — catching every exception here
    would send a transient transport blip down the ``create_container`` path,
    which then collides with the container that does in fact exist.
    """
    node = client
    for key in parts:
        try:
            node = node[key]
        except KeyError:
            node = node.create_container(key=key, metadata={})
    return node


def _walk(client: Any, parts: list[str]) -> Any | None:
    """Return the node at ``parts``, or ``None`` if any segment is missing."""
    node = client
    for key in parts:
        try:
            node = node[key]
        except KeyError:
            return None
    return node


def _child_keys(node: Any) -> set[str]:
    """Return a node's child keys; empty for leaves (arrays have no children)."""
    if node is None:
        return set()
    try:
        return set(node.keys())
    except Exception:  # noqa: BLE001 — leaf/array node
        return set()


def _classify_error(exc: BaseException) -> dict[str, str]:
    """Map an ingest exception to a ``{kind, message}`` pair for the UI.

    ``kind`` is a stable token the frontend turns into human copy. The message
    is a short fallback with any server URL stripped — Tiled's ``ClientError``
    text is ``"<status>: <colliding path> <internal API url>"``, which must not
    reach the browser. The full raw text is logged by the caller instead.

    Args:
        exc: The exception raised while ingesting one file.

    Returns:
        ``{"kind": ..., "message": ...}`` with kind in ``conflict``,
        ``unreadable``, ``unreachable``, ``auth`` or ``unknown``.
    """
    raw = str(exc) or type(exc).__name__
    name = type(exc).__name__
    match = _STATUS_RE.match(raw)
    status = match.group(1) if match else None

    if status == "409" or "Collision" in raw or "Conflict" in name:
        return {"kind": "conflict", "message": "a sample with this name already exists"}
    if status in ("401", "403"):
        return {"kind": "auth", "message": "not authorized to write to this server"}
    if "Connect" in name or "Timeout" in name or "Connection" in raw:
        return {"kind": "unreachable", "message": "could not reach the Tiled server"}
    if isinstance(exc, (ValueError, OSError)):
        return {"kind": "unreadable", "message": "file could not be read as an image"}
    return {"kind": "unknown", "message": _URL_RE.sub("", raw).strip()[:200]}


def _suggest_container_path(client: Any, parts: list[str]) -> str:
    """Return ``parts`` with a ``_N`` suffix that is unused among its siblings.

    An existing numeric suffix is bumped rather than appended to, so repeated
    suggestions give ``myset_2``, ``myset_3`` — never ``myset_2_2``.
    """
    if not parts:
        return ""
    siblings = _child_keys(_walk(client, parts[:-1]))
    match = _SUFFIX_RE.match(parts[-1])
    stem, index = (match.group(1), int(match.group(2)) + 1) if match else (parts[-1], 2)
    while f"{stem}_{index}" in siblings:
        index += 1
    return "/".join([*parts[:-1], f"{stem}_{index}"])


def preflight(
    server_uri: str | None, container_path: str, filenames: list[str]
) -> dict[str, Any]:
    """Report which uploads would collide with nodes already in the container.

    Called before any bytes are uploaded so the user can choose how to resolve
    the collision (replace / skip / new dataset / just browse the existing one)
    instead of watching a large upload fail with a 409 per file.

    Args:
        server_uri: Connected Tiled server URI.
        container_path: Slash-separated target container (e.g. ``browse/testset``).
        filenames: Original filenames the user is about to upload.

    Returns:
        ``{container_exists, existing_count, conflicts, suggested_container_path}``
        where each conflict is ``{"filename", "key"}``.
    """
    api_key = api_key_for_uri(server_uri)
    client = get_tiled_client(server_uri, api_key)
    parts = [p for p in container_path.strip("/").split("/") if p]
    node = _walk(client, parts)
    existing = _child_keys(node)
    return {
        "container_exists": node is not None,
        "existing_count": len(existing),
        "conflicts": [
            {"filename": name, "key": Path(name).stem}
            for name in filenames
            if Path(name).stem in existing
        ],
        "suggested_container_path": _suggest_container_path(client, parts),
    }


def _size_str(arr: np.ndarray) -> str:
    """Return pixel dimensions as ``"H x W"`` (first two dims)."""
    return " x ".join(str(d) for d in arr.shape[:2])


def _pad_width(filenames: list[str]) -> int:
    """Return a consistent zero-pad width for image_number across the batch."""
    widths = [
        len(m.group(1))
        for m in (_FRAME_RE.search(Path(f).stem) for f in filenames)
        if m
    ]
    return max([*widths, _MIN_PAD]) if widths else _MIN_PAD


def _image_number(stem: str, width: int, fallback_index: int) -> str:
    m = _FRAME_RE.search(stem)
    if m:
        return m.group(1).zfill(width)
    return str(fallback_index).zfill(width)


def parse_keywords(description: str) -> list[str]:
    """Split a comma-separated ``description`` into a de-duplicated tag list.

    Each tag doubles as (a) an annotation class pre-created for the dataset in
    the Annotate tab and (b) an individually-searchable value in Browse. Order
    is preserved and case-insensitive duplicates are dropped.

    Args:
        description: Raw comma-separated string entered on the Connect page.

    Returns:
        Ordered list of non-empty, de-duplicated tag strings.
    """
    tags: list[str] = []
    seen: set[str] = set()
    for part in (description or "").split(","):
        tag = part.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


def run_ingest_job(
    jid: str,
    server_uri: str | None,
    container_path: str,
    temp_files: list[tuple[str, Path]],
    description: str = "",
    on_conflict: str = "fail",
) -> None:
    """Copy each temp file into the target Tiled container.

    Args:
        jid: Job id from :func:`new_job`.
        server_uri: Connected Tiled server URI.
        container_path: Slash-separated target container (e.g. ``browse/testset``).
        temp_files: list of ``(original_filename, temp_path)``.
        description: Optional user-supplied keyword(s) stored on every node so the
            batch is identifiable/filterable in Browse (empty string → omitted).
        on_conflict: What to do when the node key already exists — ``"fail"``
            (record a ``conflict`` error), ``"replace"`` (delete then rewrite) or
            ``"skip"`` (leave the existing node, count it as skipped). The user
            picks this in the conflict dialog after :func:`preflight`.
    """
    description = (description or "").strip()
    keywords = parse_keywords(description)
    if on_conflict not in ON_CONFLICT_MODES:
        on_conflict = "fail"
    _update(jid, state="running")
    try:
        api_key = api_key_for_uri(server_uri)
        client = get_tiled_client(server_uri, api_key)
        parts = [p for p in container_path.strip("/").split("/") if p]
        target = _ensure_container(client, parts)
        width = _pad_width([orig for orig, _ in temp_files])

        # Describe the dataset at the CONTAINER level too. Browse treats each
        # child of the browse root as a "sample" and reads its container
        # metadata (not the per-array metadata written below), so this is what
        # makes the upload identifiable/filterable there.
        container_meta: dict[str, Any] = {
            "sample_name": parts[-1] if parts else "",
            "n_images": len(temp_files),
            "source_format": IMAGE_STACK_SOURCE_FORMAT,
        }
        if description:
            container_meta["description"] = description
        if keywords:
            # A LIST so Browse can offer each tag as its own filter value and
            # Annotate can pre-create one class per tag.
            container_meta["keywords"] = keywords
        try:
            target.update_metadata(metadata=container_meta)
        except Exception as exc:  # noqa: BLE001 — best-effort; per-array meta still set
            logger.warning("could not set container metadata on %s: %s", container_path, exc)

        # Only needed to resolve collisions; a container we just created is empty.
        existing_keys = _child_keys(target) if on_conflict != "fail" else set()

        for idx, (orig_name, tmp) in enumerate(temp_files):
            try:
                stem = Path(orig_name).stem
                if stem and stem in existing_keys:
                    if on_conflict == "skip":
                        _bump(jid, skipped=1)
                        continue
                    # "replace": drop the existing child so write_array can't
                    # collide. Must be delete_contents(key) — Container.delete()
                    # deletes the container ITSELF. external_only=False because
                    # we wrote these arrays into Tiled's own storage.
                    target.delete_contents(stem, recursive=True, external_only=False)
                arr = _read_array(tmp)
                meta = {
                    "image_number": _image_number(stem, width, idx),
                    "size": _size_str(arr),
                    "original_filename": orig_name,
                    "sample_name": stem,
                }
                if description:
                    meta["description"] = description
                if keywords:
                    meta["keywords"] = keywords
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    dims = ["y", "x", "channel"]
                elif arr.ndim == 2:
                    dims = ["y", "x"]
                else:
                    dims = None
                target.write_array(arr, key=stem, metadata=meta, dims=dims)
                _bump(jid, done=1)
            except Exception as exc:  # noqa: BLE001 — isolate per-file failures
                # Log the raw text (URLs and all); send only classified copy out.
                logger.warning("ingest %s failed: %s", orig_name, exc)
                _bump(jid, failed=1, error={"filename": orig_name, **_classify_error(exc)})
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        # A batch where nothing landed is a failure, not a success — reporting
        # "done" there is what made the UI show a green "Ingested 0 of 3".
        job = get_job(jid) or {}
        wholly_failed = job.get("done", 0) == 0 and job.get("failed", 0) > 0
        _update(jid, state="error" if wholly_failed else "done")
    except Exception as exc:  # noqa: BLE001 — fatal (e.g. cannot reach server)
        logger.error("ingest job %s fatal: %s", jid, exc)
        _bump(jid, error={"filename": "", **_classify_error(exc)})
        _update(jid, state="error")
    finally:
        for _, tmp in temp_files:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def node_source_kind(node: Any) -> str | None:
    """The ``source_format`` tag on *node*'s metadata, or None if absent/unreadable.

    Used to tell whether an existing Tiled node at a candidate key is the SAME
    kind of registration a scan is about to (re-)create, vs. an unrelated
    dataset that happens to share the same stem name (e.g. a raw image folder
    and its own already-registered Zarr reconstruction) — so a scan can report
    that collision honestly as "shadowed" instead of a misleading "already
    registered". Shared by :mod:`zarr_source`'s own scan function too.
    """
    try:
        return dict(getattr(node, "metadata", {}) or {}).get("source_format")
    except Exception:  # noqa: BLE001 — best-effort
        return None


#: Tag written on every container this module ingests into — lets a later scan
#: (of any kind) recognize "this key already holds a real per-slice ingest"
#: and distinguish it from an external Zarr/tiff-stack-3d registration sharing
#: the same stem name.
IMAGE_STACK_SOURCE_FORMAT = "image-stack"


def _is_image_stack_dir(path: Path) -> bool:
    """True if *path* is a directory holding 2+ directly-supported image files.

    A single image isn't a "stack" worth its own ingest — mirrors
    :func:`tiff_stack_source.inspect_tiff_stack`'s own ``len(files) < 2`` rule,
    generalized to every :data:`IMAGE_EXTS` type, not just TIFF.
    """
    if not path.is_dir():
        return False
    count = 0
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            count += 1
            if count >= 2:
                return True
    return False


def scan_and_register_image_stacks(
    server_uri: str | None,
    scan_root: str,
    container_path: str = "browse",
    on_conflict: str = "skip",
    renames: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Walk *scan_root* for folders of image slices and ingest each not already
    present, as fast per-slice registration — no 3-D pyramid is built here.

    Deliberately the "quick" path: each slice is copied into Tiled as its own
    2-D node (exactly what dropping the folder onto the Connect page's
    dropzone already does), with no multiscale volume generated. Building a
    3-D pyramid is comparatively expensive (reads every slice at least once)
    and is left to the existing, on-demand "Build 3D volume" button on the 3D
    page (:mod:`volume_build`) — a discovery scan should not silently take
    minutes per folder.

    Only top-level subdirectories of *scan_root* are candidates — one level,
    matching :func:`zarr_source.scan_and_register_zarrs`'s own non-recursive
    scope. A `.zarr` directory is never a candidate here even if it happens to
    also contain 2+ loose image files at its root (it shouldn't, but the check
    is cheap insurance against double-registering the same data two ways).

    A raw image folder and an unrelated dataset (e.g. a Zarr reconstruction of
    the same acquisition) commonly share the same stem name — in which case
    they'd derive the identical Tiled key. Rather than silently treating that
    as "already registered" (misleading — the raw slices were never actually
    ingested) or blindly replacing someone else's data, a same-key collision
    with a DIFFERENT kind of registration (per the ``source_format`` tag; see
    :func:`node_source_kind`) is reported as **shadowed**, distinctly from a
    same-kind ``skipped`` match from a previous run of this same scan.

    Args:
        server_uri: Connected Tiled server URI.
        scan_root: Absolute directory to scan.
        container_path: Parent container each discovered folder registers
            under, keyed by its own (sanitized) folder name — matching how a
            dropped folder is named today.
        on_conflict: ``"skip"`` (default, safe to re-run) or ``"replace"`` —
            applies only to a same-kind (not shadowed) collision.
        renames: Optional ``{folder_name: alternate_key}`` override, so a
            shadowed candidate can be retried under a different key without
            re-scanning everything else.

    Returns:
        Dict with ``scanned``, ``registered`` (``name``/``key``/``tiled_path``),
        ``skipped`` (names), ``shadowed`` (``name``/``key``/``existing_kind``/
        ``suggested_key`` — a different-kind collision, nothing registered),
        and ``errors`` (``name``/``error`` pairs).

    Raises:
        HTTPException: 400/404 if *scan_root* itself is unusable.
    """
    from tiled.client.register import Settings

    import zarr_source

    if on_conflict not in ("skip", "replace"):
        on_conflict = "skip"
    renames = renames or {}

    root = Path(scan_root).expanduser()
    if not root.is_absolute():
        raise HTTPException(400, f"Path must be absolute: {scan_root!r}")
    if not root.is_dir():
        raise HTTPException(404, f"No such directory: {root}")

    candidates = sorted(
        (p for p in root.iterdir() if not zarr_source._is_zarr_dir(p) and _is_image_stack_dir(p)),
        key=lambda p: p.name,
    )

    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    parts = [p for p in container_path.strip("/").split("/") if p]
    target = _ensure_container(client, parts)

    registered: list[dict[str, Any]] = []
    skipped: list[str] = []
    shadowed: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for candidate in candidates:
        default_key = Settings.init().key_from_filename(candidate.name)
        key = renames.get(candidate.name, default_key)
        existing_keys = _child_keys(target)
        if key in existing_keys:
            existing_kind = node_source_kind(target[key])
            if existing_kind != IMAGE_STACK_SOURCE_FORMAT:
                shadowed.append(
                    {
                        "name": candidate.name,
                        "key": key,
                        "existing_kind": existing_kind or "unknown",
                        "suggested_key": f"{key}_images",
                    }
                )
                continue
            if on_conflict == "skip":
                skipped.append(key)
                continue
            try:
                target.delete_contents(key, recursive=True, external_only=False)
            except Exception as exc:  # noqa: BLE001 — one bad entry must not sink the scan
                errors.append({"name": candidate.name, "error": str(exc)})
                continue

        # Copy each slice to a REAL temp file before ingesting — run_ingest_job
        # unlinks its inputs when done (correct for its normal caller, the
        # upload route's own tempfile.mkdtemp()); pointing it at scan_root's
        # actual files would delete the user's real source data.
        tmp_dir = Path(tempfile.mkdtemp(prefix="scan_ingest_"))
        try:
            temp_files: list[tuple[str, Path]] = []
            for index, src in enumerate(
                sorted(p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            ):
                dest = tmp_dir / f"{index:06d}{src.suffix.lower()}"
                shutil.copy2(src, dest)
                temp_files.append((src.name, dest))

            jid = new_job(len(temp_files), server_uri, f"{container_path}/{key}".strip("/"))
            run_ingest_job(jid, server_uri, f"{container_path}/{key}".strip("/"), temp_files, on_conflict=on_conflict)
            job = get_job(jid) or {}
            if job.get("state") == "error":
                errs = job.get("errors") or [{"error": "ingest failed"}]
                errors.append({"name": candidate.name, "error": errs[0].get("error", "ingest failed")})
            else:
                registered.append(
                    {"name": candidate.name, "key": key, "tiled_path": f"{container_path}/{key}".strip("/")}
                )
        except Exception as exc:  # noqa: BLE001 — one bad folder must not sink the scan
            errors.append({"name": candidate.name, "error": str(exc)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "scanned": len(candidates),
        "registered": registered,
        "skipped": skipped,
        "shadowed": shadowed,
        "errors": errors,
    }
