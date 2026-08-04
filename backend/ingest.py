"""Background ingest of uploaded image files into a Tiled container.

The Connect page lets the user drag-and-drop a folder of images; those bytes
are streamed to temp files by the API layer, then this module's worker copies
each file into the connected Tiled server via ``write_array`` (works for remote
servers too — Tiled uploads the array over HTTP).

Where a file lands depends on ``grouping`` (see :func:`sample_name_for`): either
directly in the target container (one sample, every file its slide — a z-stack
or time series), in its own sub-container (one sample per file), or grouped by a
``__`` filename prefix (one sub-container per specimen). Browse treats each
sub-container as a "sample" and its array children as that sample's slices.

Every array node carries metadata so the result is browsable in the
metadata-driven Browse UI (which needs fields with >=2 distinct values):

* ``image_number``    — zero-padded STRING index parsed from the filename, so it
  stays a string filter (see ``browse_helpers._typed_query_value``). Only this
  value is zero-padded — the node's own key is the raw filename stem, so lexical
  key order does not always match numeric slice order for unpadded filenames.
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

Sample-level containers additionally carry ``sample_name``/``n_images`` (and,
when the target holds several samples, ``n_samples``) — see :func:`run_ingest_job`.

Job state is held in-memory (lost on restart) — acceptable for a localhost tool.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tiled_clients import get_tiled_client

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
MAX_DECODED_BYTES = int(os.getenv("MAX_INGEST_DECODED_BYTES", str(1024 * 1024 * 1024)))
MAX_DECODED_PIXELS = int(os.getenv("MAX_INGEST_DECODED_PIXELS", "200000000"))
MAX_ARRAY_DIMENSION = int(os.getenv("MAX_INGEST_ARRAY_DIMENSION", "100000"))
# Multipart upload quotas (file count, per-file bytes, aggregate bytes) are
# enforced at the HTTP boundary — see MAX_INGEST_FILES / MAX_INGEST_FILE_BYTES /
# MAX_INGEST_TOTAL_BYTES in annotation_server.py's ingest_upload route, which
# rejects an oversized request before any bytes reach this module.

# Terminal job records older than this are pruned on the next new_job() call,
# so a long-running server doesn't accumulate an unbounded in-memory registry.
_JOB_TTL_SECONDS = float(os.getenv("INGEST_JOB_TTL_SECONDS", str(6 * 3600)))
_MAX_JOBS = int(os.getenv("INGEST_MAX_JOBS", "500"))
_TERMINAL_STATES = frozenset({"done", "error"})

# job_id -> {state, total, done, failed, skipped, errors[], container_path, server_uri}
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _prune_jobs_locked() -> None:
    """Evict terminal jobs by age, then by count, while holding ``_jobs_lock``.

    Running/pending jobs are never evicted — only ``done``/``error`` records,
    which the UI has had a chance to poll and display.
    """
    now = time.monotonic()
    for jid, job in list(_jobs.items()):
        if job.get("state") in _TERMINAL_STATES and now - job.get("_created", now) > _JOB_TTL_SECONDS:
            del _jobs[jid]
    if len(_jobs) <= _MAX_JOBS:
        return
    terminal_by_age = sorted(
        (jid for jid, job in _jobs.items() if job.get("state") in _TERMINAL_STATES),
        key=lambda jid: _jobs[jid].get("_created", 0),
    )
    for jid in terminal_by_age[: len(_jobs) - _MAX_JOBS]:
        del _jobs[jid]


def new_job(total: int, server_uri: str | None, container_path: str) -> str:
    """Register a new ingest job and return its id."""
    jid = uuid.uuid4().hex
    with _jobs_lock:
        _prune_jobs_locked()
        _jobs[jid] = {
            "state": "pending",
            "total": total,
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "container_path": container_path,
            "server_uri": server_uri,
            "_created": time.monotonic(),
        }
    return jid


def get_job(jid: str) -> dict | None:
    """Return a snapshot copy of the job, or ``None`` if unknown."""
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            return None
        snap = dict(job)
        snap.pop("_created", None)
        return snap


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
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("TIFF contains no image series")
            series = tif.series[0]
            _validate_array_layout(series.shape, series.dtype)
            array = series.asarray()
        return np.asarray(array)
    if suffix == ".npy":
        array = np.load(str(path), allow_pickle=False, mmap_mode="r")
        _validate_array_layout(array.shape, array.dtype)
        return array
    if suffix in (".png", ".jpg", ".jpeg"):
        with PILImage.open(path) as image:
            bands = max(1, len(image.getbands()))
            shape = (image.height, image.width, bands) if bands > 1 else (image.height, image.width)
            _validate_array_layout(shape, np.dtype(np.uint8))
            array = np.asarray(image).copy()
        _validate_array_layout(array.shape, array.dtype)
        return array
    raise ValueError(f"unsupported extension {suffix!r}")


def _validate_array_layout(shape: tuple[int, ...], dtype: Any) -> None:
    """Reject decoded arrays that exceed configured allocation/geometry limits."""
    if not 2 <= len(shape) <= 4:
        raise ValueError("images must have between 2 and 4 dimensions")
    dimensions = tuple(int(value) for value in shape)
    if any(value <= 0 or value > MAX_ARRAY_DIMENSION for value in dimensions):
        raise ValueError("image dimension exceeds configured limit")
    element_count = math.prod(dimensions)
    if element_count > MAX_DECODED_PIXELS:
        raise ValueError("image exceeds decoded pixel limit")
    decoded_bytes = element_count * np.dtype(dtype).itemsize
    if decoded_bytes > MAX_DECODED_BYTES:
        raise ValueError("image exceeds decoded byte limit")


def validate_container_path(container_path: str) -> list[str]:
    """Return safe Tiled key segments beneath the configured ingest root."""
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

    root_parts = [part for part in os.getenv("TILED_INGEST_ROOT", "browse").strip("/").split("/") if part]
    if not root_parts or parts[: len(root_parts)] != root_parts or len(parts) <= len(root_parts):
        raise ValueError("container_path must be a child of the configured ingest root")
    return parts


def _ensure_container(client: Any, parts: list[str]) -> Any:
    """Navigate to ``client[parts...]``, creating containers as needed.

    Only a genuine ``KeyError`` means "missing" — catching every exception here
    would send a transient transport blip down the ``create_container`` path,
    which then collides with the container that does in fact exist.

    Raises:
        ValueError: an existing key along the path is a leaf (array), not a
            container — e.g. a sample name collides with an array written by an
            earlier "single"/"per_image" ingest. Writing into it as if it were a
            container would silently attach an unrelated array's key/metadata.
    """
    node = client
    for key in parts:
        try:
            child = node[key]
        except KeyError:
            child = node.create_container(key=key, metadata={})
        else:
            try:
                child.keys()  # only containers support this; see _child_keys
            except Exception as exc:  # noqa: BLE001 — anything without .keys() is a leaf
                raise ValueError(f"{key!r} already exists and is not a container") from exc
        node = child
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


# How a batch of uploaded images maps onto Tiled containers. Browse treats each
# container under the ingest root as one "sample" and its array children as that
# sample's slices, so this choice decides what counts as a sample.
GROUPING_MODES = ("prefix", "per_image", "single")
DEFAULT_GROUPING = "prefix"

# Splits "almond_control__slice_01" into sample "almond_control" and the rest.
# A double underscore is the convention microscope exports here use; a single one
# is far too common inside ordinary names to split on.
_SAMPLE_SEPARATOR = "__"


def validate_grouping(grouping: str) -> str:
    """Normalise and check a grouping mode, defaulting when unset."""
    mode = (grouping or DEFAULT_GROUPING).strip()
    if mode not in GROUPING_MODES:
        raise ValueError(f"grouping must be one of {', '.join(GROUPING_MODES)}")
    return mode


def sample_name_for(stem: str, grouping: str) -> str | None:
    """Sub-container (sample) a file belongs to, or None to sit directly in the
    target container.

    * ``"single"``    — everything in the target container: one sample, N slices.
    * ``"per_image"`` — one sub-container per file: N samples, 1 slice each.
    * ``"prefix"``    — group by the part before ``__``. Files with no separator
      stay in the target container together, so a conventional z-stack
      (``img_0001``…``img_0030``) remains one sample rather than fragmenting into
      thirty.
    """
    if grouping == "single":
        return None
    if grouping == "per_image":
        return stem
    head, sep, _ = stem.partition(_SAMPLE_SEPARATOR)
    return head if sep and head else None


def plan_grouping(filenames: list[str], grouping: str) -> dict[str, list[str]]:
    """Preview of ``sample -> filenames`` for *grouping*.

    Drives the UI's pre-upload summary, and is :func:`run_ingest_job`'s single
    source of truth for where each file lands (it builds a filename->sample
    lookup from this instead of re-deriving it, so the two can never disagree).

    The ``""`` key means "straight into the target container" — single-sample
    mode, or a "prefix" batch that is entirely unprefixed (the conventional
    z-stack: ``img_0001``…``img_0030``, one sample).

    A **mixed** "prefix" batch — some files declare a sample via ``__``, others
    don't — is a special case. The undeclared files are NOT folded into one
    catch-all ``""`` sample alongside the named ones: that would write arrays
    and sample containers as siblings under the same parent, a layout Browse
    cannot describe (a container is either "one sample, N slices" or "N
    samples, each 1+ slices" — see ``browse_helpers._describe_sample``). Instead
    each undeclared file becomes its own single-image sample, exactly as
    ``"per_image"`` would for that file alone.
    """
    declared = {name: sample_name_for(Path(name).stem, grouping) for name in filenames}
    is_mixed = grouping == "prefix" and _has_both(declared.values())

    out: dict[str, list[str]] = {}
    for name in filenames:
        sample = declared[name]
        if sample is None:
            sample = Path(name).stem if is_mixed else ""
        out.setdefault(sample, []).append(name)
    return out


def _has_both(declared: Iterable[str | None]) -> bool:
    """True if *declared* (an iterable of ``sample | None``) has at least one
    real sample name AND at least one ``None`` — the "mixed batch" condition."""
    has_named = has_bare = False
    for value in declared:
        if value is None:
            has_bare = True
        else:
            has_named = True
        if has_named and has_bare:
            return True
    return False


def preflight(
    server_uri: str | None,
    container_path: str,
    filenames: list[str],
    grouping: str = DEFAULT_GROUPING,
) -> dict[str, Any]:
    """Report which uploads would collide with nodes already in the container.

    Called before any bytes are uploaded so the user can choose how to resolve
    the collision (replace / skip / new dataset / just browse the existing one)
    instead of watching a large upload fail with a 409 per file.

    Args:
        server_uri: Connected Tiled server URI.
        container_path: Slash-separated target container (e.g. ``browse/testset``).
        filenames: Original filenames the user is about to upload.
        grouping: Grouping mode. Determines WHERE each file will actually land —
            the target container itself, or its own named sub-container — so
            conflicts are checked against the right destination (see below), and
            echoed back as ``sample_preview`` so the UI can show the split before
            uploading.

    Returns:
        ``{container_exists, existing_count, conflicts, suggested_container_path,
        sample_preview}`` where each conflict is ``{"filename", "key"}`` and
        ``sample_preview`` is ``[{"sample", "n_images"}, ...]``.
    """
    parts = validate_container_path(container_path)
    grouping = validate_grouping(grouping)
    client = get_tiled_client(server_uri)
    node = _walk(client, parts)
    existing = _child_keys(node)
    plan = plan_grouping(filenames, grouping)

    # A grouped file lands in ITS SAMPLE's children, not the target's — checking
    # only `existing` here is what silently dropped every grouped re-upload's
    # conflict dialog (the target's children are sample names, never file stems).
    conflicts: list[dict[str, str]] = []
    for sample, names in plan.items():
        dest_existing = existing if not sample else _child_keys(_walk(client, [*parts, sample]))
        for name in names:
            stem = Path(name).stem
            if stem in dest_existing:
                conflicts.append({"filename": name, "key": stem})
    conflicts.sort(key=lambda c: filenames.index(c["filename"]))

    return {
        "container_exists": node is not None,
        "existing_count": len(existing),
        "conflicts": conflicts,
        "suggested_container_path": _suggest_container_path(client, parts),
        "sample_preview": [
            {"sample": sample or (parts[-1] if parts else ""), "n_images": len(names)}
            for sample, names in sorted(plan.items())
        ],
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
    grouping: str = DEFAULT_GROUPING,
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
        grouping: How images map onto samples — see :func:`sample_name_for` and
            :func:`plan_grouping` (which also covers the mixed-batch rule).

    Also sets ``first_path`` on the job record once the first file has
    successfully landed — the container-relative path of the array itself
    (through its sample sub-container, if any) — so a caller can jump straight
    to it (e.g. "open first image") without having to guess the layout the
    chosen ``grouping`` produced.
    """
    description = (description or "").strip()
    keywords = parse_keywords(description)
    if on_conflict not in ON_CONFLICT_MODES:
        on_conflict = "fail"
    grouping = validate_grouping(grouping)
    _update(jid, state="running")
    try:
        parts = validate_container_path(container_path)
        client = get_tiled_client(server_uri)
        target = _ensure_container(client, parts)
        filenames = [orig for orig, _ in temp_files]
        width = _pad_width(filenames)

        # plan is the single source of truth for where each file lands — reverse
        # it once instead of re-deriving each file's sample via sample_name_for,
        # so this loop can never disagree with plan_grouping's mixed-batch rule.
        plan = plan_grouping(filenames, grouping)
        file_to_sample = {name: sample for sample, names in plan.items() for name in names}
        named_samples = [s for s in plan if s]

        def _describe(
            node: Any, sample_name: str, n_images: int, where: str, *, n_samples: int | None = None
        ) -> None:
            """Write the sample-level metadata Browse reads for each sample row."""
            meta: dict[str, Any] = {"sample_name": sample_name, "n_images": n_images}
            if n_samples is not None:
                meta["n_samples"] = n_samples
            if description:
                meta["description"] = description
            if keywords:
                # A LIST so Browse can offer each tag as its own filter value and
                # Annotate can pre-create one class per tag.
                meta["keywords"] = keywords
            try:
                node.update_metadata(metadata=meta)
            except Exception as exc:  # noqa: BLE001 — best-effort; per-array meta still set
                logger.warning("could not set container metadata on %s: %s", where, exc)

        # Describe the target container. In single-sample mode (or an
        # all-unprefixed "prefix" batch) this container *is* the sample;
        # otherwise it's a parent holding one container per sample, each
        # described as it's created below. n_images is always the total file
        # count — it used to hold the SAMPLE count once grouped, which is what
        # n_samples is for.
        _describe(
            target,
            parts[-1] if parts else "",
            len(filenames),
            container_path,
            n_samples=len(named_samples) if named_samples else None,
        )

        # Per-sample sub-containers, created up front so each is described once
        # rather than re-resolved per file.
        sample_nodes: dict[str, Any] = {}
        for sample, names in plan.items():
            if not sample:
                continue
            node = _ensure_container(client, [*parts, sample])
            _describe(node, sample, len(names), f"{container_path}/{sample}")
            sample_nodes[sample] = node

        # Only needed to resolve collisions; a container we just created is empty.
        # Grown as files land (below) so a duplicate stem WITHIN this batch is
        # caught too, not just a collision with a pre-existing node.
        existing: dict[str, set[str]] = {}
        if on_conflict != "fail":
            existing[""] = _child_keys(target)
            for sample, node in sample_nodes.items():
                existing[sample] = _child_keys(node)

        sample_written: dict[str, int] = dict.fromkeys(sample_nodes, 0)
        first_path: str | None = None

        for idx, (orig_name, tmp) in enumerate(temp_files):
            try:
                stem = Path(orig_name).stem
                sample = file_to_sample.get(orig_name, "")
                # Write into the file's own sample container (or the target itself).
                dest = sample_nodes[sample] if sample else target
                existing_keys = existing.get(sample, set())
                if stem and stem in existing_keys:
                    if on_conflict == "skip":
                        _bump(jid, skipped=1)
                        continue
                # Fully decode the replacement before any destructive action. A
                # corrupt/truncated upload must never erase the valid node it was
                # intended to replace.
                arr = _read_array(tmp)
                if stem and stem in existing_keys:
                    # "replace": drop the existing child so write_array can't
                    # collide. Must be delete_contents(key) — Container.delete()
                    # deletes the container ITSELF. external_only=False because
                    # we wrote these arrays into Tiled's own storage.
                    dest.delete_contents(stem, recursive=True, external_only=False)
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
                dest.write_array(arr, key=stem, metadata=meta, dims=dims)
                if stem:
                    existing.setdefault(sample, set()).add(stem)
                if sample:
                    sample_written[sample] = sample_written.get(sample, 0) + 1
                if first_path is None:
                    first_path = f"{sample}/{stem}" if sample else stem
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

        # A sample that never received a successful write is dead weight — it
        # claims images it doesn't have, and opening it 422s ("no array slices").
        # Drop it, and correct the parent's/siblings' counts to match reality.
        empty_samples = [s for s, n in sample_written.items() if n == 0]
        for sample in empty_samples:
            try:
                target.delete_contents(sample, recursive=True, external_only=False)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.warning("could not remove empty sample %s/%s: %s", container_path, sample, exc)
        if empty_samples:
            real_samples = [s for s in named_samples if s not in empty_samples]
            job = get_job(jid) or {}
            _describe(
                target,
                parts[-1] if parts else "",
                job.get("done", 0),
                container_path,
                n_samples=len(real_samples) if real_samples else None,
            )
            for sample in real_samples:
                _describe(sample_nodes[sample], sample, sample_written[sample], f"{container_path}/{sample}")

        if first_path is not None:
            _update(jid, first_path=f"{container_path}/{first_path}")

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
        temp_parents = {tmp.parent for _, tmp in temp_files}
        for _, tmp in temp_files:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        for parent in temp_parents:
            if parent.name.startswith("ingest_"):
                try:
                    parent.rmdir()
                except OSError:
                    pass
