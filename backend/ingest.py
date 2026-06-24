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

Job state is held in-memory (lost on restart) — acceptable for a localhost tool.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger("ingest")

IMAGE_EXTS: frozenset[str] = frozenset({".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"})
# Trailing digits of the filename stem (e.g. "..._petiole22_00042" -> "00042").
# Anchored at the end so a leading date like "20260221_..." is not mistaken for
# the frame index.
_FRAME_RE = re.compile(r"(\d+)$")
_MIN_PAD = 5

# job_id -> {state, total, done, failed, errors[], container_path, server_uri}
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


def _bump(jid: str, *, done: int = 0, failed: int = 0, error: str | None = None) -> None:
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            return
        job["done"] += done
        job["failed"] += failed
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


def _ensure_container(client: Any, parts: list[str]) -> Any:
    """Navigate to ``client[parts...]``, creating containers as needed."""
    node = client
    for key in parts:
        try:
            node = node[key]
        except Exception:  # noqa: BLE001 — KeyError or transport error → create
            node = node.create_container(key=key, metadata={})
    return node


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


def run_ingest_job(
    jid: str,
    server_uri: str | None,
    container_path: str,
    temp_files: list[tuple[str, Path]],
) -> None:
    """Copy each temp file into the target Tiled container.

    Args:
        jid: Job id from :func:`new_job`.
        server_uri: Connected Tiled server URI.
        container_path: Slash-separated target container (e.g. ``browse/testset``).
        temp_files: list of ``(original_filename, temp_path)``.
    """
    _update(jid, state="running")
    try:
        api_key = api_key_for_uri(server_uri)
        client = get_tiled_client(server_uri, api_key)
        parts = [p for p in container_path.strip("/").split("/") if p]
        target = _ensure_container(client, parts)
        width = _pad_width([orig for orig, _ in temp_files])

        for idx, (orig_name, tmp) in enumerate(temp_files):
            try:
                arr = _read_array(tmp)
                stem = Path(orig_name).stem
                meta = {
                    "image_number": _image_number(stem, width, idx),
                    "size": _size_str(arr),
                    "original_filename": orig_name,
                }
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    dims = ["y", "x", "channel"]
                elif arr.ndim == 2:
                    dims = ["y", "x"]
                else:
                    dims = None
                target.write_array(arr, key=stem, metadata=meta, dims=dims)
                _bump(jid, done=1)
            except Exception as exc:  # noqa: BLE001 — isolate per-file failures
                logger.warning("ingest %s failed: %s", orig_name, exc)
                _bump(jid, failed=1, error=f"{orig_name}: {exc}")
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        _update(jid, state="done")
    except Exception as exc:  # noqa: BLE001 — fatal (e.g. cannot reach server)
        logger.error("ingest job %s fatal: %s", jid, exc)
        _bump(jid, error=str(exc))
        _update(jid, state="error")
    finally:
        for _, tmp in temp_files:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
