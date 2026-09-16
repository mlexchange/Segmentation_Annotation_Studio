"""Content-addressed array blobs for remote compute (data ≠ compute host)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ipred.paths import project_blob_dir


def arrays_dir(project_id: str) -> Path:
    """Blob directory for uploaded arrays."""
    path = project_blob_dir(project_id) / "arrays"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_array_blob(
    project_id: str,
    arr: np.ndarray,
    *,
    array_ref: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist ``arr`` and return ``{array_ref, shape, dtype, ...}``."""
    data = np.asarray(arr)
    raw = data.astype(np.float32, copy=False).tobytes()
    digest = hashlib.sha1(raw).hexdigest()[:16]
    ref = array_ref or f"{digest}_{uuid.uuid4().hex[:8]}"
    dest = arrays_dir(project_id) / ref
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "array.npy", data)
    record = {
        "array_ref": ref,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sha1": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(meta or {}),
    }
    (dest / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def load_array_blob(project_id: str, array_ref: str) -> np.ndarray:
    """Load a previously uploaded array."""
    path = arrays_dir(project_id) / array_ref / "array.npy"
    if not path.is_file():
        raise FileNotFoundError(f"array_ref not found: {array_ref}")
    return np.load(path).astype(np.float32)
