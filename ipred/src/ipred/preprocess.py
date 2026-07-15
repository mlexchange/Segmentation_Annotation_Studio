"""Cache-aware featurize for a session (composition-first)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ipred import array_source, compositions, features
from ipred.catalog import Catalog
from ipred.compose_run import run_composition
from ipred.paths import project_blob_dir

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_preprocess(
    catalog: Catalog,
    *,
    session_id: str,
    feature_setup_id: str | None = None,
    composition_id: str | None = None,
    slice_index: int = 0,
    array_ref: str | None = None,
) -> dict[str, Any]:
    """Featurize with cache hit/miss; update session current_feature_id.

    Prefers ``composition_id``; ``feature_setup_id`` is mapped via legacy table.
    Optional ``array_ref`` selects a session-uploaded blob instead of path/Tiled.
    """
    session = catalog.get_session(session_id)
    if session is None:
        raise KeyError(f"unknown session {session_id}")
    project = catalog.get_project(session.project_id)
    if project is None:
        raise KeyError(f"unknown project {session.project_id}")

    compositions.ensure_default_compositions(catalog)
    raw_id = composition_id or feature_setup_id
    if not raw_id:
        raise ValueError("composition_id or feature_setup_id required")
    try:
        cid = compositions.resolve_preprocess_id(raw_id)
    except KeyError:
        # Ensure defaults then retry
        compositions.ensure_default_compositions(catalog)
        cid = compositions.resolve_preprocess_id(raw_id)

    doc = compositions.resolve_composition(cid)
    chash = doc.get("content_hash") or compositions.content_hash(doc)

    hit = catalog.find_feature_bank(
        project_id=project.project_id,
        setup_id=cid,
        content_hash=chash,
        slice_index=slice_index,
    )
    if hit is not None and array_ref is None:
        catalog.set_session_currents(session_id, feature_id=hit["feature_id"])
        return _bank_response(hit, cache_hit=True)

    if array_ref:
        arr = _load_array_ref(project.project_id, array_ref)
    else:
        arr = array_source.read_slice(
            kind=project.kind,
            source=project.source,
            slice_index=slice_index,
            server_uri=project.server_uri,
            root=project.root,
        )
    uint8_stack, float_stack, labels, sam_emb, sam_meta = run_composition(
        arr, doc
    )
    feature_id = uuid.uuid4().hex
    blob = project_blob_dir(project.project_id) / "features" / feature_id
    blob.mkdir(parents=True, exist_ok=True)
    np.save(blob / "float_stack.npy", float_stack.astype(np.float16))
    np.save(blob / "uint8_stack.npy", uint8_stack)
    (blob / "labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )
    if sam_emb is not None:
        np.save(blob / "sam_emb.npy", sam_emb.astype(np.float32))
        (blob / "sam_meta.json").write_text(
            json.dumps(sam_meta), encoding="utf-8"
        )
    channels_dir = blob / "channels"
    channels_dir.mkdir(exist_ok=True)
    for i in range(uint8_stack.shape[-1]):
        png = features.encode_channel_png(uint8_stack, i)
        (channels_dir / f"{i:04d}.png").write_bytes(png)

    h, w, c = float_stack.shape
    snapshot = dict(doc)
    record = {
        "feature_id": feature_id,
        "project_id": project.project_id,
        "setup_id": cid,
        "content_hash": chash,
        "slice_index": int(slice_index),
        "n_channels": int(c),
        "height": int(h),
        "width": int(w),
        "blob_dir": str(blob),
        "setup_snapshot": json.dumps(snapshot),
        "status": "ready",
        "created_at": _utc_now(),
    }
    catalog.insert_feature_bank(record)
    catalog.set_session_currents(session_id, feature_id=feature_id)
    return _bank_response(record, cache_hit=False, labels=labels)


def _load_array_ref(project_id: str, array_ref: str) -> np.ndarray:
    """Load a content-addressed slice uploaded via the data plane."""
    from ipred.array_blobs import load_array_blob

    return load_array_blob(project_id, array_ref)


def load_feature_bank_arrays(blob_dir: str | Path) -> dict[str, Any]:
    """Load persisted feature arrays from a bank directory."""
    blob = Path(blob_dir)
    float_stack = np.load(blob / "float_stack.npy").astype(np.float32)
    uint8_stack = np.load(blob / "uint8_stack.npy")
    labels = json.loads((blob / "labels.json").read_text(encoding="utf-8"))
    sam_emb = None
    sam_meta = None
    if (blob / "sam_emb.npy").is_file():
        sam_emb = np.load(blob / "sam_emb.npy")
        if (blob / "sam_meta.json").is_file():
            sam_meta = json.loads(
                (blob / "sam_meta.json").read_text(encoding="utf-8")
            )
    return {
        "float_stack": float_stack,
        "uint8_stack": uint8_stack,
        "labels": labels,
        "sam_emb": sam_emb,
        "sam_meta": sam_meta,
    }


def channel_png_path(blob_dir: str | Path, index: int) -> Path:
    """Path to a cached channel PNG."""
    return Path(blob_dir) / "channels" / f"{index:04d}.png"


def _bank_response(
    record: dict[str, Any],
    *,
    cache_hit: bool,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    blob = Path(record["blob_dir"])
    if labels is None and (blob / "labels.json").is_file():
        labels = json.loads((blob / "labels.json").read_text(encoding="utf-8"))
    return {
        "feature_id": record["feature_id"],
        "project_id": record["project_id"],
        "setup_id": record["setup_id"],
        "composition_id": record["setup_id"],
        "slice_index": record["slice_index"],
        "n_channels": record["n_channels"],
        "height": record["height"],
        "width": record["width"],
        "labels": labels or [],
        "cache_hit": cache_hit,
        "blob_dir": record["blob_dir"],
    }


# Keep soft import for tests that still poke at legacy helpers
def _compute_from_setup(
    arr: np.ndarray,
    setup: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
    dict[str, Any] | None,
]:
    """Legacy bridge: convert procedure setup → composition run via resolve."""
    del snapshot
    sid = setup.get("id") or ""
    compositions.ensure_default_compositions()
    try:
        cid = compositions.resolve_preprocess_id(sid)
        doc = compositions.resolve_composition(cid)
        return run_composition(arr, doc)
    except KeyError as exc:
        raise ValueError(
            "weights/procedure setups must map to a composition; "
            f"unknown {sid}"
        ) from exc
