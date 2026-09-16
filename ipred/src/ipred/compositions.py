"""Composition documents — ordered module graphs for feature banks."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipred.catalog import Catalog
from ipred.modules import get_module, list_module_catalog
from ipred.paths import feature_models_root

logger = logging.getLogger(__name__)

KIND_COMPOSITION = "composition"


def compositions_root() -> Path:
    """Directory for composition meta.json files."""
    root = feature_models_root() / "_compositions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def content_hash(doc: dict[str, Any]) -> str:
    """Stable hash of composition structure (excludes name/timestamps)."""
    payload = {
        "kind": KIND_COMPOSITION,
        "nodes": doc.get("nodes"),
        "outputs": doc.get("outputs"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _dir(comp_id: str) -> Path:
    return compositions_root() / comp_id


def load_composition(comp_id: str) -> dict[str, Any] | None:
    """Load composition meta or None."""
    path = _dir(comp_id) / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_compositions() -> list[dict[str, Any]]:
    """List all compositions on disk."""
    root = compositions_root()
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def save_composition(
    *,
    name: str,
    nodes: list[dict[str, Any]],
    outputs: list[str],
    composition_id: str | None = None,
    builtin: bool = False,
    catalog: Catalog | None = None,
) -> dict[str, Any]:
    """Create or overwrite a composition document."""
    validate_composition({"nodes": nodes, "outputs": outputs})
    cid = composition_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    existing = load_composition(cid)
    created = existing.get("created_at") if existing else now
    meta: dict[str, Any] = {
        "id": cid,
        "name": (name or cid).strip() or cid,
        "kind": KIND_COMPOSITION,
        "builtin": bool(builtin),
        "nodes": list(nodes),
        "outputs": list(outputs),
        "created_at": created,
        "updated_at": now,
    }
    meta["content_hash"] = content_hash(meta)
    dest = _dir(cid)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if catalog is not None:
        catalog.upsert_feature_setup(
            setup_id=cid,
            name=meta["name"],
            kind=KIND_COMPOSITION,
            content_hash=meta["content_hash"],
            meta=meta,
        )
    return meta


def resolve_composition(comp_id: str) -> dict[str, Any]:
    """Load composition or raise KeyError."""
    meta = load_composition(comp_id)
    if meta is None:
        raise KeyError(f"unknown composition {comp_id}")
    return meta


def validate_composition(doc: dict[str, Any]) -> None:
    """Raise ValueError if nodes/outputs are invalid."""
    nodes = doc.get("nodes") or []
    outputs = doc.get("outputs") or []
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("composition requires at least one node")
    ids = [n.get("id") for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate node ids")
    for n in nodes:
        mid = n.get("module")
        if not mid:
            raise ValueError(f"node {n.get('id')} missing module")
        try:
            get_module(str(mid))
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        src = n.get("input_from")
        if src is not None and src not in ids:
            raise ValueError(f"node {n.get('id')} input_from unknown: {src}")
    for oid in outputs:
        if oid not in ids:
            raise ValueError(f"output node unknown: {oid}")


def preview_concat_labels(doc: dict[str, Any]) -> list[str]:
    """Labels the bank would concatenate for ``outputs`` order."""
    validate_composition(doc)
    by_id = {n["id"]: n for n in doc["nodes"]}
    labels: list[str] = []
    for oid in doc.get("outputs") or []:
        node = by_id[oid]
        mod = get_module(str(node["module"]))
        params = dict(node.get("params") or {})
        labels.extend(mod.preview_labels(params))
    return labels


def setup_id_to_composition_id(setup_id: str) -> str | None:
    """Map legacy procedure setup ids → builtin composition ids."""
    return _LEGACY_SETUP_MAP.get(setup_id)


_LEGACY_SETUP_MAP: dict[str, str] = {
    "default-skimage": "comp-skimage",
    "default-skimage-slimsam": "comp-skimage-slimsam",
    "default-slimsam-clahe": "comp-slimsam-clahe",
    "default-skimage-mark25": "comp-skimage-mark25",
    "default-mark25-clahe": "comp-mark25-clahe",
    "default-skimage-mark11": "comp-skimage-mark11",
    "default-mark11-clahe": "comp-mark11-clahe",
    # weights-only → CLAHE companion compositions
    "default-mark25": "comp-mark25-clahe",
    "default-mark11": "comp-mark11-clahe",
    "default-slimsam": "comp-skimage-slimsam",
}


def _skimage_params() -> dict[str, Any]:
    return {
        "sigma_min": 1.0,
        "sigma_max": 8.0,
        "intensity": True,
        "edges": True,
        "texture": True,
        "clahe": True,
    }


def ensure_default_compositions(catalog: Catalog | None = None) -> list[dict[str, Any]]:
    """Create builtin compositions matching legacy procedure setups."""
    specs: list[tuple[str, str, list[dict[str, Any]], list[str]]] = [
        (
            "comp-skimage",
            "Skimage multiscale",
            [{"id": "n1", "module": "skimage_multiscale", "params": _skimage_params()}],
            ["n1"],
        ),
        (
            "comp-skimage-slimsam",
            "Skimage + SlimSAM",
            [
                {"id": "n1", "module": "skimage_multiscale", "params": _skimage_params()},
                {"id": "n2", "module": "slimsam", "params": {}},
                {
                    "id": "n3",
                    "module": "pca",
                    "params": {"dims": 32},
                    "input_from": "n2",
                },
            ],
            ["n1", "n3"],
        ),
        (
            "comp-slimsam-clahe",
            "SlimSAM + CLAHE",
            [
                {
                    "id": "n1",
                    "module": "clahe",
                    "params": {
                        "clahe": True,
                        "clip_limit": 0.01,
                        "include_in_bank": True,
                    },
                },
                {
                    "id": "n2",
                    "module": "slimsam",
                    "params": {},
                    "input_from": "n1",
                },
                {
                    "id": "n3",
                    "module": "pca",
                    "params": {"dims": 64},
                    "input_from": "n2",
                },
            ],
            ["n1", "n3"],
        ),
        (
            "comp-skimage-mark25",
            "Skimage + Mark25",
            [
                {
                    "id": "n1",
                    "module": "skimage_multiscale",
                    "params": {
                        **_skimage_params(),
                    },
                },
                {
                    "id": "n2",
                    "module": "tomojepa",
                    "params": {
                        "weights_id": "mark25",
                        "input_size": 512,
                        "resize": True,
                    },
                },
                {
                    "id": "n3",
                    "module": "pca",
                    "params": {"dims": 64},
                    "input_from": "n2",
                },
            ],
            ["n1", "n3"],
        ),
        (
            "comp-mark25-clahe",
            "Mark25 + CLAHE",
            [
                {
                    "id": "n1",
                    "module": "clahe",
                    "params": {
                        "clahe": True,
                        "clip_limit": 0.01,
                        "include_in_bank": True,
                    },
                },
                {
                    "id": "n2",
                    "module": "tomojepa",
                    "params": {
                        "weights_id": "mark25",
                        "input_size": 512,
                        "resize": True,
                    },
                    "input_from": "n1",
                },
                {
                    "id": "n3",
                    "module": "pca",
                    "params": {"dims": 64},
                    "input_from": "n2",
                },
            ],
            ["n1", "n3"],
        ),
        (
            "comp-skimage-mark11",
            "Skimage + Mark11",
            [
                {"id": "n1", "module": "skimage_multiscale", "params": _skimage_params()},
                {
                    "id": "n2",
                    "module": "tomojepa",
                    "params": {
                        "weights_id": "mark11",
                        "input_size": 512,
                        "resize": True,
                    },
                },
                {
                    "id": "n3",
                    "module": "pca",
                    "params": {"dims": 64},
                    "input_from": "n2",
                },
            ],
            ["n1", "n3"],
        ),
        (
            "comp-mark11-clahe",
            "Mark11 + CLAHE",
            [
                {
                    "id": "n1",
                    "module": "clahe",
                    "params": {
                        "clahe": True,
                        "clip_limit": 0.01,
                        "include_in_bank": True,
                    },
                },
                {
                    "id": "n2",
                    "module": "tomojepa",
                    "params": {
                        "weights_id": "mark11",
                        "input_size": 512,
                        "resize": True,
                    },
                    "input_from": "n1",
                },
                {
                    "id": "n3",
                    "module": "pca",
                    "params": {"dims": 64},
                    "input_from": "n2",
                },
            ],
            ["n1", "n3"],
        ),
    ]
    created: list[dict[str, Any]] = []
    for cid, name, nodes, outputs in specs:
        existing = load_composition(cid)
        if existing is None:
            created.append(
                save_composition(
                    name=name,
                    nodes=nodes,
                    outputs=outputs,
                    composition_id=cid,
                    builtin=True,
                    catalog=catalog,
                )
            )
        else:
            if catalog is not None:
                catalog.upsert_feature_setup(
                    setup_id=cid,
                    name=existing["name"],
                    kind=KIND_COMPOSITION,
                    content_hash=existing.get("content_hash")
                    or content_hash(existing),
                    meta=existing,
                )
            created.append(existing)
    return created


def resolve_preprocess_id(setup_or_comp_id: str) -> str:
    """Normalize legacy setup id or composition id to a composition id."""
    if load_composition(setup_or_comp_id) is not None:
        return setup_or_comp_id
    mapped = setup_id_to_composition_id(setup_or_comp_id)
    if mapped and load_composition(mapped) is not None:
        return mapped
    # compositions might not be seeded yet
    if mapped:
        return mapped
    raise KeyError(f"unknown composition or setup {setup_or_comp_id}")


__all__ = [
    "KIND_COMPOSITION",
    "content_hash",
    "ensure_default_compositions",
    "list_compositions",
    "list_module_catalog",
    "load_composition",
    "preview_concat_labels",
    "resolve_composition",
    "resolve_preprocess_id",
    "save_composition",
    "setup_id_to_composition_id",
    "validate_composition",
]
