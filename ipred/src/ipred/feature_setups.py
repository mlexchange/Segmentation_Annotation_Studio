"""Feature Setup shelf — procedure or weights configurations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ipred.catalog import Catalog
from ipred.paths import feature_models_root

logger = logging.getLogger(__name__)

PROCEDURE_SKIMAGE = "skimage_multiscale_v1"
PROCEDURE_CLAHE_ENCODER = "clahe_encoder_v1"
WEIGHTS_ONNX_VISION = "onnx_vision_encoder_v1"
WEIGHTS_TOMOJEPA_MARK25 = "torch_tomojepa_mark25_v1"
WEIGHTS_TOMOJEPA_MARK11 = "torch_tomojepa_mark11_v1"
TOMOJEPA_WEIGHTS_FORMATS = frozenset(
    {WEIGHTS_TOMOJEPA_MARK25, WEIGHTS_TOMOJEPA_MARK11}
)

DEFAULT_SKIMAGE_PARAMS: dict[str, Any] = {
    "sigma_min": 1.0,
    "sigma_max": 8.0,
    "intensity": True,
    "edges": True,
    "texture": True,
    "clahe": True,
}

DEFAULT_CLAHE_ENCODER_PARAMS: dict[str, Any] = {
    "clahe": True,
    "clip_limit": 0.01,
    "kernel_size": None,
    "resize": True,
    "input_size": 512,
    "pca_dims": 64,
}

DEFAULT_SKIMAGE_MARK25_PARAMS: dict[str, Any] = {
    **DEFAULT_SKIMAGE_PARAMS,
    "resize": True,
    "input_size": 512,
    "pca_dims": 64,
}


def content_hash(meta: dict[str, Any]) -> str:
    """Stable hash of setup config (excludes timestamps / name display)."""
    payload = {
        "id": meta.get("id"),
        "kind": meta.get("kind"),
        "procedure_id": meta.get("procedure_id"),
        "params": meta.get("params"),
        "encoder_setup_id": meta.get("encoder_setup_id"),
        "weights_path": meta.get("weights_path"),
        "weights_format": meta.get("weights_format"),
        "inference": meta.get("inference"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _dir(setup_id: str) -> Path:
    return feature_models_root() / setup_id


def _write_meta(meta: dict[str, Any]) -> Path:
    dest = _dir(meta["id"])
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_setup(setup_id: str) -> dict[str, Any] | None:
    """Load a setup meta.json or None."""
    path = _dir(setup_id) / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_setups() -> list[dict[str, Any]]:
    """List all Feature Setups on disk."""
    root = feature_models_root()
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(meta)
    return out


def save_setup(
    *,
    name: str,
    kind: str,
    procedure_id: str | None = None,
    params: dict[str, Any] | None = None,
    encoder_setup_id: str | None = None,
    weights_path: str | None = None,
    weights_format: str | None = None,
    inference: dict[str, Any] | None = None,
    setup_id: str | None = None,
    builtin: bool = False,
    catalog: Catalog | None = None,
) -> dict[str, Any]:
    """Create or overwrite a Feature Setup."""
    if kind not in ("procedure", "weights"):
        raise ValueError("kind must be 'procedure' or 'weights'")
    sid = setup_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    existing = load_setup(sid)
    created = existing.get("created_at") if existing else now
    meta: dict[str, Any] = {
        "id": sid,
        "name": (name or sid).strip() or sid,
        "kind": kind,
        "builtin": bool(builtin),
        "created_at": created,
        "updated_at": now,
    }
    if kind == "procedure":
        if not procedure_id:
            raise ValueError("procedure setups require procedure_id")
        meta["procedure_id"] = procedure_id
        if params is not None:
            meta["params"] = dict(params)
        elif procedure_id == PROCEDURE_CLAHE_ENCODER:
            meta["params"] = dict(DEFAULT_CLAHE_ENCODER_PARAMS)
        else:
            meta["params"] = dict(DEFAULT_SKIMAGE_PARAMS)
        if encoder_setup_id:
            meta["encoder_setup_id"] = encoder_setup_id
    else:
        if not weights_path:
            raise ValueError("weights setups require weights_path")
        meta["weights_path"] = weights_path
        meta["weights_format"] = weights_format or WEIGHTS_ONNX_VISION
        meta["inference"] = dict(inference or {})

    meta["content_hash"] = content_hash(meta)
    _write_meta(meta)
    if catalog is not None:
        catalog.upsert_feature_setup(
            setup_id=sid,
            name=meta["name"],
            kind=kind,
            content_hash=meta["content_hash"],
            meta=meta,
        )
    return meta


def resolve_slimsam_weights_path() -> str | None:
    """Locate SlimSAM ONNX; honors FEATURE_ENCODER_ONNX."""
    env = os.getenv("FEATURE_ENCODER_ONNX")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return str(p)
    # Prefer repo-adjacent locations without importing annotate backend.
    here = Path(__file__).resolve()
    repo = here.parents[3]  # .../repo/ipred/src/ipred → repo
    candidates = [
        repo / "backend" / "models" / "slimsam-77-uniform" / "onnx" / "vision_encoder.onnx",
        repo / "frontend" / "public" / "models" / "slimsam-77-uniform" / "onnx" / "vision_encoder.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def resolve_tomojepa_weights_path(
    filename: str = "tomojepa25.pth",
    *,
    env_var: str | None = "TOMOJEPA_WEIGHTS",
) -> str | None:
    """Locate a TomoJEPA ``.pth``; honors ``env_var`` when set."""
    if env_var:
        env = os.getenv(env_var)
        if env:
            p = Path(env).expanduser().resolve()
            if p.is_file():
                return str(p)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "models" / filename,
        here.parents[3] / "ipred" / "models" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def _upsert_existing(meta: dict[str, Any], catalog: Catalog | None) -> None:
    if catalog is None:
        return
    catalog.upsert_feature_setup(
        setup_id=meta["id"],
        name=meta["name"],
        kind=meta["kind"],
        content_hash=meta.get("content_hash") or content_hash(meta),
        meta=meta,
    )


def _ensure_procedure_with_param_backfill(
    *,
    setup_id: str,
    name: str,
    procedure_id: str,
    params: dict[str, Any],
    encoder_setup_id: str | None,
    catalog: Catalog | None,
) -> dict[str, Any]:
    """Create procedure setup or backfill missing default params."""
    existing = load_setup(setup_id)
    if existing is None:
        return save_setup(
            setup_id=setup_id,
            name=name,
            kind="procedure",
            procedure_id=procedure_id,
            params=dict(params),
            encoder_setup_id=encoder_setup_id,
            builtin=True,
            catalog=catalog,
        )
    cur = dict(existing.get("params") or {})
    changed = False
    for key, val in params.items():
        if key not in cur:
            cur[key] = val
            changed = True
    if changed:
        return save_setup(
            setup_id=setup_id,
            name=existing.get("name") or name,
            kind="procedure",
            procedure_id=procedure_id,
            params=cur,
            encoder_setup_id=existing.get("encoder_setup_id") or encoder_setup_id,
            builtin=True,
            catalog=catalog,
        )
    _upsert_existing(existing, catalog)
    return existing


def _ensure_tomojepa_variant(
    catalog: Catalog | None,
    *,
    weights_id: str,
    weights_name: str,
    combo_id: str,
    combo_name: str,
    clahe_id: str,
    clahe_name: str,
    filename: str,
    env_var: str,
    weights_format: str,
) -> list[dict[str, Any]]:
    """Weights + skimage combo + CLAHE procedure for one TomoJEPA checkpoint."""
    out: list[dict[str, Any]] = []
    tomo_path = resolve_tomojepa_weights_path(filename, env_var=env_var) or ""
    weights = load_setup(weights_id)
    if weights is None:
        weights = save_setup(
            setup_id=weights_id,
            name=weights_name,
            kind="weights",
            weights_path=tomo_path or "(missing)",
            weights_format=weights_format,
            inference={"input_size": 512, "pca_dims": 32},
            builtin=True,
            catalog=catalog,
        )
    else:
        _upsert_existing(weights, catalog)
    out.append(weights)

    out.append(
        _ensure_procedure_with_param_backfill(
            setup_id=combo_id,
            name=combo_name,
            procedure_id=PROCEDURE_SKIMAGE,
            params=dict(DEFAULT_SKIMAGE_MARK25_PARAMS),
            encoder_setup_id=weights_id,
            catalog=catalog,
        )
    )
    out.append(
        _ensure_procedure_with_param_backfill(
            setup_id=clahe_id,
            name=clahe_name,
            procedure_id=PROCEDURE_CLAHE_ENCODER,
            params=dict(DEFAULT_CLAHE_ENCODER_PARAMS),
            encoder_setup_id=weights_id,
            catalog=catalog,
        )
    )
    return out


def ensure_default_setups(catalog: Catalog | None = None) -> list[dict[str, Any]]:
    """Create built-in setups if missing; return all defaults."""
    created: list[dict[str, Any]] = []
    sk = load_setup("default-skimage")
    if sk is None:
        sk = save_setup(
            setup_id="default-skimage",
            name="Default skimage multiscale",
            kind="procedure",
            procedure_id=PROCEDURE_SKIMAGE,
            params=dict(DEFAULT_SKIMAGE_PARAMS),
            builtin=True,
            catalog=catalog,
        )
    else:
        _upsert_existing(sk, catalog)
    created.append(sk)

    weights_path = resolve_slimsam_weights_path() or ""
    slim = load_setup("default-slimsam")
    if slim is None:
        slim = save_setup(
            setup_id="default-slimsam",
            name="Default SlimSAM encoder",
            kind="weights",
            weights_path=weights_path or "(missing)",
            weights_format=WEIGHTS_ONNX_VISION,
            inference={"input_size": 1024, "pca_dims": 32},
            builtin=True,
            catalog=catalog,
        )
    else:
        _upsert_existing(slim, catalog)
    created.append(slim)

    combo = load_setup("default-skimage-slimsam")
    if combo is None:
        combo = save_setup(
            setup_id="default-skimage-slimsam",
            name="Default skimage + SlimSAM",
            kind="procedure",
            procedure_id=PROCEDURE_SKIMAGE,
            params=dict(DEFAULT_SKIMAGE_PARAMS),
            encoder_setup_id="default-slimsam",
            builtin=True,
            catalog=catalog,
        )
    else:
        _upsert_existing(combo, catalog)
    created.append(combo)

    created.append(
        _ensure_procedure_with_param_backfill(
            setup_id="default-slimsam-clahe",
            name="Default SlimSAM + CLAHE",
            procedure_id=PROCEDURE_CLAHE_ENCODER,
            params=dict(DEFAULT_CLAHE_ENCODER_PARAMS),
            encoder_setup_id="default-slimsam",
            catalog=catalog,
        )
    )

    for variant in (
        {
            "weights_id": "default-mark25",
            "weights_name": "Default Mark25 TomoJEPA encoder",
            "combo_id": "default-skimage-mark25",
            "combo_name": "Default skimage + Mark25",
            "clahe_id": "default-mark25-clahe",
            "clahe_name": "Default Mark25 + CLAHE",
            "filename": "tomojepa25.pth",
            "env_var": "TOMOJEPA_WEIGHTS",
            "weights_format": WEIGHTS_TOMOJEPA_MARK25,
        },
        {
            "weights_id": "default-mark11",
            "weights_name": "Default Mark11 TomoJEPA encoder",
            "combo_id": "default-skimage-mark11",
            "combo_name": "Default skimage + Mark11",
            "clahe_id": "default-mark11-clahe",
            "clahe_name": "Default Mark11 + CLAHE",
            "filename": "tomojepa11.pth",
            "env_var": "TOMOJEPA11_WEIGHTS",
            "weights_format": WEIGHTS_TOMOJEPA_MARK11,
        },
    ):
        created.extend(_ensure_tomojepa_variant(catalog, **variant))

    return created


def resolve_setup(setup_id: str) -> dict[str, Any]:
    """Load setup or raise KeyError."""
    meta = load_setup(setup_id)
    if meta is None:
        raise KeyError(f"unknown feature setup {setup_id}")
    return meta
