"""# REMOVE THIS AND USE YOUR OWN STUFF

Persist trained CatBoost + Mondrian cal scores + feature recipe to disk.

TEMPORARY scaffold under ``$LOCAL_DATA_ROOT/.clf_models/<id>/``. Replace with
your own model registry, then delete this module and the ``/api/clf/models``
shelf routes + ModelShelfPanel.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier

import pixel_clf as pixel_clf_mod
from pixel_clf import ClfModel

logger = logging.getLogger(__name__)

# ##############################################################################
# # REMOVE THIS AND USE YOUR OWN STUFF — on-disk CatBoost model shelf
# ##############################################################################


def _shelf_root() -> Path:
    return Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve() / ".clf_models"


@dataclass(frozen=True)
class ShelfMeta:
    """Indexed metadata for a persisted classifier."""

    id: str
    name: str
    class_ids: list[int]
    n_train: int
    n_cal: int
    train_accuracy: float
    uses_sam: bool
    feature_recipe: dict[str, Any]
    created_at: str
    n_features: int


def _dir_for(model_id: str) -> Path:
    return _shelf_root() / model_id


def save_model(
    model: ClfModel,
    *,
    name: str,
    feature_recipe: dict[str, Any],
) -> ShelfMeta:
    """Write CatBoost, cal scores, PCA, and recipe to disk; return shelf meta."""
    from datetime import datetime, timezone

    shelf_id = uuid.uuid4().hex
    dest = _dir_for(shelf_id)
    dest.mkdir(parents=True, exist_ok=True)

    cb_path = dest / "model.cbm"
    model.model.save_model(str(cb_path))

    cal: dict[str, list[float]] = {
        str(k): [float(x) for x in v.tolist()] for k, v in model.cal_scores_by_class.items()
    }
    (dest / "cal_scores.json").write_text(json.dumps(cal), encoding="utf-8")

    if model.sam_pca_mean is not None and model.sam_pca_components is not None:
        np.savez_compressed(
            dest / "sam_pca.npz",
            mean=model.sam_pca_mean,
            components=model.sam_pca_components,
        )

    n_features = int(getattr(model.model, "feature_count_", 0) or 0)
    if not n_features and model.feature_labels:
        n_features = len(model.feature_labels)

    meta = {
        "id": shelf_id,
        "name": name,
        "class_ids": list(model.class_ids),
        "n_train": model.n_train,
        "n_cal": model.n_cal,
        "n_samples": model.n_samples,
        "train_accuracy": model.train_accuracy,
        "uses_sam": model.uses_sam,
        "feature_recipe": feature_recipe,
        "feature_labels": list(model.feature_labels),
        "params": model.params,
        "n_trees": model.n_trees,
        "n_features": n_features,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved classifier shelf model %s (%s)", shelf_id, name)
    return ShelfMeta(
        id=shelf_id,
        name=name,
        class_ids=list(model.class_ids),
        n_train=model.n_train,
        n_cal=model.n_cal,
        train_accuracy=model.train_accuracy,
        uses_sam=model.uses_sam,
        feature_recipe=feature_recipe,
        created_at=meta["created_at"],
        n_features=n_features,
    )


def list_models() -> list[dict[str, Any]]:
    """List saved shelf models (meta only)."""
    root = _shelf_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out.append(meta)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skip corrupt shelf entry %s: %s", child, exc)
    return out


def delete_model(shelf_id: str) -> bool:
    """Remove a shelf model directory. Returns False if missing."""
    dest = _dir_for(shelf_id)
    if not dest.is_dir():
        return False
    for p in dest.iterdir():
        p.unlink(missing_ok=True)
    dest.rmdir()
    return True


def load_into_cache(shelf_id: str) -> ClfModel:
    """Load a shelf model into the in-memory CatBoost cache for predict."""
    dest = _dir_for(shelf_id)
    meta_path = dest / "meta.json"
    cb_path = dest / "model.cbm"
    if not meta_path.is_file() or not cb_path.is_file():
        raise FileNotFoundError(f"shelf model {shelf_id} not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clf = CatBoostClassifier()
    clf.load_model(str(cb_path))

    cal_raw = json.loads((dest / "cal_scores.json").read_text(encoding="utf-8"))
    cal_scores = {int(k): np.asarray(v, dtype=np.float64) for k, v in cal_raw.items()}

    pca_mean = None
    pca_comp = None
    pca_path = dest / "sam_pca.npz"
    if pca_path.is_file():
        z = np.load(pca_path)
        pca_mean = z["mean"]
        pca_comp = z["components"]

    # Dump model json for tree browsing (best-effort)
    model_json: dict[str, Any] = {}
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            clf.save_model(str(tmp_path), format="json")
            model_json = json.loads(tmp_path.read_text(encoding="utf-8"))
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        model_json = {}

    n_trees = len(model_json.get("oblivious_trees") or [])
    feat_labels = tuple(meta.get("feature_labels") or [])
    fi_raw = np.asarray(clf.get_feature_importance(), dtype=np.float64)
    if len(feat_labels) != fi_raw.size:
        feat_labels = tuple(f"f{i}" for i in range(fi_raw.size))
    order = np.argsort(-fi_raw)
    feature_importances = tuple((feat_labels[int(i)], float(fi_raw[int(i)])) for i in order)

    entry = ClfModel(
        model_id=f"shelf:{shelf_id}",
        feature_job_id=f"shelf:{shelf_id}",
        class_ids=tuple(int(x) for x in meta["class_ids"]),
        n_samples=int(meta.get("n_samples") or 0),
        n_train=int(meta.get("n_train") or 0),
        n_cal=int(meta.get("n_cal") or 0),
        model=clf,
        feature_labels=feat_labels,
        train_accuracy=float(meta.get("train_accuracy") or 0),
        feature_importances=feature_importances,
        params=dict(meta.get("params") or {}),
        n_trees=n_trees,
        cal_scores_by_class=cal_scores,
        _model_json=model_json,
        sam_pca_mean=pca_mean,
        sam_pca_components=pca_comp,
        uses_sam=bool(meta.get("uses_sam")),
    )
    pixel_clf_mod._model_cache.set(entry.model_id, entry)
    return entry


def get_meta(shelf_id: str) -> dict[str, Any]:
    """Return meta.json for a shelf id."""
    meta_path = _dir_for(shelf_id) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"shelf model {shelf_id} not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))
