"""Train / infer / rethreshold orchestration."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from ipred import conformal, sam_embed
from ipred.catalog import Catalog
from ipred.labels import build_label_map
from ipred.paths import project_blob_dir
from ipred.preprocess import load_feature_bank_arrays
from ipred.trainers import get_trainer

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shape_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Rough (x0, y0, x1, y1) extent of one shape, for mismatch diagnostics."""
    kind = shape.get("kind")
    try:
        if kind == "rectangle":
            x, y, ww, hh = float(shape["x"]), float(shape["y"]), float(shape["w"]), float(shape["h"])
            return min(x, x + ww), min(y, y + hh), max(x, x + ww), max(y, y + hh)
        if kind == "ellipse":
            cx, cy, rx, ry = (float(shape["cx"]), float(shape["cy"]), float(shape["rx"]), float(shape["ry"]))
            return cx - rx, cy - ry, cx + rx, cy + ry
        if kind == "polygon":
            pts = shape.get("points") or []
            xs = pts[0::2]
            ys = pts[1::2]
            if not xs or not ys:
                return None
            return min(xs), min(ys), max(xs), max(ys)
        if kind == "brush":
            xs: list[float] = []
            ys: list[float] = []
            for stroke in shape.get("strokes") or []:
                pts = stroke.get("points") or []
                xs.extend(pts[0::2])
                ys.extend(pts[1::2])
            if not xs or not ys:
                return None
            return min(xs), min(ys), max(xs), max(ys)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _shape_class_summary(shapes: list[dict[str, Any]]) -> str:
    """Per-class shape count + combined bbox, for error messages when
    rasterization yields fewer classes than the caller annotated."""
    by_class: dict[int, list[tuple[float, float, float, float]]] = {}
    for shape in shapes:
        cid = int(shape.get("classId") or shape.get("class_id") or 0)
        bbox = _shape_bbox(shape)
        by_class.setdefault(cid, [])
        if bbox is not None:
            by_class[cid].append(bbox)
    parts = []
    for cid, boxes in sorted(by_class.items()):
        if boxes:
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes)
            y1 = max(b[3] for b in boxes)
            parts.append(f"class {cid}: {len(boxes)} shape(s) spanning ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")
        else:
            parts.append(f"class {cid}: shape(s) with unparsable extent")
    return "; ".join(parts) if parts else "no shapes"


def stratified_train_cal_split(
    ys: np.ndarray,
    *,
    train_frac: float = 0.8,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified indices into train / calibration."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    train_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    for cls in np.unique(ys):
        idx = np.flatnonzero(ys == cls)
        rng.shuffle(idx)
        n = idx.size
        if n == 1:
            train_parts.append(idx)
            continue
        n_train = max(1, min(n - 1, int(round(n * train_frac))))
        train_parts.append(idx[:n_train])
        cal_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    cal_idx = np.concatenate(cal_parts) if cal_parts else np.array([], dtype=np.int64)
    if cal_idx.size == 0:
        raise ValueError("calibration split empty — need more labeled pixels per class")
    return train_idx, cal_idx


def _stratified_sample(
    ys: np.ndarray, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    n = ys.shape[0]
    if n <= max_samples:
        return np.arange(n)
    classes, counts = np.unique(ys, return_counts=True)
    alloc = np.maximum(1, np.floor(counts / n * max_samples).astype(int))
    while alloc.sum() < max_samples:
        alloc[int(np.argmax(counts - alloc))] += 1
    while alloc.sum() > max_samples:
        i = int(np.argmax(alloc))
        if alloc[i] > 1:
            alloc[i] -= 1
        else:
            break
    chosen: list[np.ndarray] = []
    for cls, k in zip(classes, alloc):
        idx = np.flatnonzero(ys == cls)
        take = min(int(k), idx.size)
        chosen.append(rng.choice(idx, size=take, replace=False))
    return np.concatenate(chosen) if chosen else np.arange(0)


def _raw_pixel_features(
    bank: dict[str, Any],
    yy: np.ndarray,
    xx: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Skimage feature columns + raw (pre-PCA) SAM embedding columns, if any.

    Split out of ``_pixel_features`` so multi-slice training can pool raw SAM
    pixels across slices and fit ONE PCA over the pooled set — fitting PCA
    per-slice would make the reduced columns incomparable slice-to-slice.
    """
    float_stack = bank["float_stack"]
    h, w, _ = float_stack.shape
    ys_i = np.clip(yy.astype(np.int64), 0, h - 1)
    xs_i = np.clip(xx.astype(np.int64), 0, w - 1)
    x_sk = float_stack[ys_i, xs_i].astype(np.float32)
    labels = list(bank["labels"])
    sam_emb = bank.get("sam_emb")
    sam_meta = bank.get("sam_meta")
    # Preprocess may already bake encoder PCA into float_stack channels.
    if sam_meta and sam_meta.get("baked_into_float_stack"):
        return x_sk, labels, None
    if sam_emb is None or sam_meta is None:
        return x_sk, labels, None
    oh, ow = sam_meta["orig_hw"]
    rh, rw = sam_meta["reshaped_hw"]
    x_sam = sam_embed.bilinear_sample_emb(
        sam_emb,
        ys_i.astype(np.float64),
        xs_i.astype(np.float64),
        orig_h=int(oh),
        orig_w=int(ow),
        reshaped_h=int(rh),
        reshaped_w=int(rw),
    )
    return x_sk, labels, x_sam


def _pixel_features(
    bank: dict[str, Any],
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    sam_pca_mean: np.ndarray | None = None,
    sam_pca_components: np.ndarray | None = None,
    fit_pca: bool = False,
    sam_pca_dims: int = 32,
) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None]:
    x_sk, labels, x_sam = _raw_pixel_features(bank, yy, xx)
    if x_sam is None:
        return x_sk, labels, None, None
    mean, comp = sam_pca_mean, sam_pca_components
    if fit_pca:
        mean, comp = sam_embed.fit_pca(x_sam, n_components=sam_pca_dims)
    if mean is not None and comp is not None:
        x_sam = sam_embed.transform_pca(x_sam, mean, comp)
    x = np.concatenate([x_sk, x_sam], axis=1)
    feat_labels = labels + [f"sam{i}" for i in range(x_sam.shape[1])]
    return x, feat_labels, mean, comp


def run_train(
    catalog: Catalog,
    *,
    session_id: str,
    shapes: list[dict[str, Any]],
    feature_id: str | None = None,
    trainer_id: str = "catboost",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a plugin model on session feature bank + shapes."""
    session = catalog.get_session(session_id)
    if session is None:
        raise KeyError(f"unknown session {session_id}")
    fid = feature_id or session.current_feature_id
    if not fid:
        raise ValueError("no feature_id; run preprocess first")
    bank_row = catalog.get_feature_bank(fid)
    if bank_row is None:
        raise KeyError(f"unknown feature bank {fid}")
    bank = load_feature_bank_arrays(bank_row["blob_dir"])
    h, w, _ = bank["float_stack"].shape
    label_map = build_label_map(shapes, h, w)
    if not np.any(label_map > 0):
        raise ValueError(
            f"no labeled pixels — annotate at least one shape "
            f"({_shape_class_summary(shapes)} vs. feature bank {w}x{h})"
        )

    yy, xx = np.nonzero(label_map)
    y_all = label_map[yy, xx].astype(np.int32)
    if len(np.unique(y_all)) < 2:
        raise ValueError(
            "need at least two classes with labeled pixels — got "
            f"{sorted(int(c) for c in np.unique(y_all))} after rasterizing to the feature "
            f"bank ({w}x{h}); shapes sent: {_shape_class_summary(shapes)}. If the shape "
            "classes/extents don't match, the feature bank may be at a different "
            "resolution than the annotated slice."
        )

    cfg = dict(config or {})
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)))
    max_samples = int(cfg.get("max_samples", 200_000))
    train_frac = float(cfg.get("train_frac", 0.8))
    cap_idx = _stratified_sample(y_all, max_samples, rng)
    yy_c, xx_c, y_c = yy[cap_idx], xx[cap_idx], y_all[cap_idx]
    train_idx, cal_idx = stratified_train_cal_split(y_c, train_frac=train_frac, rng=rng)

    uses_sam = (
        bank.get("sam_emb") is not None
        and not (bank.get("sam_meta") or {}).get("baked_into_float_stack")
    )
    x_tr, feat_labels, pca_mean, pca_comp = _pixel_features(
        bank,
        yy_c[train_idx],
        xx_c[train_idx],
        fit_pca=uses_sam,
        sam_pca_dims=int(cfg.get("sam_pca_dims", 32)),
    )
    x_cal, _, _, _ = _pixel_features(
        bank,
        yy_c[cal_idx],
        xx_c[cal_idx],
        sam_pca_mean=pca_mean,
        sam_pca_components=pca_comp,
    )

    trainer = get_trainer(trainer_id)
    arts = trainer.train(
        x_tr,
        y_c[train_idx],
        x_cal,
        y_c[cal_idx],
        feature_labels=feat_labels,
        config=cfg,
    )
    return _save_trained_model(
        catalog,
        session=session,
        session_id=session_id,
        fid=fid,
        trainer_id=trainer_id,
        arts=arts,
        uses_sam=uses_sam,
        train_frac=train_frac,
        pca_mean=pca_mean,
        pca_comp=pca_comp,
    )


def _save_trained_model(
    catalog: Catalog,
    *,
    session: Any,
    session_id: str,
    fid: str,
    trainer_id: str,
    arts: Any,
    uses_sam: bool,
    train_frac: float,
    pca_mean: np.ndarray | None,
    pca_comp: np.ndarray | None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a trained ``ModelArtifacts`` to blob storage + the catalog.

    Shared tail of ``run_train`` and ``run_train_multi_slice`` — the only
    difference between single- and multi-slice training is how ``arts`` (and
    the pooled SAM PCA, if any) got built; saving/cataloging is identical.
    """
    arts.params["uses_sam"] = uses_sam
    arts.params["train_frac"] = train_frac

    model_id = uuid.uuid4().hex
    blob = project_blob_dir(session.project_id) / "models" / model_id
    blob.mkdir(parents=True, exist_ok=True)
    trainer = get_trainer(trainer_id)
    trainer.save(arts.model_handle, blob)
    feature_importances = list(arts.extras.get("feature_importances") or [])
    meta = {
        "model_id": model_id,
        "feature_id": fid,
        "trainer_id": trainer_id,
        "class_ids": arts.class_ids,
        "feature_labels": arts.feature_labels,
        "cal_scores_by_class": arts.cal_scores_by_class,
        "train_accuracy": arts.train_accuracy,
        "n_train": arts.n_train,
        "n_cal": arts.n_cal,
        "n_samples": arts.n_samples,
        "params": arts.params,
        "uses_sam": uses_sam,
        "feature_importances": feature_importances,
        **(extra_meta or {}),
    }
    if pca_mean is not None and pca_comp is not None:
        np.savez(blob / "sam_pca.npz", mean=pca_mean, components=pca_comp)
        meta["has_sam_pca"] = True
    (blob / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (blob / "cal_scores.json").write_text(
        json.dumps(arts.cal_scores_by_class), encoding="utf-8"
    )

    catalog.insert_model(
        {
            "model_id": model_id,
            "project_id": session.project_id,
            "feature_id": fid,
            "trainer_id": trainer_id,
            "blob_dir": str(blob),
            "meta_json": json.dumps(meta),
            "created_at": _utc_now(),
        }
    )
    catalog.set_session_currents(session_id, model_id=model_id)
    return {
        "model_id": model_id,
        "feature_id": fid,
        "trainer_id": trainer_id,
        "class_ids": arts.class_ids,
        "train_accuracy": arts.train_accuracy,
        "n_train": arts.n_train,
        "n_cal": arts.n_cal,
        "n_samples": arts.n_samples,
        "params": arts.params,
        "feature_importances": feature_importances,
        **(extra_meta or {}),
    }


def run_train_multi_slice(
    catalog: Catalog,
    *,
    session_id: str,
    per_slice_shapes: dict[int, list[dict[str, Any]]],
    feature_ids: dict[int, str],
    trainer_id: str = "catboost",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one model pooling labeled pixels across multiple slices.

    Each slice keeps its own feature bank (its own composition run against
    that slice), but SAM PCA — when in play — is fit ONCE on pixels pooled
    across every slice's training split, so the reduced columns mean the same
    thing regardless of which slice a pixel came from. A per-slice PCA fit
    (naively looping ``run_train``) would make them incomparable.
    """
    session = catalog.get_session(session_id)
    if session is None:
        raise KeyError(f"unknown session {session_id}")
    if not per_slice_shapes:
        raise ValueError("no slices with shapes to train on")

    cfg = dict(config or {})
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)))
    max_samples = int(cfg.get("max_samples", 200_000))
    train_frac = float(cfg.get("train_frac", 0.8))
    per_slice_cap = max(1, max_samples // max(1, len(per_slice_shapes)))

    banks: dict[int, dict[str, Any]] = {}
    train_parts: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    cal_parts: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    uses_sam = False
    summaries: list[str] = []

    for slice_index, shapes in sorted(per_slice_shapes.items()):
        fid = feature_ids.get(slice_index)
        if not fid:
            raise ValueError(f"no feature_id for slice {slice_index}")
        bank_row = catalog.get_feature_bank(fid)
        if bank_row is None:
            raise KeyError(f"unknown feature bank {fid}")
        bank = load_feature_bank_arrays(bank_row["blob_dir"])
        banks[slice_index] = bank
        h, w, _ = bank["float_stack"].shape
        label_map = build_label_map(shapes, h, w)
        if not np.any(label_map > 0):
            summaries.append(f"slice {slice_index}: no labeled pixels")
            continue
        yy, xx = np.nonzero(label_map)
        y_all = label_map[yy, xx].astype(np.int32)
        cap_idx = _stratified_sample(y_all, per_slice_cap, rng)
        yy_c, xx_c, y_c = yy[cap_idx], xx[cap_idx], y_all[cap_idx]
        train_idx, cal_idx = stratified_train_cal_split(y_c, train_frac=train_frac, rng=rng)
        train_parts.append((slice_index, yy_c[train_idx], xx_c[train_idx], y_c[train_idx]))
        cal_parts.append((slice_index, yy_c[cal_idx], xx_c[cal_idx], y_c[cal_idx]))
        uses_sam = uses_sam or (
            bank.get("sam_emb") is not None
            and not (bank.get("sam_meta") or {}).get("baked_into_float_stack")
        )
        summaries.append(
            f"slice {slice_index}: {y_all.size} labeled px, "
            f"classes {sorted(int(c) for c in np.unique(y_all))}"
        )

    if not train_parts:
        raise ValueError(
            "no labeled pixels across the given slices — " + "; ".join(summaries)
        )

    all_y_train = np.concatenate([p[3] for p in train_parts])
    all_y_cal = (
        np.concatenate([p[3] for p in cal_parts])
        if cal_parts
        else np.array([], dtype=np.int32)
    )
    if len(np.unique(np.concatenate([all_y_train, all_y_cal]))) < 2:
        raise ValueError(
            "need at least two classes with labeled pixels across the selected "
            "slices — " + "; ".join(summaries)
        )

    # Raw (pre-PCA) extraction per slice, pooled before any PCA fit.
    x_sk_train_parts, x_sam_train_parts = [], []
    x_sk_cal_parts, x_sam_cal_parts = [], []
    feat_labels: list[str] | None = None
    for slice_index, yy, xx, _y in train_parts:
        x_sk, labels, x_sam = _raw_pixel_features(banks[slice_index], yy, xx)
        feat_labels = feat_labels or labels
        x_sk_train_parts.append(x_sk)
        if x_sam is not None:
            x_sam_train_parts.append(x_sam)
    for slice_index, yy, xx, _y in cal_parts:
        x_sk, _labels, x_sam = _raw_pixel_features(banks[slice_index], yy, xx)
        x_sk_cal_parts.append(x_sk)
        if x_sam is not None:
            x_sam_cal_parts.append(x_sam)

    x_sk_train = np.concatenate(x_sk_train_parts, axis=0)
    x_sk_cal = (
        np.concatenate(x_sk_cal_parts, axis=0)
        if x_sk_cal_parts
        else np.empty((0, x_sk_train.shape[1]), dtype=np.float32)
    )

    pca_mean = pca_comp = None
    if uses_sam and x_sam_train_parts:
        x_sam_train_raw = np.concatenate(x_sam_train_parts, axis=0)
        x_sam_cal_raw = (
            np.concatenate(x_sam_cal_parts, axis=0)
            if x_sam_cal_parts
            else np.empty((0, x_sam_train_raw.shape[1]), dtype=x_sam_train_raw.dtype)
        )
        pca_mean, pca_comp = sam_embed.fit_pca(
            x_sam_train_raw, n_components=int(cfg.get("sam_pca_dims", 32))
        )
        x_sam_train = sam_embed.transform_pca(x_sam_train_raw, pca_mean, pca_comp)
        x_sam_cal = (
            sam_embed.transform_pca(x_sam_cal_raw, pca_mean, pca_comp)
            if x_sam_cal_raw.shape[0]
            else np.empty((0, pca_comp.shape[0]), dtype=np.float32)
        )
        x_tr = np.concatenate([x_sk_train, x_sam_train], axis=1)
        x_cal = (
            np.concatenate([x_sk_cal, x_sam_cal], axis=1)
            if x_sk_cal.shape[0]
            else np.empty((0, x_tr.shape[1]), dtype=np.float32)
        )
        feat_labels = (feat_labels or []) + [f"sam{i}" for i in range(x_sam_train.shape[1])]
    else:
        x_tr = x_sk_train
        x_cal = x_sk_cal

    trainer = get_trainer(trainer_id)
    arts = trainer.train(
        x_tr, all_y_train, x_cal, all_y_cal, feature_labels=feat_labels or [], config=cfg
    )
    trained_slices = sorted(slice_index for slice_index, *_ in train_parts)
    # The default feature_id is only an infer-time fallback when a caller doesn't
    # pass one explicitly — real infer calls always target one specific slice.
    default_feature_id = feature_ids[trained_slices[0]]
    return _save_trained_model(
        catalog,
        session=session,
        session_id=session_id,
        fid=default_feature_id,
        trainer_id=trainer_id,
        arts=arts,
        uses_sam=uses_sam,
        train_frac=train_frac,
        pca_mean=pca_mean,
        pca_comp=pca_comp,
        extra_meta={"trained_slice_indices": trained_slices},
    )


def run_infer(
    catalog: Catalog,
    *,
    session_id: str,
    model_id: str | None = None,
    feature_id: str | None = None,
    alpha: float = 0.05,
    row_chunk: int = 128,
) -> dict[str, Any]:
    """Full-image predict_proba + conformal maps; persist float16 proba."""
    session = catalog.get_session(session_id)
    if session is None:
        raise KeyError(f"unknown session {session_id}")
    mid = model_id or session.current_model_id
    fid = feature_id or session.current_feature_id
    if not mid or not fid:
        raise ValueError("model_id and feature_id required (train/preprocess first)")
    model_row = catalog.get_model(mid)
    bank_row = catalog.get_feature_bank(fid)
    if model_row is None or bank_row is None:
        raise KeyError("model or feature bank missing")

    blob_model = Path(model_row["blob_dir"])
    meta = json.loads((blob_model / "meta.json").read_text(encoding="utf-8"))
    trainer = get_trainer(model_row["trainer_id"])
    handle = trainer.load(blob_model)
    bank = load_feature_bank_arrays(bank_row["blob_dir"])
    pca_mean = pca_comp = None
    if (blob_model / "sam_pca.npz").is_file():
        z = np.load(blob_model / "sam_pca.npz")
        pca_mean, pca_comp = z["mean"], z["components"]

    class_ids = [int(c) for c in meta["class_ids"]]
    h, w, _ = bank["float_stack"].shape
    k = len(class_ids)
    proba = np.empty((h, w, k), dtype=np.float16)

    for y0 in range(0, h, row_chunk):
        y1 = min(h, y0 + row_chunk)
        yy = np.repeat(np.arange(y0, y1), w)
        xx = np.tile(np.arange(w), y1 - y0)
        x, _, _, _ = _pixel_features(
            bank, yy, xx, sam_pca_mean=pca_mean, sam_pca_components=pca_comp
        )
        block = trainer.predict_proba(handle, x).astype(np.float32)
        # Align columns to meta class_ids order
        model_classes = [int(c) for c in np.asarray(handle.classes_).tolist()]
        if model_classes != class_ids:
            remap = np.zeros_like(block)
            for j, cid in enumerate(class_ids):
                if cid in model_classes:
                    remap[:, j] = block[:, model_classes.index(cid)]
            block = remap
        proba[y0:y1] = block.reshape(y1 - y0, w, k).astype(np.float16)

    cal = conformal.cal_scores_from_json(meta["cal_scores_by_class"])
    q_by_class = conformal.mondrian_thresholds(cal, alpha)
    commit, status, membership = conformal.maps_from_proba(
        proba.astype(np.float32), class_ids, q_by_class
    )
    counts = conformal.counts_from_status(status)

    run_id = uuid.uuid4().hex
    blob = project_blob_dir(session.project_id) / "runs" / run_id
    blob.mkdir(parents=True, exist_ok=True)
    np.save(blob / "proba.npy", proba)
    np.save(blob / "commit.npy", commit)
    np.save(blob / "status.npy", status)
    np.save(blob / "membership.npy", membership)
    _write_label_png(blob / "commit.png", commit)
    _write_label_png(blob / "status.png", status)

    run_meta = {
        "run_id": run_id,
        "model_id": mid,
        "feature_id": fid,
        "alpha": alpha,
        "class_ids": class_ids,
        "q_by_class": {str(k_): v for k_, v in q_by_class.items()},
        "counts": counts,
    }
    (blob / "meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    now = _utc_now()
    catalog.insert_run(
        {
            "run_id": run_id,
            "project_id": session.project_id,
            "model_id": mid,
            "feature_id": fid,
            "alpha": float(alpha),
            "blob_dir": str(blob),
            "meta_json": json.dumps(run_meta),
            "created_at": now,
            "updated_at": now,
        }
    )
    catalog.set_session_currents(session_id, run_id=run_id)
    return {**run_meta, "blob_dir": str(blob)}


def run_rethreshold(
    catalog: Catalog,
    *,
    session_id: str,
    run_id: str | None = None,
    alpha: float,
) -> dict[str, Any]:
    """Recompute membership/commit/status from cached proba (no re-infer)."""
    session = catalog.get_session(session_id)
    if session is None:
        raise KeyError(f"unknown session {session_id}")
    rid = run_id or session.current_run_id
    if not rid:
        raise ValueError("run_id required")
    run = catalog.get_run(rid)
    if run is None:
        raise KeyError(f"unknown run {rid}")
    blob = Path(run["blob_dir"])
    proba = np.load(blob / "proba.npy").astype(np.float32)
    run_meta = json.loads((blob / "meta.json").read_text(encoding="utf-8"))
    model_row = catalog.get_model(run["model_id"])
    if model_row is None:
        raise KeyError("model missing for run")
    model_meta = json.loads(
        Path(model_row["blob_dir"], "meta.json").read_text(encoding="utf-8")
    )
    cal = conformal.cal_scores_from_json(model_meta["cal_scores_by_class"])
    class_ids = [int(c) for c in run_meta["class_ids"]]
    q_by_class = conformal.mondrian_thresholds(cal, alpha)
    commit, status, membership = conformal.maps_from_proba(proba, class_ids, q_by_class)
    counts = conformal.counts_from_status(status)
    np.save(blob / "commit.npy", commit)
    np.save(blob / "status.npy", status)
    np.save(blob / "membership.npy", membership)
    _write_label_png(blob / "commit.png", commit)
    _write_label_png(blob / "status.png", status)
    run_meta.update(
        {
            "alpha": alpha,
            "q_by_class": {str(k): v for k, v in q_by_class.items()},
            "counts": counts,
        }
    )
    (blob / "meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    catalog.update_run(rid, alpha=float(alpha), meta_json=json.dumps(run_meta))
    catalog.set_session_currents(session_id, run_id=rid)
    return {**run_meta, "blob_dir": str(blob)}


def _write_label_png(path: Path, arr: np.ndarray) -> None:
    img = PILImage.fromarray(arr.astype(np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def _run_blob(catalog: Catalog, run_id: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    run = catalog.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run {run_id}")
    blob = Path(run["blob_dir"])
    meta = json.loads(run["meta_json"])
    return run, blob, meta


def proba_heatmap_png(catalog: Catalog, run_id: str, class_index: int) -> bytes:
    """Return grayscale PNG of softmax channel ``class_index`` (0…K-1)."""
    _run, blob, meta = _run_blob(catalog, run_id)
    proba_path = blob / "proba.npy"
    if not proba_path.is_file():
        raise FileNotFoundError("proba.npy missing — run Predict first")
    proba = np.load(proba_path)
    if proba.ndim != 3:
        raise ValueError("proba must be HxWxK")
    k = int(proba.shape[2])
    if not 0 <= int(class_index) < k:
        raise ValueError(f"class_index {class_index} out of range 0..{k - 1}")
    ch = np.clip(proba[:, :, int(class_index)].astype(np.float32), 0.0, 1.0)
    uint8 = np.clip(np.round(ch * 255.0), 0, 255).astype(np.uint8)
    buf = BytesIO()
    PILImage.fromarray(uint8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def threshold_class_label_map(
    catalog: Catalog,
    run_id: str,
    *,
    class_id: int,
    threshold: float,
) -> dict[str, Any]:
    """Binary class map from softmax: ``class_id`` where ``p >= threshold``, else 0.

    Returns width/height and base64-encoded uint8 label map for mask-set caches.
    """
    import base64

    _run, blob, meta = _run_blob(catalog, run_id)
    class_ids = [int(c) for c in meta.get("class_ids") or []]
    if int(class_id) not in class_ids:
        raise ValueError(f"class_id {class_id} not in run class_ids {class_ids}")
    j = class_ids.index(int(class_id))
    proba_path = blob / "proba.npy"
    if not proba_path.is_file():
        raise FileNotFoundError("proba.npy missing — run Predict first")
    proba = np.load(proba_path).astype(np.float32)
    t = float(threshold)
    if not 0.0 <= t <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    h, w, _ = proba.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[proba[:, :, j] >= t] = np.uint8(int(class_id))
    raw = labels.reshape(-1).tobytes()
    return {
        "run_id": run_id,
        "class_id": int(class_id),
        "class_index": int(j),
        "threshold": t,
        "width": int(w),
        "height": int(h),
        "n_positive": int(np.count_nonzero(labels)),
        "label_map_b64": base64.b64encode(raw).decode("ascii"),
    }
