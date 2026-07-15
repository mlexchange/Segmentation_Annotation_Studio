"""CatBoost pixel classifier on multiscale (+ optional SlimSAM) features.

Trains on an 80% stratified split of annotated pixels; holds out 20% for
Mondrian (class-conditional) split-conformal calibration. Predict returns
per-pixel prediction sets at user-chosen alpha: commit map = singletons only.
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from PIL import Image as PILImage

from cache import TTLCache
from coco_export import shape_to_mask
from multiscale_features import FeatureJob
from sam_embed import bilinear_sample_emb, fit_pca, transform_pca

logger = logging.getLogger(__name__)

_model_cache: TTLCache = TTLCache(ttl_seconds=1800.0, max_entries=16)
_pred_cache: TTLCache = TTLCache(ttl_seconds=600.0, max_entries=32)
_SAM_PCA_DIMS = 32

# status_map values
STATUS_ABSTAIN = 0
STATUS_SINGLETON = 1
STATUS_MULTI = 2


@dataclass(frozen=True)
class ClfModel:
    """Cached CatBoost model tied to a feature job (+ conformal calibration)."""

    model_id: str
    feature_job_id: str
    class_ids: tuple[int, ...]
    n_samples: int
    n_train: int
    n_cal: int
    model: CatBoostClassifier
    feature_labels: tuple[str, ...]
    train_accuracy: float
    feature_importances: tuple[tuple[str, float], ...]
    params: dict[str, Any]
    n_trees: int
    # Sorted nonconformity scores per class (calibration), for any-alpha quantiles
    cal_scores_by_class: dict[int, np.ndarray]
    _model_json: dict[str, Any] = field(repr=False, compare=False)
    sam_pca_mean: np.ndarray | None = None
    sam_pca_components: np.ndarray | None = None
    uses_sam: bool = False


@dataclass(frozen=True)
class ConformalPredictResult:
    """Commit map + set status + meta for one predict call."""

    commit_map: np.ndarray  # HxW uint8 classId or 0
    status_map: np.ndarray  # HxW uint8 abstain/singleton/multi
    alpha: float
    q_by_class: dict[int, float]
    counts: dict[str, int]
    class_ids: tuple[int, ...]


@dataclass(frozen=True)
class CachedPrediction:
    """Server-side predict artifacts for PNG GETs."""

    pred_id: str
    commit_png: bytes
    status_png: bytes
    meta: dict[str, Any]


def build_label_map(shapes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Compose a sparse int label map from shapes (0 = unlabeled). Last shape wins."""
    labels = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        class_id = int(shape.get("classId") or shape.get("class_id") or 0)
        if class_id <= 0 or class_id > 255:
            continue
        mask = shape_to_mask(shape, height, width)
        labels[mask] = class_id
    return labels


def stratified_train_cal_split(
    ys: np.ndarray,
    *,
    train_frac: float = 0.8,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified indices into train / calibration (disjoint, cover all)."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    train_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    for cls in np.unique(ys):
        idx = np.flatnonzero(ys == cls)
        rng.shuffle(idx)
        n = idx.size
        # Ensure at least one cal sample when class has >=2 pixels
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
    ys: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices into *ys* capped at *max_samples*, roughly per-class balanced."""
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


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity scores.

    Uses level ``ceil((n+1)(1-alpha))/n`` clipped to [1/n, 1].
    """
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        return 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    # Standard: q_level = ceil((n+1)(1-alpha)) / n
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def mondrian_thresholds(
    cal_scores_by_class: dict[int, np.ndarray],
    alpha: float,
) -> dict[int, float]:
    """Per-class conformal thresholds q_y(alpha)."""
    return {int(c): conformal_quantile(scores, alpha) for c, scores in cal_scores_by_class.items()}


def _dump_model_json(clf: CatBoostClassifier) -> dict[str, Any]:
    """Serialize model to CatBoost JSON (oblivious trees)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.json"
        clf.save_model(str(path), format="json")
        return json.loads(path.read_text(encoding="utf-8"))


def tree_view(
    model: ClfModel,
    tree_index: int = 0,
) -> dict[str, Any]:
    """Human-readable view of one oblivious tree (splits + leaf values)."""
    trees = model._model_json.get("oblivious_trees") or []
    n = len(trees)
    if n == 0:
        raise ValueError("model has no trees")
    if tree_index < 0 or tree_index >= n:
        raise IndexError(f"tree_index {tree_index} out of range 0..{n - 1}")
    tree = trees[tree_index]
    splits_out: list[dict[str, Any]] = []
    for sp in tree.get("splits") or []:
        fi = int(sp.get("float_feature_index", 0))
        label = (
            model.feature_labels[fi]
            if 0 <= fi < len(model.feature_labels)
            else f"feature_{fi}"
        )
        splits_out.append(
            {
                "feature_index": fi,
                "feature_label": label,
                "threshold": float(sp.get("border", 0.0)),
                "split_type": sp.get("split_type", "FloatFeature"),
            }
        )
    leaf_values = [float(v) for v in (tree.get("leaf_values") or [])]
    return {
        "tree_index": tree_index,
        "n_trees": n,
        "depth": len(splits_out),
        "splits": splits_out,
        "n_leaves": len(leaf_values),
        "leaf_values": leaf_values[:64],
        "leaf_values_truncated": len(leaf_values) > 64,
    }


def _job_has_sam(job: FeatureJob) -> bool:
    return (
        job.sam_emb is not None
        and job.sam_orig_hw is not None
        and job.sam_reshaped_hw is not None
    )


def _sample_sk_and_sam(
    job: FeatureJob,
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    sam_pca_mean: np.ndarray | None = None,
    sam_pca_components: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build skimage rows and optional raw or PCA-projected SAM rows."""
    h, w, c = job.float_stack.shape
    ys_i = np.clip(ys.astype(np.int64), 0, h - 1)
    xs_i = np.clip(xs.astype(np.int64), 0, w - 1)
    x_sk = job.float_stack[ys_i, xs_i].astype(np.float32)
    if not _job_has_sam(job):
        return x_sk, None
    assert job.sam_emb is not None and job.sam_orig_hw and job.sam_reshaped_hw
    oh, ow = job.sam_orig_hw
    rh, rw = job.sam_reshaped_hw
    x_sam = bilinear_sample_emb(
        job.sam_emb, ys_i.astype(np.float64), xs_i.astype(np.float64),
        orig_h=oh, orig_w=ow, reshaped_h=rh, reshaped_w=rw,
    )
    if sam_pca_mean is not None and sam_pca_components is not None:
        x_sam = transform_pca(x_sam, sam_pca_mean, sam_pca_components)
    return x_sk, x_sam


def _build_feature_matrix(
    job: FeatureJob,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    uses_sam: bool,
    sam_pca_mean: np.ndarray | None,
    sam_pca_components: np.ndarray | None,
    sam_pca_dims: int,
    fit_pca_now: bool,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray | None, np.ndarray | None]:
    """Assemble X and feature labels; optionally fit SAM PCA."""
    c = job.float_stack.shape[-1]
    # Always sample raw (no PCA inside); apply PCA once here.
    x_sk, x_sam_raw = _sample_sk_and_sam(job, yy, xx)
    pca_mean = sam_pca_mean
    pca_comp = sam_pca_components
    if uses_sam and x_sam_raw is not None:
        if fit_pca_now:
            pca_mean, pca_comp = fit_pca(x_sam_raw, n_components=sam_pca_dims)
        if pca_mean is not None and pca_comp is not None:
            x_sam = transform_pca(x_sam_raw, pca_mean, pca_comp)
        else:
            x_sam = x_sam_raw
        X = np.concatenate([x_sk, x_sam], axis=1)
        feat_labels = tuple(job.labels) + tuple(f"sam{i}" for i in range(x_sam.shape[1]))
    else:
        X = x_sk
        feat_labels = tuple(job.labels) if len(job.labels) == c else tuple(f"f{i}" for i in range(c))
    return X, feat_labels, pca_mean, pca_comp


def train_classifier(
    job: FeatureJob,
    shapes: list[dict[str, Any]],
    *,
    iterations: int = 200,
    depth: int = 6,
    learning_rate: float = 0.1,
    max_samples: int = 200_000,
    random_seed: int = 0,
    sam_pca_dims: int = _SAM_PCA_DIMS,
    train_frac: float = 0.8,
) -> ClfModel:
    """Fit CatBoost on 80% of labels; calibrate Mondrian scores on 20%."""
    h, w, c = job.float_stack.shape
    label_map = build_label_map(shapes, h, w)
    if not np.any(label_map > 0):
        raise ValueError("no labeled pixels — annotate at least one shape")

    yy, xx = np.nonzero(label_map)
    y_all = label_map[yy, xx].astype(np.int32)
    class_ids = tuple(int(x) for x in sorted(np.unique(y_all).tolist()))
    if len(class_ids) < 2:
        raise ValueError("need at least two classes with labeled pixels")

    rng = np.random.default_rng(random_seed)
    # Cap then split
    cap_idx = _stratified_sample(y_all, max_samples, rng)
    yy_c = yy[cap_idx]
    xx_c = xx[cap_idx]
    y_c = y_all[cap_idx]

    train_idx, cal_idx = stratified_train_cal_split(y_c, train_frac=train_frac, rng=rng)
    yy_tr, xx_tr, y_tr = yy_c[train_idx], xx_c[train_idx], y_c[train_idx]
    yy_cal, xx_cal, y_cal = yy_c[cal_idx], xx_c[cal_idx], y_c[cal_idx]

    uses_sam = _job_has_sam(job)
    X_tr, feat_labels, sam_pca_mean, sam_pca_components = _build_feature_matrix(
        job,
        yy_tr,
        xx_tr,
        uses_sam=uses_sam,
        sam_pca_mean=None,
        sam_pca_components=None,
        sam_pca_dims=sam_pca_dims,
        fit_pca_now=uses_sam,
    )

    params = {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "loss_function": "MultiClass",
        "random_seed": int(random_seed),
        "uses_sam": uses_sam,
        "sam_pca_dims": int(sam_pca_components.shape[0]) if sam_pca_components is not None else 0,
        "train_frac": float(train_frac),
    }
    clf = CatBoostClassifier(
        iterations=params["iterations"],
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        loss_function="MultiClass",
        verbose=False,
        allow_writing_files=False,
        random_seed=params["random_seed"],
        thread_count=-1,
    )
    clf.fit(X_tr, y_tr)

    pred_train = np.asarray(clf.predict(X_tr)).reshape(-1).astype(np.int32)
    train_accuracy = float(np.mean(pred_train == y_tr))

    # Calibration nonconformity: s = 1 - p_true
    X_cal, _, _, _ = _build_feature_matrix(
        job,
        yy_cal,
        xx_cal,
        uses_sam=uses_sam,
        sam_pca_mean=sam_pca_mean,
        sam_pca_components=sam_pca_components,
        sam_pca_dims=sam_pca_dims,
        fit_pca_now=False,
    )
    proba_cal = np.asarray(clf.predict_proba(X_cal), dtype=np.float64)
    classes = np.asarray(clf.classes_)
    class_to_col = {int(c): i for i, c in enumerate(classes.tolist())}
    cal_scores_by_class: dict[int, np.ndarray] = {}
    for cid in class_ids:
        mask = y_cal == cid
        if not np.any(mask):
            continue
        col = class_to_col[cid]
        scores = 1.0 - proba_cal[mask, col]
        cal_scores_by_class[cid] = np.sort(scores.astype(np.float64))

    if not cal_scores_by_class:
        raise ValueError("no calibration scores — need labeled pixels in every class for cal split")

    raw_imp = np.asarray(clf.get_feature_importance(), dtype=np.float64)
    if len(feat_labels) != raw_imp.size:
        feat_labels = tuple(f"f{i}" for i in range(raw_imp.size))
    order = np.argsort(-raw_imp)
    feature_importances = tuple(
        (feat_labels[int(i)], float(raw_imp[int(i)])) for i in order
    )

    model_json = _dump_model_json(clf)
    n_trees = len(model_json.get("oblivious_trees") or [])

    model_id = uuid.uuid4().hex
    entry = ClfModel(
        model_id=model_id,
        feature_job_id=job.job_id,
        class_ids=class_ids,
        n_samples=int(y_c.shape[0]),
        n_train=int(y_tr.shape[0]),
        n_cal=int(y_cal.shape[0]),
        model=clf,
        feature_labels=feat_labels,
        train_accuracy=train_accuracy,
        feature_importances=feature_importances,
        params=params,
        n_trees=n_trees,
        cal_scores_by_class=cal_scores_by_class,
        _model_json=model_json,
        sam_pca_mean=sam_pca_mean,
        sam_pca_components=sam_pca_components,
        uses_sam=uses_sam,
    )
    _model_cache.set(model_id, entry)
    return entry


def get_model(model_id: str) -> ClfModel | None:
    """Look up a cached classifier model."""
    m = _model_cache.get(model_id)
    return m if isinstance(m, ClfModel) else None


def store_prediction(result: ConformalPredictResult) -> CachedPrediction:
    """Cache commit/status PNGs; return handle for GET endpoints."""
    pred_id = uuid.uuid4().hex
    meta = {
        "pred_id": pred_id,
        "alpha": result.alpha,
        "class_ids": list(result.class_ids),
        "q_by_class": {str(k): v for k, v in result.q_by_class.items()},
        "counts": result.counts,
    }
    cached = CachedPrediction(
        pred_id=pred_id,
        commit_png=encode_label_png(result.commit_map),
        status_png=encode_label_png(result.status_map),
        meta=meta,
    )
    _pred_cache.set(pred_id, cached)
    return cached


def get_prediction(pred_id: str) -> CachedPrediction | None:
    """Look up a cached conformal prediction."""
    p = _pred_cache.get(pred_id)
    return p if isinstance(p, CachedPrediction) else None


def _features_for_coords(
    model: ClfModel,
    job: FeatureJob,
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    x_sk, x_sam = _sample_sk_and_sam(
        job,
        ys,
        xs,
        sam_pca_mean=model.sam_pca_mean,
        sam_pca_components=model.sam_pca_components,
    )
    if model.uses_sam:
        if x_sam is None:
            raise ValueError("model was trained with SlimSAM but job has no sam_emb")
        return np.concatenate([x_sk, x_sam], axis=1)
    return x_sk


def predict_conformal(
    model: ClfModel,
    job: FeatureJob,
    *,
    alpha: float = 0.05,
    row_chunk: int = 128,
    preserve_shapes: list[dict[str, Any]] | None = None,
) -> ConformalPredictResult:
    """Mondrian conformal sets → commit map (singletons) + status map."""
    shelf = model.feature_job_id.startswith("shelf:")
    if not shelf and job.job_id != model.feature_job_id:
        raise ValueError("model was trained on a different feature job")
    if model.uses_sam and not _job_has_sam(job):
        raise ValueError("model expects SlimSAM embeddings on the feature job")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    # Shelf models: sanity-check feature width via a 1-pixel probe
    if shelf and model.feature_labels:
        probe = _features_for_coords(
            model, job, np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
        )
        expected = len(model.feature_labels)
        if probe.shape[1] != expected:
            raise ValueError(
                f"feature width mismatch: job has {probe.shape[1]}, model expects {expected}"
            )

    q_by_class = mondrian_thresholds(model.cal_scores_by_class, alpha)
    # p_y >= 1 - q_y  ↔  included
    thr_prob = {cid: 1.0 - q for cid, q in q_by_class.items()}

    h, w, _ = job.float_stack.shape
    classes = np.asarray(model.model.classes_, dtype=np.int32)
    class_list = [int(c) for c in classes.tolist()]
    commit = np.zeros((h, w), dtype=np.uint8)
    status = np.zeros((h, w), dtype=np.uint8)

    for y0 in range(0, h, row_chunk):
        y1 = min(h, y0 + row_chunk)
        yy = np.repeat(np.arange(y0, y1), w)
        xx = np.tile(np.arange(w), y1 - y0)
        block = _features_for_coords(model, job, yy, xx)
        proba = np.asarray(model.model.predict_proba(block), dtype=np.float64)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, proba], axis=1)

        n = proba.shape[0]
        in_set = np.zeros((n, len(class_list)), dtype=bool)
        for j, cid in enumerate(class_list):
            t = thr_prob.get(cid)
            if t is None:
                continue
            in_set[:, j] = proba[:, j] >= t

        set_sizes = in_set.sum(axis=1)
        # Singleton: exactly one class; commit that classId
        singleton = set_sizes == 1
        multi = set_sizes > 1
        abstain = set_sizes == 0

        st = np.full(n, STATUS_ABSTAIN, dtype=np.uint8)
        st[singleton] = STATUS_SINGLETON
        st[multi] = STATUS_MULTI

        cm = np.zeros(n, dtype=np.uint8)
        if np.any(singleton):
            # argmax among members = the only True column
            cols = np.argmax(in_set[singleton], axis=1)
            cm[singleton] = classes[cols].astype(np.uint8)

        commit[y0:y1] = cm.reshape(y1 - y0, w)
        status[y0:y1] = st.reshape(y1 - y0, w)

    if preserve_shapes:
        preserve = build_label_map(preserve_shapes, h, w) > 0
        commit[preserve] = 0
        # Keep status for overlay elsewhere; still mark preserved as abstain-ish
        status[preserve] = STATUS_ABSTAIN

    counts = {
        "singleton": int(np.sum(status == STATUS_SINGLETON)),
        "multi": int(np.sum(status == STATUS_MULTI)),
        "abstain": int(np.sum(status == STATUS_ABSTAIN)),
    }
    return ConformalPredictResult(
        commit_map=commit,
        status_map=status,
        alpha=float(alpha),
        q_by_class=q_by_class,
        counts=counts,
        class_ids=model.class_ids,
    )


def predict_labels(
    model: ClfModel,
    job: FeatureJob,
    *,
    row_chunk: int = 128,
    preserve_shapes: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Backward-compat argmax map (no conformal). Prefer ``predict_conformal``."""
    if job.job_id != model.feature_job_id:
        raise ValueError("model was trained on a different feature job")
    if model.uses_sam and not _job_has_sam(job):
        raise ValueError("model expects SlimSAM embeddings on the feature job")
    h, w, _ = job.float_stack.shape
    classes = np.asarray(model.model.classes_, dtype=np.int32)
    out = np.empty((h, w), dtype=np.uint8)
    for y0 in range(0, h, row_chunk):
        y1 = min(h, y0 + row_chunk)
        yy = np.repeat(np.arange(y0, y1), w)
        xx = np.tile(np.arange(w), y1 - y0)
        block = _features_for_coords(model, job, yy, xx)
        proba = np.asarray(model.model.predict_proba(block), dtype=np.float64)
        if proba.ndim == 1:
            pred_idx = (proba >= 0.5).astype(np.int32)
        else:
            pred_idx = np.argmax(proba, axis=1).astype(np.int32)
        pred = classes[pred_idx]
        out[y0:y1] = pred.astype(np.uint8).reshape(y1 - y0, w)
    if preserve_shapes:
        preserve = build_label_map(preserve_shapes, h, w) > 0
        out[preserve] = 0
    return out


def encode_label_png(label_map: np.ndarray) -> bytes:
    """Encode HxW uint8 classId/status map as a grayscale PNG."""
    img = PILImage.fromarray(np.asarray(label_map, dtype=np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def train_result_dict(model: ClfModel) -> dict[str, Any]:
    """JSON payload for the train endpoint / UI results panel."""
    preview = tree_view(model, 0) if model.n_trees else None
    q_default = mondrian_thresholds(model.cal_scores_by_class, 0.05)
    return {
        "model_id": model.model_id,
        "n_samples": model.n_samples,
        "n_train": model.n_train,
        "n_cal": model.n_cal,
        "class_ids": list(model.class_ids),
        "train_accuracy": model.train_accuracy,
        "params": model.params,
        "n_trees": model.n_trees,
        "uses_sam": model.uses_sam,
        "q_by_class_alpha_05": {str(k): v for k, v in q_default.items()},
        "feature_importances": [
            {"label": lab, "importance": imp} for lab, imp in model.feature_importances
        ],
        "tree_preview": preview,
    }
