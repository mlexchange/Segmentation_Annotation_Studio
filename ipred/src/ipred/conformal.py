"""Mondrian split-conformal thresholds and maps from cached proba."""

from __future__ import annotations

from typing import Any

import numpy as np

STATUS_ABSTAIN = 0
STATUS_SINGLETON = 1
STATUS_MULTI = 2


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        return 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def mondrian_thresholds(
    cal_scores_by_class: dict[int, np.ndarray],
    alpha: float,
) -> dict[int, float]:
    """Per-class conformal thresholds q_y(alpha)."""
    return {
        int(c): conformal_quantile(scores, alpha)
        for c, scores in cal_scores_by_class.items()
    }


def maps_from_proba(
    proba: np.ndarray,
    class_ids: list[int],
    q_by_class: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build commit, status, and packed membership from HxWxK proba.

    Returns:
        commit (HxW uint8), status (HxW uint8), membership_bits (HxW uint32)
        where bit i means class_ids[i] is in the prediction set.
    """
    if proba.ndim != 3:
        raise ValueError("proba must be HxWxK")
    h, w, k = proba.shape
    if k != len(class_ids):
        raise ValueError("proba K must match class_ids length")
    thr_prob = {cid: 1.0 - q for cid, q in q_by_class.items()}
    in_set = np.zeros((h, w, k), dtype=bool)
    for j, cid in enumerate(class_ids):
        t = thr_prob.get(int(cid))
        if t is None:
            continue
        in_set[..., j] = proba[..., j] >= t

    set_sizes = in_set.sum(axis=2)
    status = np.full((h, w), STATUS_ABSTAIN, dtype=np.uint8)
    status[set_sizes == 1] = STATUS_SINGLETON
    status[set_sizes > 1] = STATUS_MULTI

    commit = np.zeros((h, w), dtype=np.uint8)
    singleton = set_sizes == 1
    if np.any(singleton):
        cols = np.argmax(in_set[singleton], axis=1)
        classes = np.asarray(class_ids, dtype=np.uint8)
        commit[singleton] = classes[cols]

    membership = np.zeros((h, w), dtype=np.uint32)
    for j in range(min(k, 32)):
        membership |= in_set[..., j].astype(np.uint32) << j

    return commit, status, membership


def counts_from_status(status: np.ndarray) -> dict[str, int]:
    """Count abstain / singleton / multi pixels."""
    return {
        "singleton": int(np.sum(status == STATUS_SINGLETON)),
        "multi": int(np.sum(status == STATUS_MULTI)),
        "abstain": int(np.sum(status == STATUS_ABSTAIN)),
    }


def cal_scores_from_json(raw: dict[str, Any]) -> dict[int, np.ndarray]:
    """Parse cal_scores.json into numpy arrays."""
    out: dict[int, np.ndarray] = {}
    for k, v in raw.items():
        out[int(k)] = np.asarray(v, dtype=np.float64)
    return out
