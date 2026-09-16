"""CatBoost TrainerPlugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier

from ipred.trainers.base import ModelArtifacts


class CatBoostTrainer:
    """CatBoost multiclass pixel trainer with Mondrian cal scores."""

    id = "catboost"

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
        *,
        feature_labels: list[str],
        config: dict[str, Any],
    ) -> ModelArtifacts:
        """Fit CatBoost and collect per-class nonconformity scores."""
        iterations = int(config.get("iterations", 200))
        depth = int(config.get("depth", 6))
        learning_rate = float(config.get("learning_rate", 0.1))
        random_seed = int(config.get("random_seed", 0))
        params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "loss_function": "MultiClass",
            "random_seed": random_seed,
        }
        clf = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_seed,
            thread_count=-1,
        )
        clf.fit(x_train, y_train)
        pred = np.asarray(clf.predict(x_train)).reshape(-1).astype(np.int32)
        train_accuracy = float(np.mean(pred == y_train))

        proba_cal = np.asarray(clf.predict_proba(x_cal), dtype=np.float64)
        classes = [int(c) for c in np.asarray(clf.classes_).tolist()]
        class_to_col = {c: i for i, c in enumerate(classes)}
        cal_scores: dict[int, list[float]] = {}
        for cid in classes:
            mask = y_cal == cid
            if not np.any(mask):
                continue
            col = class_to_col[cid]
            scores = 1.0 - proba_cal[mask, col]
            cal_scores[cid] = np.sort(scores.astype(np.float64)).tolist()
        if not cal_scores:
            raise ValueError("no calibration scores")

        labels = list(feature_labels)
        raw_imp = np.asarray(clf.get_feature_importance(), dtype=np.float64)
        if len(labels) != raw_imp.size:
            labels = [f"f{i}" for i in range(raw_imp.size)]
        order = np.argsort(-raw_imp)
        feature_importances = [
            {"label": labels[int(i)], "importance": float(raw_imp[int(i)])} for i in order
        ]

        return ModelArtifacts(
            class_ids=classes,
            feature_labels=labels,
            cal_scores_by_class=cal_scores,
            train_accuracy=train_accuracy,
            n_train=int(y_train.shape[0]),
            n_cal=int(y_cal.shape[0]),
            n_samples=int(y_train.shape[0] + y_cal.shape[0]),
            params=params,
            extras={"feature_importances": feature_importances},
            model_handle=clf,
        )

    def predict_proba(self, model_handle: Any, x: np.ndarray) -> np.ndarray:
        """Return NxK float64 probabilities."""
        proba = np.asarray(model_handle.predict_proba(x), dtype=np.float64)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, proba], axis=1)
        return proba

    def save(self, model_handle: Any, dest_dir: Path) -> None:
        """Write ``model.cbm``."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        model_handle.save_model(str(dest_dir / "model.cbm"))

    def load(self, dest_dir: Path) -> Any:
        """Load ``model.cbm``."""
        clf = CatBoostClassifier()
        clf.load_model(str(dest_dir / "model.cbm"))
        return clf
