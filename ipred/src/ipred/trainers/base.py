"""Trainer plugin interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class ModelArtifacts:
    """Artifacts produced by ``train``."""

    class_ids: list[int]
    feature_labels: list[str]
    cal_scores_by_class: dict[int, list[float]]
    train_accuracy: float
    n_train: int
    n_cal: int
    n_samples: int
    params: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)
    # In-memory handle for immediate predict_proba (plugin-specific)
    model_handle: Any = None


class TrainerPlugin(Protocol):
    """Train / predict_proba contract for pixel classifiers."""

    id: str

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
        """Fit model and compute Mondrian calibration scores."""

    def predict_proba(self, model_handle: Any, x: np.ndarray) -> np.ndarray:
        """Return NxK probabilities aligned with training class order."""

    def save(self, model_handle: Any, dest_dir: Path) -> None:
        """Persist model weights into *dest_dir*."""

    def load(self, dest_dir: Path) -> Any:
        """Load model weights from *dest_dir*."""
