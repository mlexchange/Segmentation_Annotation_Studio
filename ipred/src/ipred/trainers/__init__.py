"""Trainer plugin registry."""

from __future__ import annotations

from ipred.trainers.base import ModelArtifacts, TrainerPlugin
from ipred.trainers.catboost_trainer import CatBoostTrainer

_REGISTRY: dict[str, TrainerPlugin] = {
    CatBoostTrainer.id: CatBoostTrainer(),
}


def get_trainer(trainer_id: str) -> TrainerPlugin:
    """Return a registered trainer or raise KeyError."""
    try:
        return _REGISTRY[trainer_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown trainer {trainer_id!r}; available: {sorted(_REGISTRY)}"
        ) from exc


def list_trainers() -> list[str]:
    """Return registered trainer ids."""
    return sorted(_REGISTRY)
