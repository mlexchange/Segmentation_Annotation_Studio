"""Filesystem roots for ipred under LOCAL_DATA_ROOT."""

from __future__ import annotations

import os
from pathlib import Path


def local_data_root() -> Path:
    """Return resolved ``LOCAL_DATA_ROOT`` (default ``~/data``)."""
    return Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()


def engine_root() -> Path:
    """Return ``$LOCAL_DATA_ROOT/ipred``."""
    root = local_data_root() / "ipred"
    root.mkdir(parents=True, exist_ok=True)
    return root


def catalog_db_path() -> Path:
    """SQLite catalog path."""
    return engine_root() / "catalog.db"


def feature_models_root() -> Path:
    """Feature Setup shelf directory."""
    root = local_data_root() / ".feature_models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_blob_dir(project_id: str) -> Path:
    """Blob root for one project."""
    path = engine_root() / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path
