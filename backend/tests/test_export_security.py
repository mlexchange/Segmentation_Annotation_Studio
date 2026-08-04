"""HTTP-boundary tests for export/import filesystem containment."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from annotation_server import app


def _minimal_coco(directory: Path) -> None:
    """Create the smallest valid COCO document accepted by the importer."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_annotations.coco.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []})
    )


def test_coco_import_rejects_absolute_directory_outside_export_root(
    tmp_path: Path, monkeypatch
) -> None:
    """The import endpoint must not become an arbitrary host-file reader."""
    export_root = tmp_path / "exports"
    outside = tmp_path / "private"
    _minimal_coco(outside)
    monkeypatch.setenv("EXPORT_ROOT", str(export_root))

    response = TestClient(app).post(
        "/api/import/coco", params={"dataset_dir": str(outside)}
    )

    assert response.status_code == 403
    assert str(outside) not in response.text


def test_coco_import_accepts_directory_beneath_export_root(
    tmp_path: Path, monkeypatch
) -> None:
    """Existing local export/import workflows remain available inside the root."""
    export_root = tmp_path / "exports"
    dataset = export_root / "dataset" / "train"
    _minimal_coco(dataset)
    monkeypatch.setenv("EXPORT_ROOT", str(export_root))

    response = TestClient(app).post(
        "/api/import/coco", params={"dataset_dir": str(dataset)}
    )

    assert response.status_code == 200
    assert response.json()["classes"] == []


def test_coco_import_rejects_symlink_escape(tmp_path: Path, monkeypatch) -> None:
    """Resolving a symlink beneath EXPORT_ROOT must not escape containment."""
    export_root = tmp_path / "exports"
    outside = tmp_path / "outside"
    _minimal_coco(outside)
    export_root.mkdir()
    link = export_root / "linked"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("EXPORT_ROOT", str(export_root))

    response = TestClient(app).post(
        "/api/import/coco", params={"dataset_dir": str(link)}
    )

    assert response.status_code == 403
