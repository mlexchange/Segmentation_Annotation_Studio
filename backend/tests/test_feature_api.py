"""API smoke tests for multiscale feature endpoints (in-process, no Tiled)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import multiscale_features as features_mod
from annotation_server import app


def test_get_feature_channel_roundtrip(monkeypatch) -> None:
    """POST is skipped; exercise GET against a pre-stored job."""
    gray = np.linspace(0, 1, 16 * 16, dtype=np.float64).reshape(16, 16)
    stack, float_stack, labels = features_mod.compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=False,
        texture=False,
        clahe=False,
    )
    job = features_mod.store_job(stack, labels, float_stack=float_stack)
    client = TestClient(app)
    res = client.get(f"/api/image/features/{job.job_id}/0")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/png")
    assert len(res.content) > 50

    missing = client.get("/api/image/features/does-not-exist/0")
    assert missing.status_code == 404
