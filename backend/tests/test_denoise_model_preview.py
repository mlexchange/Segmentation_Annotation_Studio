"""``GET /api/image/slice?denoise_method=model`` — previewing a trained denoiser.

This path is deliberately unlike the classical-filter one: it applies a saved
Noise2Noise/Noise2Void run, so it takes ``ML_LOCK`` (GPU work) and its output is
already display-mapped rather than raw. These tests pin the guards that make it
fail *comprehensibly* — the failure modes here are all "you pointed it at the
wrong thing", and a 500 would tell the user nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
from annotation_server import app


def _volume(n: int = 3, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    clean = np.zeros((n, size, size), dtype=np.float64)
    clean[:, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 3000.0
    return np.clip(clean + rng.normal(0, 200, clean.shape), 0, 65535).astype(np.uint16)


@pytest.fixture
def fake_source(monkeypatch):
    monkeypatch.setattr(
        annotation_server.arrays_mod, "resolve_array", lambda *a, **k: _volume()
    )
    annotation_server._slice_cache.clear()
    yield
    annotation_server._slice_cache.clear()


async def _get(**params) -> tuple[int, bytes]:
    query = {"source": "browse/ds", "kind": "local", "slice_index": 0, **params}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/image/slice", params=query)
    return r.status_code, r.content


@pytest.mark.asyncio
async def test_model_without_a_run_id_is_a_422_not_a_500(fake_source) -> None:
    status, body = await _get(denoise_method="model")

    assert status == 422
    assert b"denoise_run_id" in body


@pytest.mark.asyncio
async def test_pointing_at_a_segmentation_run_explains_the_mismatch(fake_source, monkeypatch) -> None:
    """`/api/train/runs` lists every run regardless of task, so selecting a
    segmentation run here is an easy mistake — it must say so rather than fail
    somewhere deep in the forward pass on a channel-count mismatch."""
    import train_common

    monkeypatch.setattr(
        train_common, "load_run_config",
        lambda run_id: {"model_family": "dlsia_tunet", "task": "segmentation", "image_size": 512},
    )

    status, body = await _get(denoise_method="model", denoise_run_id="seg-run")

    assert status == 422
    assert b"not a denoiser" in body


@pytest.mark.asyncio
async def test_a_denoiser_run_is_accepted_past_the_family_guard(fake_source, monkeypatch) -> None:
    """Complements the rejection test: prove the guard keys on the run's family
    and task, and lets a real denoiser through to the (dependency-gated) work."""
    import train_common

    monkeypatch.setattr(
        train_common, "load_run_config",
        lambda run_id: {"model_family": "dlsia_denoiser", "task": "denoising",
                        "image_size": 32, "render": {}},
    )
    monkeypatch.setattr(train_common, "dlsia_available", lambda: False)

    status, body = await _get(denoise_method="model", denoise_run_id="denoise-run")

    # Past the "wrong family" guard, stopped by the dependency guard instead.
    assert status == 503
    assert b"dlsia" in body


@pytest.mark.asyncio
async def test_a_busy_device_reports_409_rather_than_queueing(fake_source, monkeypatch) -> None:
    """A preview must not block behind a multi-minute training run holding the
    lock — the UI shows this as "device busy", which is actionable."""
    import train_common

    monkeypatch.setattr(
        train_common, "load_run_config",
        lambda run_id: {"model_family": "dlsia_denoiser", "task": "denoising",
                        "image_size": 32, "render": {}},
    )
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")

    import tiling
    monkeypatch.setattr(tiling, "qlty_available", lambda: True)

    train_common.ML_LOCK.acquire()
    try:
        status, body = await _get(denoise_method="model", denoise_run_id="denoise-run")
    finally:
        train_common.ML_LOCK.release()

    assert status == 409
    assert b"busy" in body.lower()


@pytest.mark.asyncio
async def test_the_lock_is_released_when_inference_raises(fake_source, monkeypatch) -> None:
    """The real leak risk: an exception AFTER the lock is taken. If it escaped
    without releasing, every later preview and training job would deadlock on a
    permanently-held lock."""
    import train_common

    monkeypatch.setattr(
        train_common, "load_run_config",
        lambda run_id: {"model_family": "dlsia_denoiser", "task": "denoising",
                        "image_size": 32, "render": {}},
    )
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")

    def _boom(run_id):
        raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(train_common, "load_adapter_state", _boom)

    import tiling
    monkeypatch.setattr(tiling, "qlty_available", lambda: True)

    status, _ = await _get(denoise_method="model", denoise_run_id="denoise-run")

    assert status == 500
    assert not train_common.ML_LOCK.locked(), "ML_LOCK leaked after a failed inference"


@pytest.mark.asyncio
async def test_classical_methods_still_ignore_denoise_run_id(fake_source) -> None:
    """A stray run id on a classical request must not change its behaviour."""
    _, plain = await _get(denoise_method="gaussian", denoise_strength=0.5)
    status, with_run = await _get(
        denoise_method="gaussian", denoise_strength=0.5, denoise_run_id="ignored"
    )

    assert status == 200
    assert plain == with_run
