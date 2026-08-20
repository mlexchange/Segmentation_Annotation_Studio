"""Tests for baking a denoised volume into a new Tiled dataset.

The output has to be a first-class, openable dataset — not a hidden sidecar —
so these pin down the Browse-visible metadata contract as well as the job
mechanics (cancellation, per-slice error tolerance, refusing to overwrite).

Fake Tiled nodes follow the dict-backed pattern from ``test_tiled_mask_sync.py``
/ ``test_masks_from_tiled.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
import denoise_bake
import export_jobs
from annotation_server import app
from schemas import DenoiseBakeRequest

SIZE = 48


def _volume(n_slices: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = np.zeros((n_slices, SIZE, SIZE), dtype=np.float64)
    clean[:, SIZE // 4 : 3 * SIZE // 4, SIZE // 4 : 3 * SIZE // 4] = 3000.0
    return np.clip(clean + rng.normal(0, 250, clean.shape), 0, 65535).astype(np.uint16)


class _FakeContainer:
    """Records write_array / update_metadata calls."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.array_meta: dict[str, dict] = {}
        self.array_dims: dict[str, object] = {}
        self.metadata: dict = {}

    def write_array(self, data, *, key: str, dims=None, metadata=None) -> None:
        self.arrays[key] = np.asarray(data).copy()
        self.array_meta[key] = dict(metadata or {})
        self.array_dims[key] = dims

    def update_metadata(self, metadata: dict) -> None:
        self.metadata = dict(metadata)

    def keys(self):
        return list(self.arrays)


class _FakeClient:
    """Root client: ``existing`` maps already-present paths (for overwrite checks)."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = existing or set()

    def __getitem__(self, key: str):
        if key in self.existing:
            return object()
        raise KeyError(key)


@pytest.fixture
def baked(monkeypatch):
    """Wire a fake source volume + capture container, returning both."""
    volume = _volume()
    container = _FakeContainer()
    monkeypatch.setattr(denoise_bake.arrays_mod, "resolve_array", lambda *a, **k: volume)
    monkeypatch.setattr(denoise_bake, "get_tiled_client", lambda uri=None: _FakeClient())
    monkeypatch.setattr(denoise_bake.ingest_mod, "_ensure_container", lambda c, parts: container)
    return {"volume": volume, "container": container}


def _request(**over) -> DenoiseBakeRequest:
    base = {
        "source": "browse/dataset",
        "server_uri": "http://fake",
        "method": "gaussian",
        "strength": 0.5,
    }
    base.update(over)
    return DenoiseBakeRequest(**base)


def _run(request: DenoiseBakeRequest) -> dict:
    jid = export_jobs.new_job("")
    denoise_bake.run_denoise_bake_job(jid, request)
    return export_jobs.get_job(jid)


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

def test_default_target_is_a_sibling_of_the_source() -> None:
    """Landing next to the source is what makes it discoverable in Browse."""
    assert denoise_bake.default_target_path("browse/dataset") == "browse/dataset_denoised"
    assert denoise_bake.default_target_path("browse/expt/scan1") == "browse/expt/scan1_denoised"


def test_default_target_rejects_an_empty_source() -> None:
    with pytest.raises(ValueError):
        denoise_bake.default_target_path("   ")


# ---------------------------------------------------------------------------
# The happy path and the Browse metadata contract
# ---------------------------------------------------------------------------

def test_writes_one_array_per_source_slice(baked) -> None:
    job = _run(_request())

    assert job["state"] == "done"
    assert len(baked["container"].arrays) == baked["volume"].shape[0]
    assert job["result"]["n_slices"] == baked["volume"].shape[0]


def test_output_preserves_source_shape_and_dtype(baked) -> None:
    """The bake writes into a real Tiled array; a silent float64 promotion would
    quadruple storage and change how Browse renders it."""
    _run(_request())

    for key, arr in baked["container"].arrays.items():
        assert arr.shape == (SIZE, SIZE), key
        assert arr.dtype == baked["volume"].dtype, key


def test_output_is_actually_denoised_not_a_copy(baked) -> None:
    _run(_request(method="tv", strength=0.9))

    first = baked["container"].arrays["slice_0000"]
    assert not np.array_equal(first, baked["volume"][0])


def test_container_metadata_carries_what_browse_needs(baked) -> None:
    """``sample_name`` + ``n_images`` are what ``_describe_sample`` and the
    facet builder read. ``n_images`` must match what was really written."""
    _run(_request(description="denoised, grains"))

    meta = baked["container"].metadata
    assert meta["sample_name"] == "dataset_denoised"
    assert meta["n_images"] == len(baked["container"].arrays)
    assert meta["keywords"] == ["denoised", "grains"]
    assert meta["description"] == "denoised, grains"


def test_container_metadata_records_reproducible_provenance(baked) -> None:
    _run(_request(method="tv", strength=0.25))

    meta = baked["container"].metadata
    assert meta["denoise_source"] == "browse/dataset"
    assert meta["denoise_method"] == "tv"
    assert meta["denoise_strength"] == pytest.approx(0.25)
    assert meta["denoise_created_at"]


def test_array_metadata_has_zero_padded_image_number_and_dims(baked) -> None:
    """The Tiled node KEY sorts lexically, so ``image_number`` is the only thing
    that makes numeric slice order recoverable — it has to be padded."""
    _run(_request())

    numbers = [baked["container"].array_meta[k]["image_number"] for k in sorted(baked["container"].arrays)]
    assert numbers == ["0000", "0001", "0002", "0003"]
    assert all(baked["container"].array_dims[k] == ["y", "x"] for k in baked["container"].arrays)
    assert baked["container"].array_meta["slice_0000"]["size"] == f"{SIZE} x {SIZE}"


def test_slice_keys_sort_lexically_into_numeric_order(baked) -> None:
    """Belt-and-braces on the ordering trap: the generated keys are themselves
    zero-padded, so even a consumer that ignores image_number gets it right."""
    _run(_request())

    keys = sorted(baked["container"].arrays)
    assert keys == [f"slice_{i:04d}" for i in range(baked["volume"].shape[0])]


# ---------------------------------------------------------------------------
# 3-D methods
# ---------------------------------------------------------------------------

def test_a_3d_method_bakes_the_whole_volume(baked) -> None:
    job = _run(_request(method="gaussian3d", strength=0.5))

    assert job["state"] == "done"
    assert len(baked["container"].arrays) == baked["volume"].shape[0]


def test_a_3d_bake_differs_from_its_2d_counterpart(monkeypatch) -> None:
    """Confirms the z-window is genuinely used during a bake, not just in the
    preview route."""
    volume = _volume()
    results = {}
    for method in ("gaussian", "gaussian3d"):
        container = _FakeContainer()
        monkeypatch.setattr(denoise_bake.arrays_mod, "resolve_array", lambda *a, **k: volume)
        monkeypatch.setattr(denoise_bake, "get_tiled_client", lambda uri=None: _FakeClient())
        monkeypatch.setattr(denoise_bake.ingest_mod, "_ensure_container", lambda c, p, _c=container: _c)
        _run(_request(method=method))
        results[method] = container.arrays["slice_0001"]

    assert not np.array_equal(results["gaussian"], results["gaussian3d"])


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_refuses_to_overwrite_an_existing_dataset(monkeypatch) -> None:
    """A half-overwritten volume mixing two filters is worse than a clear error."""
    volume = _volume()
    monkeypatch.setattr(denoise_bake.arrays_mod, "resolve_array", lambda *a, **k: volume)
    monkeypatch.setattr(
        denoise_bake, "get_tiled_client",
        lambda uri=None: _FakeClient(existing={"browse/dataset_denoised"}),
    )

    job = _run(_request())

    assert job["state"] == "error"
    assert "already exists" in job["error"]


def test_method_none_is_rejected(baked) -> None:
    job = _run(_request(method="none"))
    assert job["state"] == "error"
    assert "method" in job["error"].lower()


def test_an_unavailable_method_is_rejected_before_writing_anything(baked) -> None:
    import denoise as denoise_mod

    if "wavelet" in denoise_mod.available_methods():
        pytest.skip("PyWavelets installed, wavelet genuinely available")

    job = _run(_request(method="wavelet"))

    assert job["state"] == "error"
    assert baked["container"].arrays == {}


def test_a_target_outside_the_ingest_root_is_rejected(baked) -> None:
    """`validate_container_path` enforces this; the bake must surface it as a
    job error rather than writing somewhere unexpected."""
    job = _run(_request(target_path="somewhere/else"))

    assert job["state"] == "error"
    assert baked["container"].arrays == {}


def test_one_bad_slice_is_recorded_and_the_source_copied_through(monkeypatch) -> None:
    """Slice indices must stay 1:1 with the source. Skipping a failed slice
    outright would silently shift every later index in the output volume."""
    volume = _volume()
    container = _FakeContainer()
    monkeypatch.setattr(denoise_bake.arrays_mod, "resolve_array", lambda *a, **k: volume)
    monkeypatch.setattr(denoise_bake, "get_tiled_client", lambda uri=None: _FakeClient())
    monkeypatch.setattr(denoise_bake.ingest_mod, "_ensure_container", lambda c, p: container)

    real = denoise_bake.denoise_mod.denoise_slice

    def _flaky(arr, method, strength=0.5, extra=None):
        if getattr(_flaky, "calls", 0) == 2:
            _flaky.calls = getattr(_flaky, "calls", 0) + 1
            raise RuntimeError("filter blew up")
        _flaky.calls = getattr(_flaky, "calls", 0) + 1
        return real(arr, method, strength, extra)

    monkeypatch.setattr(denoise_bake.denoise_mod, "denoise_slice", _flaky)

    job = _run(_request())

    assert job["state"] == "done"
    assert len(container.arrays) == volume.shape[0]  # 1:1 preserved
    assert len(job["result"]["errors"]) == 1
    assert job["result"]["errors"][0]["slice"] == 2
    # The failed slice fell back to an unfiltered copy of the source.
    assert np.array_equal(container.arrays["slice_0002"], volume[2])


def test_cancellation_stops_early_keeps_what_was_written_and_says_so(monkeypatch, baked) -> None:
    calls = {"n": 0}

    def _cancel_after_two(jid):
        calls["n"] += 1
        return calls["n"] > 2

    monkeypatch.setattr(denoise_bake.export_jobs, "cancel_requested", _cancel_after_two)

    job = _run(_request())

    assert job["state"] == "done"
    assert job["result"]["cancelled"] is True
    assert 0 < len(baked["container"].arrays) < baked["volume"].shape[0]
    # n_images must describe what was ACTUALLY written, not the source depth.
    assert baked["container"].metadata["n_images"] == len(baked["container"].arrays)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

async def _post(**body) -> tuple[int, dict]:
    payload = {"source": "browse/dataset", "method": "gaussian", "strength": 0.5, **body}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/denoise/bake", json=payload)
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, {}


@pytest.mark.asyncio
async def test_route_returns_a_job_id_and_registers_a_real_job(monkeypatch) -> None:
    monkeypatch.setattr(annotation_server, "_require_tiled_server", lambda uri: "http://fake")
    monkeypatch.setattr(
        annotation_server.denoise_bake_mod, "run_denoise_bake_job", lambda jid, req: None
    )

    status, body = await _post()

    assert status == 200
    assert export_jobs.get_job(body["job_id"]) is not None


@pytest.mark.asyncio
async def test_route_rejects_method_none_and_unknown_methods(monkeypatch) -> None:
    monkeypatch.setattr(annotation_server, "_require_tiled_server", lambda uri: "http://fake")

    assert (await _post(method="none"))[0] == 422
    assert (await _post(method="not-a-filter"))[0] == 422


@pytest.mark.asyncio
async def test_route_rejects_out_of_range_strength(monkeypatch) -> None:
    monkeypatch.setattr(annotation_server, "_require_tiled_server", lambda uri: "http://fake")
    assert (await _post(strength=2.0))[0] == 422


@pytest.mark.asyncio
async def test_route_requires_a_configured_tiled_server(monkeypatch) -> None:
    def _reject(uri):
        raise annotation_server.HTTPException(403, "Tiled server is not configured")

    monkeypatch.setattr(annotation_server, "_require_tiled_server", _reject)

    assert (await _post())[0] == 403


def test_openapi_registers_the_bake_and_auto_routes() -> None:
    paths = app.openapi()["paths"]
    assert "post" in paths["/api/denoise/bake"]
    assert "get" in paths["/api/denoise/auto"]
