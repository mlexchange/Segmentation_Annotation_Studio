"""HTTP-boundary tests for the denoise parameters on ``GET /api/image/slice``.

Denoising is applied in this route rather than inside ``images.render_slice``
on purpose (see the route's docstring): ``render_slice`` early-returns for RGB
inputs, and ``volumes.build_volume`` normalizes independently, so a hook in the
shared helper would silently skip colour sources and desynchronize the 3D tab.
These tests pin that placement down, plus the caching and the z-window handling
the 3-D filters need.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage

import annotation_server
from annotation_server import app

SIZE = 96


def _volume(n_slices: int = 5, seed: int = 0) -> np.ndarray:
    """(n, H, W) uint16 step-edge phantom with independent per-slice noise."""
    rng = np.random.default_rng(seed)
    clean = np.zeros((n_slices, SIZE, SIZE), dtype=np.float64)
    clean[:, SIZE // 4 : 3 * SIZE // 4, SIZE // 4 : 3 * SIZE // 4] = 3000.0
    return np.clip(clean + rng.normal(0, 250, clean.shape), 0, 65535).astype(np.uint16)


@pytest.fixture
def fake_source(monkeypatch):
    """Serve a NumPy volume as a local source, counting slice reads.

    Returns the read-counter dict so tests can prove the cache is doing its job.
    """
    volume = _volume()
    counter = {"reads": 0}
    real_read_slice = annotation_server.arrays_mod.read_slice

    def _counting_read_slice(node, meta, idx):
        counter["reads"] += 1
        return real_read_slice(node, meta, idx)

    monkeypatch.setattr(
        annotation_server.arrays_mod, "resolve_array", lambda *a, **k: volume
    )
    monkeypatch.setattr(annotation_server.arrays_mod, "read_slice", _counting_read_slice)
    annotation_server._slice_cache.clear()
    yield {"volume": volume, "counter": counter}
    annotation_server._slice_cache.clear()


async def _get_slice(**params) -> tuple[int, bytes, dict]:
    query = {"source": "vol", "kind": "local", **params}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/image/slice", params=query)
    return response.status_code, response.content, dict(response.headers)


def _decode(png: bytes) -> np.ndarray:
    return np.asarray(PILImage.open(io.BytesIO(png)).convert("L"), dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_denoise_none_is_the_pre_existing_behaviour(fake_source) -> None:
    """The default must be byte-identical to a request that omits the params
    entirely — denoising is strictly opt-in."""
    status_a, png_a, _ = await _get_slice(slice_index=2)
    status_b, png_b, _ = await _get_slice(slice_index=2, denoise_method="none")

    assert status_a == status_b == 200
    assert png_a == png_b


@pytest.mark.asyncio
async def test_denoising_actually_changes_the_rendered_png(fake_source) -> None:
    _, raw_png, _ = await _get_slice(slice_index=2)
    status, denoised_png, _ = await _get_slice(
        slice_index=2, denoise_method="tv", denoise_strength=0.8
    )

    assert status == 200
    assert denoised_png != raw_png


@pytest.mark.asyncio
async def test_denoised_slice_is_smoother_than_the_raw_slice(fake_source) -> None:
    """Not just "different" — measurably less pixel-to-pixel variation inside a
    region that is uniform in the underlying phantom."""
    _, raw_png, _ = await _get_slice(slice_index=2)
    _, den_png, _ = await _get_slice(slice_index=2, denoise_method="tv", denoise_strength=0.8)

    interior = (slice(30, 66), slice(30, 66))
    raw_var = float(np.std(np.diff(_decode(raw_png)[interior], axis=1)))
    den_var = float(np.std(np.diff(_decode(den_png)[interior], axis=1)))

    assert den_var < raw_var


@pytest.mark.asyncio
async def test_cache_control_header_is_preserved(fake_source) -> None:
    _, _, headers = await _get_slice(slice_index=1, denoise_method="gaussian")
    assert headers["cache-control"] == "private, max-age=300"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_denoise_method_is_rejected_with_422(fake_source) -> None:
    status, _, _ = await _get_slice(slice_index=0, denoise_method="not-a-filter")
    assert status == 422


@pytest.mark.asyncio
async def test_out_of_range_strength_is_rejected_by_pydantic(fake_source) -> None:
    for bad in (-0.5, 1.5):
        status, _, _ = await _get_slice(
            slice_index=0, denoise_method="gaussian", denoise_strength=bad
        )
        assert status == 422, bad


@pytest.mark.asyncio
async def test_a_method_missing_its_optional_dependency_is_rejected_not_500(fake_source) -> None:
    """``denoise_wavelet`` imports fine without PyWavelets but raises at call
    time, so an unavailable method has to be caught up front and reported as a
    client error rather than surfacing as a 500."""
    import denoise as denoise_mod

    if "wavelet" in denoise_mod.available_methods():
        pytest.skip("PyWavelets is installed here, so wavelet is genuinely available")

    status, body, _ = await _get_slice(slice_index=0, denoise_method="wavelet")

    assert status == 422
    assert b"unavailable" in body.lower() or b"wavelet" in body.lower()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_identical_denoise_requests_are_served_from_cache(fake_source) -> None:
    """This route had no server-side cache before; without one, every revisit or
    slider tweak re-pays the filter cost (measured: seconds for NLM/TV at 3232²)."""
    await _get_slice(slice_index=2, denoise_method="tv", denoise_strength=0.5)
    reads_after_first = fake_source["counter"]["reads"]
    assert reads_after_first > 0

    await _get_slice(slice_index=2, denoise_method="tv", denoise_strength=0.5)

    assert fake_source["counter"]["reads"] == reads_after_first


@pytest.mark.asyncio
async def test_changing_any_denoise_parameter_misses_the_cache(fake_source) -> None:
    await _get_slice(slice_index=2, denoise_method="tv", denoise_strength=0.5)
    baseline = fake_source["counter"]["reads"]

    await _get_slice(slice_index=2, denoise_method="tv", denoise_strength=0.9)
    assert fake_source["counter"]["reads"] > baseline

    bumped = fake_source["counter"]["reads"]
    await _get_slice(slice_index=2, denoise_method="gaussian", denoise_strength=0.5)
    assert fake_source["counter"]["reads"] > bumped


@pytest.mark.asyncio
async def test_undenoised_requests_are_not_cached(fake_source) -> None:
    """The un-denoised path is already cheap and was uncached before; keep it
    that way so the cache's few large entries stay reserved for denoised PNGs."""
    await _get_slice(slice_index=2)
    first = fake_source["counter"]["reads"]

    await _get_slice(slice_index=2)

    assert fake_source["counter"]["reads"] > first


# ---------------------------------------------------------------------------
# 3-D methods and the z-window
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_3d_method_reads_neighbouring_slices(fake_source) -> None:
    """The 3-D filters exist to exploit slice-to-slice correlation, so they must
    genuinely pull a z-window rather than degenerating to a per-slice filter."""
    await _get_slice(slice_index=2, denoise_method="gaussian3d")
    window_reads = fake_source["counter"]["reads"]

    annotation_server._slice_cache.clear()
    fake_source["counter"]["reads"] = 0
    await _get_slice(slice_index=2, denoise_method="gaussian", denoise_strength=0.5)
    single_reads = fake_source["counter"]["reads"]

    assert window_reads > single_reads


@pytest.mark.asyncio
async def test_a_3d_method_at_the_volume_edge_clamps_instead_of_failing(fake_source) -> None:
    """Slice 0 has no lower neighbours; the window must clamp rather than read
    a negative index (which STACK sources interpret as counting from the end)."""
    for idx in (0, 4):  # first and last slice of the 5-slice fake volume
        status, png, _ = await _get_slice(slice_index=idx, denoise_method="gaussian3d")
        assert status == 200, idx
        assert len(png) > 0


@pytest.mark.asyncio
async def test_a_3d_method_on_a_single_slice_source_falls_back_to_2d(monkeypatch) -> None:
    """A 2-D source has no z-context at all. Falling back to the 2-D sibling is
    friendlier than erroring out on a method the UI legitimately offered."""
    single = _volume(n_slices=1)[0]  # (H, W) — shape_kind HW
    monkeypatch.setattr(
        annotation_server.arrays_mod, "resolve_array", lambda *a, **k: single
    )
    annotation_server._slice_cache.clear()

    status, png, _ = await _get_slice(slice_index=0, denoise_method="median3d")

    assert status == 200
    assert len(png) > 0
    annotation_server._slice_cache.clear()


# ---------------------------------------------------------------------------
# Crop preview
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crop_returns_only_the_cropped_region(fake_source) -> None:
    """Crop-before-filter is what makes slow filters interactive (measured at
    3232²: bilateral 7.5s full-slice vs 0.71s cropped)."""
    status, png, _ = await _get_slice(
        slice_index=2, denoise_method="tv", denoise_crop=48
    )

    assert status == 200
    assert _decode(png).shape == (48, 48)


@pytest.mark.asyncio
async def test_crop_is_centred(fake_source) -> None:
    """The phantom's bright square is centred, so a small centred crop should be
    entirely inside it — proving the crop isn't taken from a corner."""
    _, png, _ = await _get_slice(slice_index=2, denoise_method="gaussian", denoise_crop=24)

    assert float(_decode(png).mean()) > 128.0


@pytest.mark.asyncio
async def test_crop_larger_than_the_slice_returns_the_whole_slice(fake_source) -> None:
    _, png, _ = await _get_slice(
        slice_index=2, denoise_method="gaussian", denoise_crop=2048
    )
    assert _decode(png).shape == (SIZE, SIZE)


@pytest.mark.asyncio
async def test_crop_brightness_matches_the_full_slice_not_the_crop_contents(fake_source) -> None:
    """Global normalization must keep sampling the WHOLE volume even for a crop,
    or the preview would auto-level to the crop's own range and look nothing
    like the main canvas."""
    _, full_png, _ = await _get_slice(slice_index=2, denoise_method="gaussian")
    _, crop_png, _ = await _get_slice(
        slice_index=2, denoise_method="gaussian", denoise_crop=24
    )

    full = _decode(full_png)
    centre = full[SIZE // 2 - 12 : SIZE // 2 + 12, SIZE // 2 - 12 : SIZE // 2 + 12]

    # Same underlying pixels, same normalization ⇒ closely matching brightness.
    assert abs(float(_decode(crop_png).mean()) - float(centre.mean())) < 12.0


# ---------------------------------------------------------------------------
# Capability reporting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capability_advertises_denoise_methods() -> None:
    """The UI builds its method menu from this, and must not offer a method the
    server cannot actually run."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/train/capability")

    assert response.status_code == 200
    denoise_info = response.json()["denoise"]
    assert denoise_info["available"] is True
    methods = {m["method"]: m for m in denoise_info["methods"]}
    assert "tv" in methods and methods["tv"]["available"] is True
    assert methods["nlm"]["cost"] == "slow"
    assert methods["gaussian3d"]["z_radius"] > 0
