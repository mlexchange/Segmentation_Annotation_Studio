"""Tests for ``GET /api/image/volume`` and its backing ``volumes.py`` module.

The 3D tab volume-renders the whole stack at once, so the server downsamples
to a small uint8 intensity grid instead of shipping native resolution. Three
things have to hold for that to work:

1. ``volumes.volume_dims`` is a shared contract with the frontend's own label
   rasterizer (``volumeDimsFor``/``gridFor``) — the two must agree on grid
   size voxel-for-voxel, or the segmentation overlay renders offset from the
   raw data. The parametrized table below pins that formula down exactly.
2. Every voxel is normalized with the same ``images.normalize_scalar_unit``
   the 2-D canvas (``render_slice``) uses, so a voxel's brightness matches
   what ``/api/image/slice`` shows at that position.
3. The route wires ``arrays_mod.resolve_array`` -> ``array_shape_meta`` ->
   (optionally) ``images_mod._sample_global_stats`` -> ``volumes_mod.build_volume``
   together correctly for both NHW/NHWC array nodes and STACK container nodes,
   guards the Tiled path behind ``_require_tiled_server``, and caches results
   in the module-level ``_volume_cache``.

Route-level tests follow ``test_reset_tiled.py``'s pattern (ASGI transport,
``@pytest.mark.asyncio``, monkeypatching ``annotation_server`` module
attributes directly). The STACK fake-node tests follow ``test_arrays_stack.py``'s
dict-backed fake node pattern. No real Tiled server or filesystem is touched;
``kind="local"`` tests monkeypatch ``annotation_server.arrays_mod.resolve_array``
to hand back an in-memory NumPy array directly instead of going through
``local_fs.open_array``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
import images
import volumes
from annotation_server import app


@pytest.fixture(autouse=True)
def _isolated_volume_cache():
    """Every test gets a clean module-level volume cache, both directions.

    Several tests below deliberately reuse simple ``source`` values like
    ``"x"``; without this they could read each other's cached payload/meta
    instead of exercising the code path the test claims to.
    """
    annotation_server._volume_cache.clear()
    yield
    annotation_server._volume_cache.clear()


# ---------------------------------------------------------------------------
# 1. The shared dims contract (pure function, no HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_slices", "height", "width", "max_dim", "sxy", "ny", "nx", "sz", "nz"),
    [
        (1, 512, 512, 256, 2, 256, 256, 1, 1),
        (700, 1000, 900, 256, 4, 250, 225, 3, 234),
        (700, 1000, 900, 384, 3, 333, 300, 2, 350),
        (100, 515, 100, 256, 3, 171, 33, 1, 100),
        (5, 10000, 10, 256, 40, 250, 1, 1, 5),
    ],
)
def test_volume_dims_matches_the_shared_frontend_contract_table(
    n_slices, height, width, max_dim, sxy, ny, nx, sz, nz
) -> None:
    result = volumes.volume_dims(n_slices, height, width, max_dim)

    assert result == {"sxy": sxy, "ny": ny, "nx": nx, "sz": sz, "nz": nz}


# ---------------------------------------------------------------------------
# 2. NHW numpy node round trip via the route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nhw_numpy_node_round_trips_through_the_route(monkeypatch) -> None:
    arr = (np.arange(6 * 40 * 50, dtype=np.uint16) % 4000).reshape(6, 40, 50)
    monkeypatch.setattr(annotation_server.arrays_mod, "resolve_array", lambda *a, **k: arr)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/image/volume", params={"source": "x", "kind": "local", "max_dim": 32}
        )

    assert response.status_code == 200
    vol_meta = json.loads(response.headers["x-volume-meta"])
    expected_dims = volumes.volume_dims(6, 40, 50, 32)
    assert {k: vol_meta[k] for k in expected_dims} == expected_dims
    assert len(response.content) == vol_meta["nz"] * vol_meta["ny"] * vol_meta["nx"]
    assert response.headers["cache-control"] == "private, max-age=300"


# ---------------------------------------------------------------------------
# 3. Normalization parity with render_slice (the 2D/3D brightness guarantee)
# ---------------------------------------------------------------------------


def test_normalize_scalar_unit_matches_render_slice_red_channel() -> None:
    """volumes.py normalizes with the exact function render_slice's grayscale
    branch uses, so a voxel's brightness matches the 2-D canvas.

    NOTE: ``render_slice`` quantizes to uint8 by truncating
    (``(data * 255).astype(np.uint8)``, no ``.round()``), while
    ``volumes._to_gray_u8`` quantizes by rounding
    (``(unit * 255.0).round().astype(np.uint8)``). The two therefore agree on
    the underlying ``[0, 1]`` unit value but can land on an adjacent integer
    after quantization — confirmed below to differ by at most 1. This is a
    real (minor) inconsistency between the 2-D and 3-D quantization paths;
    see the final report for this test file.
    """
    arr = np.array([[10, 40, 90, 160], [20, 60, 130, 250]], dtype=np.uint16)
    opts = {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"}
    global_range = (0.0, 250.0)

    unit = images.normalize_scalar_unit(arr, opts, global_range)
    rounded = (unit * 255.0).round().astype(np.uint8)
    rgb = images.render_slice(arr, opts, global_range)

    for y, x in [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)]:
        assert abs(int(rounded[y, x]) - int(rgb[y, x, 0])) <= 1


def test_build_volume_voxel_matches_hand_computed_normalize_scalar_unit() -> None:
    """With no downsampling (sxy == sz == 1) a build_volume voxel must equal
    round(normalize_scalar_unit(source_pixel) * 255) exactly — the same
    formula/rounding volumes._to_gray_u8 uses internally."""
    arr = np.array(
        [
            [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]],
            [[5, 15, 25, 35], [45, 55, 65, 75], [85, 95, 105, 115], [125, 135, 145, 155]],
        ],
        dtype=np.uint16,
    )
    meta = {"n_slices": 2, "height": 4, "width": 4, "dtype": "uint16", "is_rgb": False, "shape_kind": "NHW"}
    opts = {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0}
    global_range = (0.0, 160.0)

    payload, vol_meta = volumes.build_volume(arr, meta, opts, max_dim=4, global_range=global_range)
    assert vol_meta["sxy"] == 1 and vol_meta["sz"] == 1  # exact grid: no striding to reason through

    volume = np.frombuffer(payload, dtype=np.uint8).reshape(vol_meta["nz"], vol_meta["ny"], vol_meta["nx"])

    for z, y, x in [(0, 0, 0), (0, 3, 3), (1, 2, 1)]:
        expected_unit = images.normalize_scalar_unit(arr[z], opts, global_range)
        expected = int(np.round(expected_unit[y, x] * 255.0))
        assert int(volume[z, y, x]) == expected


# ---------------------------------------------------------------------------
# 4. STACK path (dict-backed fake node, following test_arrays_stack.py)
# ---------------------------------------------------------------------------


class _FakeStackNode(dict):
    """Dict-backed stand-in for a Tiled container-stack: ``node[key]`` -> array."""


def test_stack_path_no_downsampling_each_slice_matches_its_constant_value() -> None:
    values = [0, 50, 100, 150, 200]
    keys = [f"slice_{i}" for i in range(5)]
    node = _FakeStackNode({k: np.full((20, 20), v, dtype=np.uint8) for k, v in zip(keys, values)})
    meta = {
        "n_slices": 5, "height": 20, "width": 20, "dtype": "uint8",
        "is_rgb": False, "shape_kind": "STACK", "keys": keys,
    }
    opts = {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0}
    global_range = (0.0, 200.0)

    payload, vol_meta = volumes.build_volume(node, meta, opts, max_dim=32, global_range=global_range)

    assert vol_meta["sxy"] == 1  # max_dim(32) > height/width(20): no downsampling
    assert vol_meta["sz"] == 1
    assert (vol_meta["nz"], vol_meta["ny"], vol_meta["nx"]) == (5, 20, 20)
    assert vol_meta["skipped_z"] == []

    volume = np.frombuffer(payload, dtype=np.uint8).reshape(5, 20, 20)
    for i, v in enumerate(values):
        expected_unit = images.normalize_scalar_unit(np.array([[v]], dtype=np.uint8), opts, global_range)
        expected = round(float(expected_unit[0, 0]) * 255.0)
        assert np.array_equal(volume[i], np.full((20, 20), expected, dtype=np.uint8))


def test_stack_path_with_downsampling_trims_to_the_floor_shape_not_ceil() -> None:
    """gridFor FLOORS after striding but a plain ``arr[::stride]`` CEILS —
    build_volume must trim, not just rely on the stride, or the output would
    be one row/col larger than the dims contract (and than the frontend's
    own label volume) whenever height/width isn't an exact multiple of sxy.
    """
    values = [10, 20, 30]
    keys = [f"slice_{i}" for i in range(3)]
    node = _FakeStackNode({k: np.full((20, 20), v, dtype=np.uint8) for k, v in zip(keys, values)})
    meta = {
        "n_slices": 3, "height": 20, "width": 20, "dtype": "uint8",
        "is_rgb": False, "shape_kind": "STACK", "keys": keys,
    }
    opts = {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0}
    dims = volumes.volume_dims(3, 20, 20, max_dim=7)

    payload, vol_meta = volumes.build_volume(node, meta, opts, max_dim=7, global_range=(0.0, 30.0))

    assert vol_meta["sxy"] == dims["sxy"] > 1  # actually downsampling this time
    volume = np.frombuffer(payload, dtype=np.uint8).reshape(vol_meta["nz"], vol_meta["ny"], vol_meta["nx"])
    assert volume.shape == (dims["nz"], dims["ny"], dims["nx"])

    # A bare `arr[::sxy]` over 20 elements would ceil to a longer axis than
    # dims' floor division — assert the untrimmed length really is bigger, so
    # this test would have caught a missing trim rather than passing by luck.
    raw_len = len(range(0, 20, dims["sxy"]))
    assert dims["nx"] < raw_len


def test_one_unreadable_stack_slice_is_zeroed_not_a_failure() -> None:
    """A single bad sibling in a STACK container must not fail the whole
    volume: its output z-slice is recorded in skipped_z and left as zero,
    while every other slice is still correctly populated."""

    class _FlakyStackNode(_FakeStackNode):
        def __getitem__(self, key):
            if key == "bad":
                raise RuntimeError("simulated unreadable slice")
            return super().__getitem__(key)

    good_values = {"a": 10, "c": 30, "d": 40, "e": 50}
    keys = ["a", "bad", "c", "d", "e"]
    node = _FlakyStackNode({k: np.full((8, 8), v, dtype=np.uint8) for k, v in good_values.items()})
    node["bad"] = np.zeros((8, 8), dtype=np.uint8)  # present, but __getitem__ always raises for it
    meta = {
        "n_slices": 5, "height": 8, "width": 8, "dtype": "uint8",
        "is_rgb": False, "shape_kind": "STACK", "keys": keys,
    }
    opts = {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0}
    global_range = (0.0, 50.0)

    payload, vol_meta = volumes.build_volume(node, meta, opts, max_dim=32, global_range=global_range)

    assert vol_meta["skipped_z"] == [1]  # z-index 1 == the "bad" key's position

    volume = np.frombuffer(payload, dtype=np.uint8).reshape(vol_meta["nz"], vol_meta["ny"], vol_meta["nx"])
    assert np.all(volume[1] == 0)
    for i, key in enumerate(keys):
        if key == "bad":
            continue
        expected_unit = images.normalize_scalar_unit(
            np.array([[good_values[key]]], dtype=np.uint8), opts, global_range
        )
        expected = round(float(expected_unit[0, 0]) * 255.0)
        assert np.array_equal(volume[i], np.full((8, 8), expected, dtype=np.uint8))


# ---------------------------------------------------------------------------
# 5. NHWC luminance weighting
# ---------------------------------------------------------------------------


def test_nhwc_luminance_weights_differ_per_channel() -> None:
    """Each solid-color slice must reduce to a different gray level, ordered
    by the standard luminance weights (green heaviest, blue lightest) — this
    is the guarantee that an RGB volume isn't just averaging channels."""
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[..., 0] = 255
    green = np.zeros((10, 10, 3), dtype=np.uint8)
    green[..., 1] = 255
    blue = np.zeros((10, 10, 3), dtype=np.uint8)
    blue[..., 2] = 255
    arr = np.stack([red, green, blue], axis=0)
    meta = {"n_slices": 3, "height": 10, "width": 10, "dtype": "uint8", "is_rgb": True, "shape_kind": "NHWC"}
    opts = {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0}
    global_range = (0.0, 255.0)

    payload, vol_meta = volumes.build_volume(arr, meta, opts, max_dim=32, global_range=global_range)
    volume = np.frombuffer(payload, dtype=np.uint8).reshape(vol_meta["nz"], vol_meta["ny"], vol_meta["nx"])

    red_mean, green_mean, blue_mean = (float(volume[i].mean()) for i in range(3))
    assert green_mean > red_mean > blue_mean

    # global_range == (0, 255) makes normalize_scalar_unit a passthrough, so
    # the recovered gray level should match the textbook luminance weights.
    assert abs(red_mean - round(0.299 * 255)) <= 2
    assert abs(green_mean - round(0.587 * 255)) <= 2
    assert abs(blue_mean - round(0.114 * 255)) <= 2


# ---------------------------------------------------------------------------
# 7. Unconfigured tiled server rejected before doing any work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unconfigured_tiled_server_is_rejected_before_building_a_volume(monkeypatch) -> None:
    def _reject(uri):
        raise annotation_server.HTTPException(403, "Tiled server is not configured")

    monkeypatch.setattr(annotation_server, "_require_tiled_server", _reject)
    # If the route did any work before checking the server, this would blow up
    # instead of a clean 403 — resolve_array must never be reached.
    monkeypatch.setattr(
        annotation_server.arrays_mod,
        "resolve_array",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resolve_array should not run")),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/image/volume", params={"source": "x", "kind": "tiled"}
        )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# 8. Cache actually caches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_requests_with_identical_params_hit_the_volume_cache(monkeypatch) -> None:
    arr = np.arange(4 * 16 * 16, dtype=np.uint8).reshape(4, 16, 16)
    calls: list[tuple] = []

    def _counting_resolve(source, kind, server_uri, root):
        calls.append((source, kind, server_uri, root))
        return arr

    monkeypatch.setattr(annotation_server.arrays_mod, "resolve_array", _counting_resolve)

    params = {"source": "cache-test", "kind": "local", "max_dim": 32}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.get("/api/image/volume", params=params)
        r2 = await client.get("/api/image/volume", params=params)
        r3 = await client.get("/api/image/volume", params={**params, "max_dim": 64})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200
    # r2 is identical to r1 -> served from _volume_cache, no second resolve_array
    # call. r3 changes max_dim (part of the cache key) -> must miss and re-run.
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# 9. OpenAPI registration
# ---------------------------------------------------------------------------


def test_openapi_image_volume_route_is_registered() -> None:
    schema = app.openapi()

    assert "/api/image/volume" in schema["paths"]
    assert "get" in schema["paths"]["/api/image/volume"]


# ---------------------------------------------------------------------------
# Quality ladder + voxel budget (mirrored by frontend/src/lib/volume/volumeDims.ts)
# ---------------------------------------------------------------------------

def test_effective_max_dim_never_exceeds_the_voxel_budget() -> None:
    """The budget is what makes raising the ladder to 1024 safe: a cubic source
    would otherwise reach 1024**3 = 1 GB per volume, and the 3-D tab holds two."""
    for n_slices, h, w in [(63, 3232, 3232), (2000, 2000, 2000), (5000, 512, 512), (1, 4096, 4096)]:
        for requested in volumes.QUALITY_LADDER:
            eff = volumes.effective_max_dim(n_slices, h, w, requested)
            d = volumes.volume_dims(n_slices, h, w, eff)
            assert d["nz"] * d["ny"] * d["nx"] <= volumes.MAX_VOLUME_VOXELS, (n_slices, h, w, requested)


def test_effective_max_dim_never_exceeds_what_was_asked_for() -> None:
    for requested in volumes.QUALITY_LADDER:
        assert volumes.effective_max_dim(63, 3232, 3232, requested) <= requested


def test_a_thin_stack_reaches_the_top_of_the_ladder() -> None:
    """The point of budgeting voxels instead of capping each axis: 63 x 3232^2
    at 1024 is only ~41 MB, so it should NOT be clamped."""
    assert volumes.effective_max_dim(63, 3232, 3232, 1024) == 1024


def test_a_cubic_source_is_clamped_below_the_top_of_the_ladder() -> None:
    assert volumes.effective_max_dim(2000, 2000, 2000, 1024) < 1024


def test_effective_max_dim_is_monotonic_in_the_request() -> None:
    """Asking for more quality must never yield a coarser grid."""
    seen = [volumes.effective_max_dim(2000, 2000, 2000, q) for q in volumes.QUALITY_LADDER]
    assert seen == sorted(seen)


def test_a_non_ladder_request_that_fits_passes_through_untouched() -> None:
    """The route accepts any int in 32..1024, not just ladder entries, and this
    must never round one UP — the frontend rasterizes its label volume onto
    whatever grid comes back."""
    for requested in (32, 100, 257, 511, 1000):
        assert volumes.effective_max_dim(63, 3232, 3232, requested) == requested


def test_clamping_only_ever_reduces() -> None:
    for n_slices, h, w in [(63, 3232, 3232), (2000, 2000, 2000), (5000, 512, 512)]:
        for requested in (32, 100, 256, 384, 700, 1024):
            assert volumes.effective_max_dim(n_slices, h, w, requested) <= requested
