"""Inference reapplies the input denoising recorded on the run.

The other half of the invariant covered by ``test_denoise_train_input.py``:
there it is proved that training and inference *derive* identical pixels from
the same recorded settings; here it is proved that ``run_infer_job`` actually
consults the RUN for them, and never the request. If it read the request (or
nothing), a model trained on denoised input would silently predict on raw
pixels — a distribution shift with no error attached.

Reuses ``test_infer_jobs_tiling``'s fakes rather than defining a second set.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import export_jobs  # noqa: E402
import images as images_mod  # noqa: E402
import arrays as arrays_mod  # noqa: E402
import dlsia_runtime  # noqa: E402
import infer_jobs  # noqa: E402
import train_common  # noqa: E402
from schemas import InferRequest  # noqa: E402
from tests.test_infer_jobs_tiling import _fake_config, _install_common_fakes  # noqa: E402

# Captured at import, before any test stubs it out.
_REAL_RENDER_SLICE = images_mod.render_slice


def _run() -> dict:
    request = InferRequest(run_id="run-1", kind="local", source="s.tif", slice_indices=[0])
    jid = export_jobs.new_job("test")
    infer_jobs.run_infer_job(jid, request)
    return export_jobs.get_job(jid)


def _spy_on_denoise(monkeypatch) -> list:
    """Record every ``denoising_render_slice_fn`` argument infer_jobs passes."""
    seen: list = []
    real = train_common.denoising_render_slice_fn

    def _spy(denoise):
        seen.append(denoise)
        return real(denoise)

    monkeypatch.setattr(train_common, "denoising_render_slice_fn", _spy)
    return seen


def test_a_run_with_recorded_denoising_reapplies_it(monkeypatch) -> None:
    config = _fake_config(tiling_flag=False)
    config["denoise"] = {"method": "tv", "strength": 0.6}
    _install_common_fakes(monkeypatch, config)
    seen = _spy_on_denoise(monkeypatch)

    assert _run()["state"] == "done"
    assert seen == [{"method": "tv", "strength": 0.6}]


def test_a_run_without_denoising_predicts_on_raw_pixels(monkeypatch) -> None:
    config = _fake_config(tiling_flag=False)
    config["denoise"] = None
    _install_common_fakes(monkeypatch, config)
    seen = _spy_on_denoise(monkeypatch)

    assert _run()["state"] == "done"
    assert seen == [None]


def test_a_legacy_run_with_no_denoise_key_predicts_on_raw_pixels(monkeypatch) -> None:
    """Every run saved before this field existed was trained on raw pixels, so
    an absent key must mean "no denoising" rather than KeyError."""
    config = _fake_config(tiling_flag=False)
    config.pop("denoise", None)
    _install_common_fakes(monkeypatch, config)
    seen = _spy_on_denoise(monkeypatch)

    assert _run()["state"] == "done"
    assert seen == [None]


def test_the_recorded_denoising_actually_reaches_the_rendered_pixels(monkeypatch) -> None:
    """Not just "the helper was called with the right argument" — the pixels the
    model receives must genuinely be the denoised ones."""
    config = _fake_config(tiling_flag=False)
    config["denoise"] = {"method": "tv", "strength": 0.8}
    _install_common_fakes(monkeypatch, config)

    # A genuinely noisy slice, and the REAL render_slice (the shared fakes stub
    # it to zeros, which denoising could not measurably change).
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.normal(128, 60, (16, 16)), 0, 255).astype(np.uint8)
    monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda n: {"height": 16, "width": 16})
    monkeypatch.setattr(arrays_mod, "read_slice", lambda n, m, i: noisy)
    monkeypatch.setattr(images_mod, "render_slice", _REAL_RENDER_SLICE)

    fed: list[np.ndarray] = []

    def _capturing_to_tensor():
        def _to_tensor(rgb):
            fed.append(np.asarray(rgb).copy())
            return torch.from_numpy(np.asarray(rgb).transpose(2, 0, 1)).float() / 255.0
        return _to_tensor

    monkeypatch.setattr(dlsia_runtime, "make_to_tensor_fn", _capturing_to_tensor)

    assert _run()["state"] == "done"
    assert fed, "the model was never fed a slice"

    plain = _REAL_RENDER_SLICE(noisy, config["render"], None)
    assert not np.array_equal(fed[0], plain), "model received raw pixels despite a denoised run"
