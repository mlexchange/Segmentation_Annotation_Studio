"""Model-input denoising: applied at training, reapplied automatically at inference.

The whole point of routing both sides through
``train_common.denoising_render_slice_fn`` is that they CANNOT disagree. A model
trained on denoised pixels but predicting on raw ones (or vice versa) is a silent
distribution shift — no exception, no warning, just quietly worse predictions —
so these tests are about that invariant, not about the filtering itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import images as images_mod
import train_common
from schemas import DenoiseTrainOpts


def _noisy(size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = np.zeros((size, size), dtype=np.float64)
    clean[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 3000.0
    return np.clip(clean + rng.normal(0, 250, clean.shape), 0, 65535).astype(np.uint16)


OPTS = {"norm": "slice", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"}


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("off", [None, {"method": "none"}, {"method": "none", "strength": 0.7}])
def test_no_denoising_returns_render_slice_itself(off) -> None:
    """Not merely equivalent — the same function object, so the un-denoised path
    is provably byte-identical to what it was before this feature existed."""
    assert train_common.denoising_render_slice_fn(off) is images_mod.render_slice


def test_a_learned_denoiser_is_not_accepted_as_a_preprocessor() -> None:
    """'model' would need its own run and a GPU forward pass per slice; it is
    deliberately not supported here, and must degrade to "no denoising" rather
    than being silently mistaken for a classical filter name."""
    assert train_common.denoising_render_slice_fn({"method": "model"}) is images_mod.render_slice


def test_denoising_changes_the_rendered_pixels() -> None:
    arr = _noisy()

    plain = images_mod.render_slice(arr, OPTS, None)
    denoised = train_common.denoising_render_slice_fn(
        DenoiseTrainOpts(method="tv", strength=0.8)
    )(arr, OPTS, None)

    assert denoised.shape == plain.shape
    assert not np.array_equal(denoised, plain)


def test_denoised_training_pixels_are_smoother_than_raw() -> None:
    """Sanity that it denoises rather than merely perturbing: less pixel-to-pixel
    variation inside a region that is uniform in the underlying phantom."""
    arr = _noisy()
    interior = (slice(20, 44), slice(20, 44))

    plain = images_mod.render_slice(arr, OPTS, None)[interior].astype(np.float64)
    denoised = train_common.denoising_render_slice_fn(
        DenoiseTrainOpts(method="tv", strength=0.8)
    )(arr, OPTS, None)[interior].astype(np.float64)

    assert np.std(np.diff(denoised, axis=1)) < np.std(np.diff(plain, axis=1))


def test_a_dict_from_a_saved_run_behaves_like_the_pydantic_model() -> None:
    """Training passes a DenoiseTrainOpts; inference passes the plain dict
    reloaded from config.json. Both must produce identical pixels — this is the
    exact seam where train and predict could drift apart."""
    arr = _noisy()

    from_model = train_common.denoising_render_slice_fn(
        DenoiseTrainOpts(method="tv", strength=0.6)
    )(arr, OPTS, None)
    from_dict = train_common.denoising_render_slice_fn(
        {"method": "tv", "strength": 0.6}
    )(arr, OPTS, None)

    assert np.array_equal(from_model, from_dict)


def test_strength_defaults_match_between_the_two_representations() -> None:
    arr = _noisy()

    explicit = train_common.denoising_render_slice_fn({"method": "tv", "strength": 0.5})(arr, OPTS, None)
    defaulted = train_common.denoising_render_slice_fn({"method": "tv"})(arr, OPTS, None)

    assert np.array_equal(explicit, defaulted)


def test_a_colour_slice_passes_through_untouched() -> None:
    """render_slice early-returns for RGB, and the classical filters are 2-D
    grayscale — a colour source must not crash the training render."""
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = 200

    out = train_common.denoising_render_slice_fn({"method": "tv", "strength": 0.5})(rgb, OPTS, None)

    assert np.array_equal(out, images_mod.render_slice(rgb, OPTS, None))


# ---------------------------------------------------------------------------
# The round trip: what training saves is what inference reapplies
# ---------------------------------------------------------------------------

def test_a_run_records_its_input_denoising(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(train_common, "runs_dir", lambda: tmp_path)

    train_common.save_run(
        "run-denoised",
        model_family="dlsia_tunet",
        model_config={},
        classes=[{"classId": 1, "label": "a"}],
        render=OPTS,
        image_size=512,
        hyperparams={"tiling": True},
        source_keys=["k"],
        adapter_state={},
        metrics={},
        denoise={"method": "tv", "strength": 0.6},
    )

    config = train_common.load_run_config("run-denoised")
    assert config["denoise"] == {"method": "tv", "strength": 0.6}


def test_inference_reconstructs_the_exact_training_preprocessing(tmp_path, monkeypatch) -> None:
    """The end-to-end invariant. Pixels fed to the model at predict time must be
    identical to those it trained on — reconstructed from the RUN, with the
    caller supplying nothing."""
    pytest.importorskip("torch")
    monkeypatch.setattr(train_common, "runs_dir", lambda: tmp_path)
    arr = _noisy()

    train_opts = DenoiseTrainOpts(method="tv", strength=0.6)
    trained_on = train_common.denoising_render_slice_fn(train_opts)(arr, OPTS, None)

    train_common.save_run(
        "run-x", model_family="dlsia_tunet", model_config={}, classes=[],
        render=OPTS, image_size=512, hyperparams={}, source_keys=[],
        adapter_state={}, metrics={}, denoise=train_opts.model_dump(),
    )

    # Exactly what infer_jobs does: read the run, rebuild the render fn.
    config = train_common.load_run_config("run-x")
    predicted_on = train_common.denoising_render_slice_fn(config.get("denoise"))(arr, OPTS, None)

    assert np.array_equal(trained_on, predicted_on)


def test_a_run_trained_without_denoising_predicts_on_raw_pixels(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(train_common, "runs_dir", lambda: tmp_path)
    arr = _noisy()

    train_common.save_run(
        "run-plain", model_family="dlsia_tunet", model_config={}, classes=[],
        render=OPTS, image_size=512, hyperparams={}, source_keys=[],
        adapter_state={}, metrics={},
    )

    config = train_common.load_run_config("run-plain")
    out = train_common.denoising_render_slice_fn(config.get("denoise"))(arr, OPTS, None)

    assert np.array_equal(out, images_mod.render_slice(arr, OPTS, None))


def test_a_legacy_run_with_no_denoise_key_is_treated_as_raw(tmp_path, monkeypatch) -> None:
    """Runs saved before this field existed have no "denoise" key at all. They
    were trained on raw pixels, so that is what they must predict on."""
    pytest.importorskip("torch")
    monkeypatch.setattr(train_common, "runs_dir", lambda: tmp_path)

    import json

    d = tmp_path / "legacy"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "run_id": "legacy", "model_family": "dlsia_tunet", "classes": [],
        "render": OPTS, "image_size": 512, "hyperparams": {}, "source_keys": [],
    }))

    config = train_common.load_run_config("legacy")
    assert "denoise" not in config or config["denoise"] is None
    assert train_common.denoising_render_slice_fn(config.get("denoise")) is images_mod.render_slice


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

def test_train_request_denoise_is_optional_and_defaults_to_none() -> None:
    from schemas import DlsiaTunetConfig, TrainRequest

    req = TrainRequest(
        sources=[{"kind": "tiled", "source": "browse/ds", "server_uri": "u",
                  "slices": {"0": []}, "split_by_slice": {}, "negative_slices": []}],
        classes=[{"classId": 1, "label": "a", "color": "#ff0000"}],
        model=DlsiaTunetConfig(),
    )

    assert req.denoise is None


def test_train_request_rejects_an_out_of_range_strength() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        DenoiseTrainOpts(method="tv", strength=1.5)


# ---------------------------------------------------------------------------
# Resume inherits the parent's input denoising
# ---------------------------------------------------------------------------

def _resume_request():
    from schemas import DlsiaTunetConfig, TrainRequest

    return TrainRequest(
        sources=[{"kind": "tiled", "source": "browse/ds", "server_uri": "u",
                  "slices": {"0": []}, "split_by_slice": {}, "negative_slices": []}],
        classes=[{"classId": 1, "label": "a", "color": "#ff0000"}],
        model=DlsiaTunetConfig(),
        resume_from_run_id="parent",
    )


def test_resume_inherits_the_parents_denoising_even_if_the_client_omits_it() -> None:
    """The parent's weights were fit to denoised pixels; continuing to train
    them on raw ones degrades them with no error. Inherited like image_size and
    tiling, not left to the caller."""
    import train_jobs

    request = _resume_request()
    assert request.denoise is None  # client sent nothing

    train_jobs._apply_parent_architecture(
        {"hyperparams": {}, "model_config": {}, "denoise": {"method": "tv", "strength": 0.6}},
        request,
    )

    assert request.denoise is not None
    assert (request.denoise.method, request.denoise.strength) == ("tv", 0.6)


def test_resume_clears_denoising_the_parent_did_not_use() -> None:
    """The mirror case: a raw-trained parent must not pick up whatever the
    client happened to send."""
    import train_jobs

    request = _resume_request()
    request.denoise = DenoiseTrainOpts(method="nlm", strength=0.9)

    train_jobs._apply_parent_architecture({"hyperparams": {}, "model_config": {}}, request)

    assert request.denoise is None


def test_resume_overrides_a_conflicting_client_choice() -> None:
    import train_jobs

    request = _resume_request()
    request.denoise = DenoiseTrainOpts(method="gaussian", strength=0.1)

    train_jobs._apply_parent_architecture(
        {"hyperparams": {}, "model_config": {}, "denoise": {"method": "tv", "strength": 0.6}},
        request,
    )

    assert (request.denoise.method, request.denoise.strength) == ("tv", 0.6)
