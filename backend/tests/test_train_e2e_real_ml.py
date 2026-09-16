"""Real (not mocked) end-to-end check of the core train/save/load contract.

Skips automatically if the `ml` extra (torch/dlsia) isn't installed. When it
IS installed, this proves the ported train_common/dlsia_runtime/
autoencoder_runtime machinery actually works — builds a real model, runs a
real training step, saves real weights, reloads them, and confirms inference
on the reloaded model reproduces the trained model's output — not just that
the error paths are clean (see test_train_common.py for those).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("dlsia")

import autoencoder_runtime  # noqa: E402
import dlsia_runtime  # noqa: E402
import train_common  # noqa: E402
from schemas import DlsiaDenoiserConfig, DlsiaTunetConfig  # noqa: E402


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _synthetic_seg_pairs(n: int, size: int, n_classes: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        rgb = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        label = rng.integers(0, n_classes, size=(size, size), dtype=np.uint8)
        pairs.append((rgb, label))
    return pairs


def test_dlsia_tunet_train_save_reload_infer_round_trip(runs_dir: Path) -> None:
    n_classes = 2
    image_size = 64
    model_cfg = DlsiaTunetConfig(
        hyperparams={
            "epochs": 1,
            "depth": 2,
            "base_channels": 4,
            "growth_rate": 1.2,
            "batch_size": 2,
            "image_size": image_size,
            "tiling": False,
        }
    )

    logs: list[str] = []
    built = train_common.build_family(model_cfg, n_classes, "cpu", logs.append)
    assert built.set_train_mode is not None

    train_pairs = _synthetic_seg_pairs(4, image_size, n_classes)
    metrics = train_common.run_training_loop(
        train_pairs=train_pairs,
        val_pairs=[],
        image_size=image_size,
        n_classes=n_classes,
        epochs=model_cfg.hyperparams.epochs,
        batch_size=model_cfg.hyperparams.batch_size,
        seed=model_cfg.hyperparams.seed,
        flip_augment=False,
        to_tensor_fn=built.to_tensor_fn,
        forward_fn=built.forward_fn,
        trainable_params=built.trainable_params,
        lr=model_cfg.hyperparams.lr,
        device="cpu",
        set_train_mode=built.set_train_mode,
    )
    assert metrics["epochs_completed"] == 1
    assert metrics["cancelled"] is False
    assert np.isfinite(metrics["final_train_loss"])

    run_id = "test-tunet-run"
    train_common.save_run(
        run_id,
        model_family=model_cfg.model_family,
        model_config=built.model_config_snapshot,
        classes=[{"classId": 1, "label": "a", "color": "#f00"}, {"classId": 2, "label": "b", "color": "#0f0"}],
        render={},
        image_size=image_size,
        hyperparams=model_cfg.hyperparams.model_dump(),
        source_keys=["local:fake.tif"],
        adapter_state=built.adapter_state_fn(),
        metrics=metrics,
    )

    # list_runs / load_run_config round-trip.
    runs = train_common.list_runs()
    assert any(r["run_id"] == run_id for r in runs)
    loaded_config = train_common.load_run_config(run_id)
    assert loaded_config["model_family"] == "dlsia_tunet"
    assert loaded_config["task"] == "segmentation"

    # Reload the ACTUAL saved weights into a fresh model and confirm inference
    # runs and reproduces the just-trained model's output (not just "doesn't crash").
    loaded_state = train_common.load_adapter_state(run_id)
    reloaded_model = dlsia_runtime.load_model(loaded_state, "cpu")
    reloaded_model.eval()
    reloaded_forward = dlsia_runtime.make_forward_fn(reloaded_model)

    # eval() on both sides: TUNet's BatchNorm gives different output in train
    # mode (batch statistics) vs eval mode (running stats) — comparing them in
    # mismatched modes would fail even with byte-identical weights.
    built.set_train_mode(False)
    sample_rgb, _ = train_pairs[0]
    batch = built.to_tensor_fn(sample_rgb).unsqueeze(0)
    with torch.no_grad():
        original_logits = built.forward_fn(batch)
        reloaded_logits = reloaded_forward(batch)
    assert original_logits.shape == (1, n_classes, image_size, image_size)
    torch.testing.assert_close(original_logits, reloaded_logits)

    train_common.delete_run(run_id)
    assert not any(r["run_id"] == run_id for r in train_common.list_runs())


def test_dlsia_denoiser_cnn_ae_build_save_reload_round_trip(runs_dir: Path) -> None:
    """Pure-torch denoiser architecture (no dlsia needed for this one) — build,
    save, and reload real weights, confirming the autoencoder_runtime six-function
    template round-trips correctly through train_common.build_family/save_run."""
    image_size = 64
    model_cfg = DlsiaDenoiserConfig(
        architecture="cnn_ae",
        training_scheme="ae",
        ae_compression=4,
        hyperparams={"depth": 2, "base_channels": 4, "image_size": image_size},
    )

    built = train_common.build_family(model_cfg, 0, "cpu", lambda _msg: None)

    run_id = "test-ae-run"
    train_common.save_run(
        run_id,
        model_family=model_cfg.model_family,
        model_config={**built.model_config_snapshot, "training_scheme": "ae", "architecture": "cnn_ae"},
        classes=[],
        render={},
        image_size=image_size,
        hyperparams=model_cfg.hyperparams.model_dump(),
        source_keys=[],
        adapter_state=built.adapter_state_fn(),
        metrics={"epochs_completed": 0, "cancelled": False},
        task="denoising",
    )

    loaded_config = train_common.load_run_config(run_id)
    assert loaded_config["task"] == "denoising"
    assert loaded_config["model_config"]["architecture"] == "cnn_ae"

    loaded_state = train_common.load_adapter_state(run_id)
    reloaded = autoencoder_runtime.load_model(loaded_state, "cpu")
    reloaded.eval()
    reloaded_forward = autoencoder_runtime.make_forward_fn(reloaded)

    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(image_size, image_size), dtype=np.uint8)
    to_tensor = autoencoder_runtime.make_to_tensor_fn()
    batch = to_tensor(gray).unsqueeze(0)
    with torch.no_grad():
        out = reloaded_forward(batch)
    assert out.shape == (1, 1, image_size, image_size)
