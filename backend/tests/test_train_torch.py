"""Torch-dependent tests for train_common: the parts test_train_common.py
can't exercise without a real tensor backend. Skipped entirely when torch
isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

import train_common  # noqa: E402


def _to_tensor_fn(rgb: np.ndarray) -> "torch.Tensor":
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def _make_forward_fn(n_classes: int):
    """A fixed, untrained 1x1 conv: deterministic given the input, and — since
    a conv has no cross-sample state like BatchNorm — independent of which
    other samples share its batch. Batching validation must not change any
    individual sample's logits, only how many forward calls it takes."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, n_classes, kernel_size=1)
    conv.eval()

    def _forward(batch_images: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            return conv(batch_images)

    return _forward


def _val_pairs(n: int, size: int, n_classes: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [
        (
            rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8),
            rng.integers(0, n_classes, size=(size, size), dtype=np.uint8),
        )
        for _ in range(n)
    ]


def test_evaluate_batching_matches_one_sample_at_a_time() -> None:
    """Regression guard for the validation-batching refactor: grouping samples
    into forward-pass batches must report the same loss/mIoU as scoring them
    one at a time, not a pooled-across-the-batch approximation."""
    n_classes = 3
    image_size = 8
    val_pairs = _val_pairs(5, image_size, n_classes)
    forward_fn = _make_forward_fn(n_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=train_common.IGNORE_INDEX)

    loss_1, miou_1 = train_common._evaluate(
        val_pairs, image_size, n_classes, _to_tensor_fn, forward_fn, "cpu", criterion, batch_size=1
    )
    loss_3, miou_3 = train_common._evaluate(
        val_pairs, image_size, n_classes, _to_tensor_fn, forward_fn, "cpu", criterion, batch_size=3
    )
    loss_all, miou_all = train_common._evaluate(
        val_pairs, image_size, n_classes, _to_tensor_fn, forward_fn, "cpu", criterion, batch_size=len(val_pairs)
    )

    assert loss_3 == pytest.approx(loss_1, abs=1e-5)
    assert loss_all == pytest.approx(loss_1, abs=1e-5)
    assert miou_3 == pytest.approx(miou_1, abs=1e-6)
    assert miou_all == pytest.approx(miou_1, abs=1e-6)


def test_evaluate_batch_size_defaults_to_one() -> None:
    """`batch_size` is an optional trailing arg — callers written before it
    existed must keep behaving exactly as they did before."""
    n_classes = 2
    image_size = 8
    val_pairs = _val_pairs(3, image_size, n_classes)
    forward_fn = _make_forward_fn(n_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=train_common.IGNORE_INDEX)

    default_result = train_common._evaluate(
        val_pairs, image_size, n_classes, _to_tensor_fn, forward_fn, "cpu", criterion
    )
    explicit_result = train_common._evaluate(
        val_pairs, image_size, n_classes, _to_tensor_fn, forward_fn, "cpu", criterion, batch_size=1
    )
    assert default_result == explicit_result


def test_run_training_loop_uses_the_training_batch_size_for_validation() -> None:
    """End-to-end smoke test: a training run with a val set larger than the
    batch size completes and reports validation metrics, regardless of how
    many samples land in a single forward-pass batch."""
    n_classes = 2
    image_size = 8
    train_pairs = _val_pairs(4, image_size, n_classes, seed=1)
    val_pairs = _val_pairs(5, image_size, n_classes, seed=2)

    torch.manual_seed(0)
    conv = nn.Conv2d(3, n_classes, kernel_size=1)

    def forward_fn(batch_images: "torch.Tensor") -> "torch.Tensor":
        return conv(batch_images)

    metrics = train_common.run_training_loop(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        image_size=image_size,
        n_classes=n_classes,
        epochs=1,
        batch_size=3,
        seed=0,
        flip_augment=False,
        to_tensor_fn=_to_tensor_fn,
        forward_fn=forward_fn,
        trainable_params=list(conv.parameters()),
        lr=1e-3,
        device="cpu",
    )

    assert metrics["epochs_completed"] == 1
    assert metrics["val_miou"] is not None
    assert metrics["final_val_loss"] is not None
