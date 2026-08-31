"""dlsia TUNet model building for the Train tab.

A TUNet trains from scratch — no pretrained checkpoint, no license
restriction beyond dlsia's own (BSD). dlsia is an optional dependency
(``backend/pyproject.toml``'s ``ml`` extra, alongside ``torch``) — this
module is import-safe without it installed; only
:func:`build_model`/:func:`load_model` actually import it, guarded by
``train_common.dlsia_available()``.

dlsia's own ``train_scripts.train_segmentation`` runs a whole training call
synchronously with no per-batch/per-epoch hook, so it can't report progress
or honour a cooperative cancel — this app calls dlsia only for the ``TUNet``
model class (and its own save/load helpers) and drives training through
``train_common.run_training_loop`` instead.
"""

from __future__ import annotations

from typing import Any


def build_model(
    n_classes: int, image_size: int, depth: int, base_channels: int, growth_rate: float, device: str
) -> Any:
    """Construct a fresh (untrained) dlsia TUNet for *n_classes* output channels.

    ``image_shape`` is fixed to ``(image_size, image_size)`` at construction —
    dlsia's TUNet precomputes exact per-layer tensor sizes from it, so every
    training and inference image must be letterboxed to this same size.
    """
    from dlsia.core.networks.tunet import TUNet  # noqa: PLC0415 — optional dependency

    model = TUNet(
        image_shape=(image_size, image_size),
        in_channels=3,
        out_channels=n_classes,
        depth=depth,
        base_channels=base_channels,
        growth_rate=growth_rate,
    )
    return model.to(device)


def network_dict(model: Any) -> dict[str, Any]:
    """The full ``{topo_dict, state_dict}`` dlsia uses to reconstruct a TUNet
    (see ``TUNet.save_network_parameters`` / ``TUNetwork_from_file``). Stored
    as-is in this run's ``adapter.pt`` — TUNet trains from scratch, so there's
    no base/delta split the way LoRA has."""
    return model.save_network_parameters(name=None)


def load_model(state: dict[str, Any], device: str) -> Any:
    """Reconstruct a TUNet from a saved :func:`network_dict`."""
    from dlsia.core.networks.tunet import TUNet  # noqa: PLC0415

    model = TUNet(**state["topo_dict"])
    model.load_state_dict(state["state_dict"])
    return model.to(device)


def make_forward_fn(model: Any):
    """Return ``forward(batch_images) -> logits`` for :func:`train_common.run_training_loop`.

    TUNet's transposed-conv decoder is symmetric with its encoder, so output
    spatial size already matches input size — no resizing needed.
    """

    def _forward(batch_images: Any) -> Any:
        return model(batch_images)

    return _forward


def make_set_train_mode_fn(model: Any):
    """Return ``set_train_mode(is_training)`` for :func:`train_common.run_training_loop`.

    TUNet defaults to ``nn.BatchNorm2d`` — its running mean/var should only
    update during training, not while computing validation metrics.
    """

    def _set_train_mode(is_training: bool) -> None:
        model.train(is_training)

    return _set_train_mode


def make_to_tensor_fn():
    """Return ``(rgb_uint8_hwc) -> float CPU tensor (3,H,W)`` scaled to [0, 1].

    No ImageNet normalisation — the model trains from scratch on this data,
    so there's no pretrained-backbone convention to match.
    """
    import numpy as np
    import torch

    def _to_tensor(rgb_uint8: "np.ndarray") -> Any:
        return torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(2, 0, 1).float() / 255.0

    return _to_tensor
