"""dlsia TUNet model building for the Train tab's self-supervised denoiser
(Noise2Noise / Noise2Void) — the third model family alongside dlsia TUNet
segmentation (see :mod:`dlsia_runtime`).

Mirrors :mod:`dlsia_runtime`'s function template closely — same shared TUNet
architecture, same save/load/forward-fn/train-mode plumbing — but wired for
regression instead of classification:

* Fixed at ``in_channels=1, out_channels=1``: a single noisy intensity
  channel in, a single denoised intensity channel out. There is no
  "n_classes" here — the output channel count isn't user-configurable, unlike
  the segmentation family's ``out_channels=n_classes``.
* No softmax/argmax anywhere in this module. A classification head turns
  logits into class probabilities/labels; a regression head's raw output IS
  the answer (the denoised pixel value), so applying either would corrupt it.

dlsia is an optional dependency (``backend/pyproject.toml``'s ``ml`` extra,
alongside ``torch``) — this module is import-safe without it installed; only
:func:`build_model`/:func:`load_model` actually import it, guarded by
``train_common.dlsia_available()``.

Training itself is driven by ``backend/denoise_train.py`` (not this module,
and not written here) — this module only builds/saves/loads the network and
its forward pass, the same division of labour ``dlsia_runtime`` has with the
generic loop in ``train_common.run_training_loop``.

Documented future extension points (not implemented here — real classes in
``dlsia.core.networks``, kept as an easy on-ramp if TUNet turns out not to be
the best fit for denoising specifically):
  * ``MSDNet`` / ``MSDAE`` — mixed-scale dense networks; an alternative to
    TUNet's encoder/decoder that some denoising literature prefers.
  * ``MSAE`` / ``SparseAutoEncoder`` — autoencoder-style architectures.
Any of these could plug in behind the same function names this module
exposes, without changing ``train_common.build_family``'s call sites.
"""

from __future__ import annotations

from typing import Any


def build_model(image_size: int, depth: int, base_channels: int, growth_rate: float, device: str) -> Any:
    """Construct a fresh (untrained) dlsia TUNet for single-channel denoising.

    ``image_shape`` is fixed to ``(image_size, image_size)`` at construction —
    dlsia's TUNet precomputes exact per-layer tensor sizes from it, so every
    training and inference image must be letterboxed/tiled to this same size
    (same constraint as the segmentation family's TUNet — see
    ``dlsia_runtime.build_model``).

    Fixed at ``in_channels=1, out_channels=1``: denoising reads and writes a
    single raw intensity channel, never the 3-channel RGB convention the
    segmentation family renders — there is no class count to parameterize.
    """
    from dlsia.core.networks.tunet import TUNet  # noqa: PLC0415 — optional dependency

    model = TUNet(
        image_shape=(image_size, image_size),
        in_channels=1,
        out_channels=1,
        depth=depth,
        base_channels=base_channels,
        growth_rate=growth_rate,
        # dlsia's TUNet already defaults `final_activation` to None (verified
        # against dlsia.core.networks.tunet.TUNet.__init__/forward: with no
        # final_activation configured, forward() returns the last conv's raw
        # output unchanged — no softmax/sigmoid is applied). Passed explicitly
        # here anyway, and pinned to None, so a regression head can never
        # silently gain an output-squashing activation if a future dlsia
        # release changed that default — the denoised intensity must come
        # back as an unbounded raw value, not something clamped to (0, 1).
        final_activation=None,
    )
    return model.to(device)


def network_dict(model: Any) -> dict[str, Any]:
    """The full ``{topo_dict, state_dict}`` dlsia uses to reconstruct a TUNet
    (see ``TUNet.save_network_parameters`` / ``TUNetwork_from_file``). Stored
    as-is in this run's ``adapter.pt`` — the denoiser trains from scratch, so
    there's no base/delta split the way LoRA has."""
    return model.save_network_parameters(name=None)


def load_model(state: dict[str, Any], device: str) -> Any:
    """Reconstruct a denoiser TUNet from a saved :func:`network_dict`."""
    from dlsia.core.networks.tunet import TUNet  # noqa: PLC0415

    model = TUNet(**state["topo_dict"])
    model.load_state_dict(state["state_dict"])
    return model.to(device)


def make_forward_fn(model: Any):
    """Return ``forward(batch_images) -> denoised`` for
    :func:`train_common.run_training_loop`.

    TUNet's transposed-conv decoder is symmetric with its encoder, so output
    spatial size already matches input size — no resizing needed. The output
    is the raw regression prediction (denoised intensity): no softmax/argmax
    here, unlike the segmentation family's per-class logits.
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
    """Return ``(gray_uint8_hw) -> float CPU tensor (1,H,W)`` scaled to [0, 1].

    Mirrors ``dlsia_runtime.make_to_tensor_fn``'s uint8->[0,1] convention, but
    for a single intensity channel instead of 3-channel RGB: denoising trains
    on raw grayscale intensity, not the rendered RGB tiles segmentation uses.
    Accepts either ``(H, W)`` or a single-channel ``(H, W, 1)`` array.
    """
    import numpy as np
    import torch

    def _to_tensor(gray_uint8: "np.ndarray") -> Any:
        arr = np.ascontiguousarray(gray_uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        return torch.from_numpy(arr).unsqueeze(0).float() / 255.0

    return _to_tensor
