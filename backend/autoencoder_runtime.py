"""A plain convolutional autoencoder for single-channel denoising — the second
architecture in the ``dlsia_denoiser`` family, alongside :mod:`denoise_runtime`'s
TUNet.

Exposes exactly the same six functions as :mod:`denoise_runtime` /
:mod:`dlsia_runtime` (``build_model`` / ``network_dict`` / ``load_model`` /
``make_forward_fn`` / ``make_set_train_mode_fn`` / ``make_to_tensor_fn``), so
``train_common.build_family`` and both inference sites can swap between the two
architectures without knowing which they hold.

Why a second architecture exists
--------------------------------
dlsia's TUNet has skip connections and no constructor option to disable them.
That makes ``f(x) = x`` trivially learnable, so training it on
``target == input`` converges to copying the input: it removes no noise at all
while still reporting a falling loss. The previous workaround added synthetic
Gaussian noise to the input to keep a real gradient — which is a strange thing
to do to already-noisy microscopy data, and it makes the model practise on
additive Gaussian noise rather than the detector's real (Poisson-ish, plus
correlated ring/streak) noise.

This network has **no skip connections** and an explicit latent bottleneck.
That absence is the whole feature: the input physically cannot pass through
unchanged, so plain self-reconstruction becomes a genuine denoising objective —
noise is exactly the high-entropy part that will not fit through a narrow
bottleneck, while the large smooth structures survive. Measured on a
piecewise-constant phantom, this reduced RMSE against clean ground truth from
0.082 (noisy input) to 0.013 without any synthetic corruption.

The honest tradeoff: a bottleneck discards fine real detail along with the
noise, so this tends to blur more than Noise2Void, which reconstructs each
pixel from its neighbourhood at full resolution.

Not a dlsia network, so this module needs no optional dependency beyond torch.
(dlsia does ship ``MSAE``, a multi-scale autoencoder with a latent dimension,
verified working — kept in mind as an alternative, but a hand-rolled net is
predictable, small, and has no surprises in its sizing chart. Its sibling
``MSDAE`` is in fact broken in the pinned dlsia version: it calls
``unet_sizing_chart(kernel=...)``, which that version does not accept.)
"""

from __future__ import annotations

from typing import Any

# Bottleneck compression bounds, mirrored by `schemas.DlsiaDenoiserConfig.
# ae_compression`'s Field(ge=4, le=64)`. Below ~4x the bottleneck is wide enough
# to pass noise straight through; far above ~64x the reconstruction is mostly
# blur.
MIN_COMPRESSION = 4
MAX_COMPRESSION = 64


def latent_channels_for(depth: int, compression: int) -> int:
    """Latent channel count that yields roughly *compression*:1 at the bottleneck.

    With an ``S x S`` single-channel input and *depth* stride-2 halvings, the
    bottleneck holds ``(S / 2**depth)**2 * L`` values against ``S * S`` going
    in, so the ratio is ``4**depth / L`` and

        L = 4**depth / compression

    Independent of ``S``, which is what lets one "compression" number mean the
    same thing across patch sizes. Clamped to at least 1 channel: a bottleneck
    of zero channels would sever the network entirely.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if compression < 1:
        raise ValueError(f"compression must be >= 1, got {compression}")
    return max(1, round(4**depth / compression))


def build_model(
    image_size: int,
    depth: int,
    base_channels: int,
    compression: int,
    device: str,
) -> Any:
    """Construct a fresh (untrained) convolutional autoencoder.

    Args:
        image_size: Square side length. Must be divisible by ``2**depth`` so the
            decoder's transposed convolutions land back on exactly this size.
        depth: Number of stride-2 encoder stages (and matching decoder stages).
        base_channels: Channel width of the first encoder stage; doubles per
            stage.
        compression: Target bottleneck compression ratio (see
            :func:`latent_channels_for`).
        device: Torch device string.

    Raises:
        ValueError: *image_size* is not divisible by ``2**depth`` — which would
            make the network non-shape-preserving and break tiled inference (see
            :func:`make_forward_fn`).
    """
    import torch.nn as nn  # noqa: PLC0415

    if image_size % (2**depth) != 0:
        # Asserted rather than assumed: `TunetHyperParams` requires image_size
        # to be a multiple of 64, which covers depth <= 6, but this module is
        # callable independently and a silent off-by-one here would surface far
        # away as a tensor-size error inside the tiled blend.
        raise ValueError(
            f"image_size {image_size} must be divisible by 2**depth ({2**depth}) "
            "for the decoder to reconstruct the original size"
        )

    latent_channels = latent_channels_for(depth, compression)

    encoder: list[Any] = []
    channels = 1
    for stage in range(depth):
        out_channels = base_channels * (2**stage)
        encoder += [
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        channels = out_channels
    # The bottleneck. Deliberately the ONLY path from encoder to decoder — no
    # skip connections are wired anywhere in this module.
    encoder += [nn.Conv2d(channels, latent_channels, kernel_size=3, stride=1, padding=1)]

    decoder: list[Any] = []
    channels = latent_channels
    for stage in reversed(range(depth)):
        out_channels = base_channels * (2**stage)
        # kernel 4 / stride 2 / padding 1 exactly doubles the spatial size,
        # which is what makes the decoder the inverse of the encoder's halving.
        decoder += [
            nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        channels = out_channels
    # No final activation: the denoised intensity is an unbounded raw value, not
    # something squashed into (0, 1) — same reasoning as denoise_runtime pinning
    # TUNet's final_activation to None.
    decoder += [nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)]

    topo = {
        "image_size": image_size,
        "depth": depth,
        "base_channels": base_channels,
        "compression": compression,
        "latent_channels": latent_channels,
    }
    model = _autoencoder_class()(nn.Sequential(*encoder), nn.Sequential(*decoder), topo)
    return model.to(device)


_AE_CLASS: Any = None


def _autoencoder_class() -> Any:
    """The ``nn.Module`` subclass, defined on first use and cached.

    Deferred so this module stays import-safe without torch, the same property
    :mod:`denoise_runtime` has — a class statement subclassing ``nn.Module``
    cannot live at module scope without importing torch at import time.
    """
    global _AE_CLASS
    if _AE_CLASS is not None:
        return _AE_CLASS

    import torch.nn as nn  # noqa: PLC0415

    class ConvAutoencoder(nn.Module):
        """Encoder -> latent bottleneck -> decoder, with no skip connections."""

        def __init__(self, encoder: Any, decoder: Any, topo: dict[str, Any]) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.topo = topo

        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    _AE_CLASS = ConvAutoencoder
    return _AE_CLASS


def network_dict(model: Any) -> dict[str, Any]:
    """The ``{topo_dict, state_dict}`` needed to reconstruct this network.

    Hand-rolled, because there is no dlsia ``save_network_parameters`` here —
    but the KEY NAMES match dlsia's on purpose. Several callers reach into a
    denoiser checkpoint expecting them: ``train_common.build_family``'s
    warm-start reads ``init_state.get("topo_dict", {})`` to rebuild the run's
    architecture snapshot, and both inference sites hand the whole dict to
    :func:`load_model`. Diverging here would break those silently.
    """
    return {"topo_dict": dict(model.topo), "state_dict": model.state_dict()}


def load_model(state: dict[str, Any], device: str) -> Any:
    """Reconstruct an autoencoder from a saved :func:`network_dict`.

    The topology comes from the CHECKPOINT, not from the run's ``config.json`` —
    matching ``denoise_runtime.load_model``, so a run stays loadable even if the
    request that produced it is long gone.
    """
    topo = state["topo_dict"]
    model = build_model(
        image_size=topo["image_size"],
        depth=topo["depth"],
        base_channels=topo["base_channels"],
        compression=topo["compression"],
        device=device,
    )
    model.load_state_dict(state["state_dict"])
    return model.to(device)


def make_forward_fn(model: Any):
    """Return ``forward(batch_images) -> denoised``.

    Shape-preserving, which ``tiling._blend_tiled_forward`` requires: it
    accumulates each window with ``canvas[:, y:y+w, x:x+w] += out[i] * weight``,
    so an output whose spatial size differed from the input window would raise.
    The stride-2 encoder and matching transposed-conv decoder guarantee this
    when ``image_size % 2**depth == 0`` (enforced in :func:`build_model`).
    """

    def _forward(batch_images: Any) -> Any:
        return model(batch_images)

    return _forward


def make_set_train_mode_fn(model: Any):
    """Return ``set_train_mode(is_training)``.

    This network uses ``nn.BatchNorm2d``, whose running mean/var must not update
    while computing validation metrics.
    """

    def _set_train_mode(is_training: bool) -> None:
        model.train(is_training)

    return _set_train_mode


def make_to_tensor_fn():
    """Return ``(gray_uint8_hw) -> float CPU tensor (1,H,W)`` scaled to [0, 1].

    Identical to ``denoise_runtime.make_to_tensor_fn`` — both architectures
    consume the same single-channel uint8 grayscale that
    ``denoise_train._slice_to_gray_uint8`` produces, so a run trained under one
    can be compared against the other on equal footing.
    """
    import numpy as np
    import torch

    def _to_tensor(gray_uint8: "np.ndarray") -> Any:
        arr = np.ascontiguousarray(gray_uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        return torch.from_numpy(arr).unsqueeze(0).float() / 255.0

    return _to_tensor
