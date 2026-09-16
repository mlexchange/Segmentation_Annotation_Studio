"""Mark25 / TomoJEPA dense embeddings for feature banks."""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_DEFAULT_INPUT_SIZE = 512
_PATCH = 16


def torch_available() -> bool:
    """True when torch and timm can be imported."""
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_weights_path(explicit: str | None = None) -> Path | None:
    """Resolve TomoJEPA weights from explicit path, env, or repo default."""
    if explicit and explicit != "(missing)":
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p
    env = os.getenv("TOMOJEPA_WEIGHTS")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    here = Path(__file__).resolve()
    # .../repo/ipred/src/ipred → repo/ipred/models
    candidates = [
        here.parents[2] / "models" / "tomojepa25.pth",
        here.parents[3] / "ipred" / "models" / "tomojepa25.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def encoder_available(weights_path: str | None = None) -> bool:
    """True when torch/timm are importable and weights exist."""
    if not torch_available():
        return False
    return resolve_weights_path(weights_path) is not None


def load_checkpoint_state(path: Path | str) -> dict[str, Any]:
    """Load ``ckpt['net']`` with optional ``module.`` prefix stripped."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "net" not in ckpt:
        raise ValueError(f"expected TomoJEPA ckpt with 'net' key: {path}")
    return {k.replace("module.", ""): v for k, v in ckpt["net"].items()}


def build_encoder(**kwargs: Any):
    """Build uninitialized DINOv3ViTEncoder (requires torch/timm)."""
    from ipred.tomojepa_encoder import DINOv3ViTEncoder

    return DINOv3ViTEncoder(**kwargs)


@lru_cache(maxsize=4)
def _cached_model(path_str: str):
    import torch

    from ipred.tomojepa_encoder import DINOv3ViTEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint_state(path_str)
    # Mark25 dense proj is 64-D; Mark11 is 256-D — read from checkpoint.
    proj_w = state.get("proj.8.weight")
    if proj_w is None:
        raise ValueError(f"checkpoint missing proj.8.weight: {path_str}")
    proj_dim = int(proj_w.shape[0])
    net = DINOv3ViTEncoder(
        proj_dim=proj_dim, img_size=_DEFAULT_INPUT_SIZE, in_chans=1, pretrained=False
    )
    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)
    return net, device


def _pad_to_multiple(h: int, w: int, multiple: int = _PATCH) -> tuple[int, int]:
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    return h + ph, w + pw


def _gray_unit_interval(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3].astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif a.ndim == 2:
        gray = a.astype(np.float64)
    else:
        raise ValueError(f"unsupported array shape {a.shape}")
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return ((gray - lo) / (hi - lo)).astype(np.float32)


def to_minus_one_one(gray01: np.ndarray) -> np.ndarray:
    """Linear map from ``[0, 1]`` to ``[-1, 1]`` (Mark25 / TomoJEPA convention)."""
    return (np.asarray(gray01, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)


def encode_dense_embeddings(
    arr: np.ndarray,
    *,
    weights_path: str | None = None,
    input_size: int = _DEFAULT_INPUT_SIZE,
    resize: bool = True,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return ``(emb Hp×Wp×64, orig_hw, reshaped_hw)``.

    When ``resize`` is True (default), the grayscale slice is resized to a
    square ``input_size×input_size``. When False, native resolution is kept and
    only padded to a multiple of 16 (Mark25 ``dynamic_img_size``).

    Intensity pipeline: min-max to ``[0, 1]`` (CLAHE already in that range when
    passed from ``clahe_encoder_v1``), then linear shift to ``[-1, 1]`` before
    the network.
    """
    import torch

    path = resolve_weights_path(weights_path)
    if path is None:
        raise FileNotFoundError("TomoJEPA weights (.pth) not found")
    if not torch_available():
        raise ImportError(
            "TomoJEPA requires torch and timm — install with: pip install -e '.[torch]'"
        )

    gray = _gray_unit_interval(arr)
    oh, ow = int(gray.shape[0]), int(gray.shape[1])
    if resize:
        size = max(int(input_size), _PATCH)
        resized = np.asarray(
            PILImage.fromarray(gray, mode="F").resize(
                (size, size), PILImage.BILINEAR
            ),
            dtype=np.float32,
        )
        rh, rw = size, size
    else:
        resized = gray.astype(np.float32, copy=True)
        rh, rw = oh, ow

    pad_h, pad_w = _pad_to_multiple(rh, rw, _PATCH)
    if pad_h != rh or pad_w != rw:
        canvas = np.zeros((pad_h, pad_w), dtype=np.float32)
        canvas[:rh, :rw] = resized
        resized = canvas
    else:
        pad_h, pad_w = rh, rw

    # TomoJEPA expects intensities in [-1, 1] after CLAHE / unit-interval prep.
    model_in = to_minus_one_one(resized)

    x = torch.from_numpy(np.array(model_in, dtype=np.float32, copy=True)[None, None, None])
    with _model_lock:
        net, device = _cached_model(str(path))
        x = x.to(device)
        with torch.inference_mode():
            _glob, dense, _feats = net(x)
        dense_np = dense.detach().cpu().numpy()[0, 0]  # [L, 64]

    grid_h = pad_h // _PATCH
    grid_w = pad_w // _PATCH
    if dense_np.shape[0] != grid_h * grid_w:
        side = int(round(dense_np.shape[0] ** 0.5))
        grid_h = grid_w = side
    emb = dense_np.reshape(grid_h, grid_w, dense_np.shape[-1]).astype(np.float32)
    # reshaped_hw is the coordinate frame covered by the emb grid (incl. pad).
    return emb, (oh, ow), (pad_h, pad_w)
