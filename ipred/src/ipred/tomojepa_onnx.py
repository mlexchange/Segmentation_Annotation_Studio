"""TomoJEPA ONNX Runtime backend (optional; falls back to torch)."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from ipred.tomojepa_embed import _gray_unit_interval, to_minus_one_one

logger = logging.getLogger(__name__)

_session_lock = threading.Lock()
_PATCH = 16
_DEFAULT_INPUT_SIZE = 512


def onnx_available() -> bool:
    """True when onnxruntime can be imported."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=4)
def _session_for(path_str: str):
    import onnxruntime as ort

    return ort.InferenceSession(path_str, providers=["CPUExecutionProvider"])


def fixed_spatial_size(weights_path: str | Path) -> int | None:
    """Return fixed square H=W from ONNX input, or None when dynamic / unknown."""
    if not onnx_available():
        return None
    path = Path(weights_path)
    if not path.is_file():
        return None
    with _session_lock:
        sess = _session_for(str(path.resolve()))
        shape = sess.get_inputs()[0].shape
    if not shape:
        return None
    # [N,C,D,H,W] or [N,C,H,W]
    dims = list(shape)
    if len(dims) >= 2:
        h, w = dims[-2], dims[-1]
        if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
            return int(h)
    return None


def onnx_matches_input_size(
    weights_path: str | Path, input_size: int
) -> bool:
    """True when ONNX is dynamic or fixed spatial size equals ``input_size``."""
    fixed = fixed_spatial_size(weights_path)
    if fixed is None:
        return True
    return fixed == int(input_size)


def _pad_to_multiple(h: int, w: int, multiple: int = _PATCH) -> tuple[int, int]:
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    return h + ph, w + pw


def encode_dense_embeddings(
    arr: np.ndarray,
    *,
    weights_path: str,
    input_size: int = _DEFAULT_INPUT_SIZE,
    resize: bool = True,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Run exported TomoJEPA ONNX; return ``(emb Hp×Wp×D, orig_hw, reshaped_hw)``.

    Expects ONNX input name ``input`` with shape ``[1,1,1,H,W]`` or ``[1,1,H,W]``
    float32 in ``[-1, 1]``, and output ``dense`` as ``[1,1,L,D]`` or ``[1,L,D]``.

    When the graph has a fixed spatial size, that size is used (must match
    ``input_size`` when ``resize`` is True, else ValueError).
    """
    if not onnx_available():
        raise ImportError("onnxruntime required for TomoJEPA ONNX")
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    fixed = fixed_spatial_size(path)
    effective_size = int(input_size)
    if fixed is not None:
        if resize and fixed != effective_size:
            raise ValueError(
                f"ONNX fixed size {fixed} != input_size {effective_size}; "
                "use torch fallback or re-export ONNX"
            )
        effective_size = fixed

    gray = _gray_unit_interval(arr)
    oh, ow = int(gray.shape[0]), int(gray.shape[1])
    if resize:
        size = max(effective_size, _PATCH)
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
        if fixed is not None and (rh != fixed or rw != fixed):
            raise ValueError(
                f"array {rh}x{rw} does not match ONNX fixed size {fixed}"
            )

    pad_h, pad_w = _pad_to_multiple(rh, rw, _PATCH)
    if pad_h != rh or pad_w != rw:
        canvas = np.zeros((pad_h, pad_w), dtype=np.float32)
        canvas[:rh, :rw] = resized
        resized = canvas
    else:
        pad_h, pad_w = rh, rw

    model_in = to_minus_one_one(resized)
    x5 = np.array(model_in, dtype=np.float32, copy=True)[None, None, None]
    x4 = np.array(model_in, dtype=np.float32, copy=True)[None, None]

    with _session_lock:
        sess = _session_for(str(path.resolve()))
        in_meta = sess.get_inputs()[0]
        in_name = in_meta.name
        # Choose rank matching exported model
        shape = in_meta.shape
        rank = len(shape) if shape else 5
        feed = {in_name: x5 if rank >= 5 else x4}
        outs = sess.run(None, feed)
    dense = np.asarray(outs[0], dtype=np.float32)
    # Squeeze to [L, D]
    while dense.ndim > 2:
        if dense.shape[0] == 1:
            dense = dense[0]
        else:
            break
    if dense.ndim != 2:
        raise ValueError(f"unexpected ONNX dense shape {dense.shape}")

    grid_h = pad_h // _PATCH
    grid_w = pad_w // _PATCH
    if dense.shape[0] != grid_h * grid_w:
        side = int(round(dense.shape[0] ** 0.5))
        grid_h = grid_w = side
    emb = dense.reshape(grid_h, grid_w, dense.shape[-1]).astype(np.float32)
    return emb, (oh, ow), (pad_h, pad_w)


def export_tomojepa_onnx(
    pth_path: str | Path,
    onnx_path: str | Path,
    *,
    input_size: int = 512,
    opset: int = 17,
) -> Path:
    """Export ``DINOv3ViTEncoder`` checkpoint to ONNX (fixed square size)."""
    import torch

    from ipred.tomojepa_embed import build_encoder, load_checkpoint_state

    pth = Path(pth_path)
    out = Path(onnx_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint_state(pth)
    proj_dim = int(state["proj.8.weight"].shape[0])
    net = build_encoder(proj_dim=proj_dim, img_size=input_size, in_chans=1)
    net.load_state_dict(state, strict=True)
    net.eval()

    class _DenseOnly(torch.nn.Module):
        def __init__(self, enc: Any) -> None:
            super().__init__()
            self.enc = enc

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _g, dense, _f = self.enc(x)
            return dense

    wrapper = _DenseOnly(net)
    dummy = torch.zeros(1, 1, 1, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["dense"],
        opset_version=opset,
        dynamo=False,
    )
    logger.info("exported TomoJEPA ONNX → %s (proj_dim=%s)", out, proj_dim)
    return out.resolve()
