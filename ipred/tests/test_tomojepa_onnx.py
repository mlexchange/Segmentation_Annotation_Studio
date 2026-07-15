"""TomoJEPA ONNX encode smoke (skips if onnx file missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipred import tomojepa_onnx

ONNX25 = Path(__file__).resolve().parents[1] / "models" / "tomojepa25.onnx"
needs_onnx = pytest.mark.skipif(
    not ONNX25.is_file() or not tomojepa_onnx.onnx_available(),
    reason="tomojepa25.onnx not available",
)


@needs_onnx
def test_onnx_encode_dense_shape() -> None:
    arr = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(64, 48)
    emb, orig, reshaped = tomojepa_onnx.encode_dense_embeddings(
        arr, weights_path=str(ONNX25), input_size=128
    )
    assert orig == (64, 48)
    assert reshaped == (128, 128)
    assert emb.shape == (8, 8, 64)
