#!/usr/bin/env python3
"""Export TomoJEPA Mark25/Mark11 checkpoints to ONNX.

Usage:
  python -m ipred.scripts.export_tomojepa_onnx
  python -m ipred.scripts.export_tomojepa_onnx --weights ipred/models/tomojepa11.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_tomojepa_onnx")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        action="append",
        default=None,
        help="Path to .pth (repeatable). Default: both tomojepa25/11 if present.",
    )
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as weights)",
    )
    args = parser.parse_args(argv)

    from ipred.tomojepa_onnx import export_tomojepa_onnx

    here = Path(__file__).resolve()
    models = here.parents[2] / "models"
    weights = args.weights
    if not weights:
        weights = []
        for name in ("tomojepa25.pth", "tomojepa11.pth"):
            p = models / name
            if p.is_file():
                weights.append(str(p))
    if not weights:
        logger.error("no weights found")
        return 1

    for w in weights:
        pth = Path(w)
        out_dir = args.out_dir or pth.parent
        onnx_path = out_dir / (pth.stem + ".onnx")
        export_tomojepa_onnx(pth, onnx_path, input_size=args.input_size)
        logger.info("wrote %s", onnx_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
