#!/usr/bin/env python3
"""
Upload up to N image + sidecar JSON pairs from generated_data into the local Tiled catalog.

Requires a running Tiled server with a writable catalog (see tiled/config.yml) and scopes
that allow create/write (same API key as the browse app).

Example:
  cd /path/to/test_skills/backend
  export TILED_URI=http://127.0.0.1:8010
  export TILED_API_KEY=...
  python scripts/seed_generated_data_to_tiled.py --source /Users/ahexemer/Desktop/generated_data --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
try:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND_DIR / ".env")
except ImportError:
    pass


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _load_pairs(source: Path) -> list[tuple[Path, Path]]:
    """Return (png_path, txt_path) for stems that have both files."""
    pairs: list[tuple[Path, Path]] = []
    for png in sorted(source.rglob("*.png")):
        txt = png.with_suffix(".txt")
        if txt.is_file():
            pairs.append((png, txt))
    return pairs


def _ensure_container(parent, key: str, metadata: dict | None = None):
    try:
        return parent[key]
    except Exception:
        return parent.create_container(key=key, metadata=metadata or {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Tiled with generated_data PNG+JSON pairs.")
    default_source = os.environ.get("SEED_GENERATED_DATA_DIR") or str(
        (_BACKEND_DIR.parent / "generated_data").resolve()
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(default_source),
        help=(
            "Root folder to scan for *.png with matching *.txt JSON sidecars "
            "(default: <repo>/generated_data, override with SEED_GENERATED_DATA_DIR)"
        ),
    )
    parser.add_argument("--limit", type=int, default=5, help="Max number of datasets to upload")
    parser.add_argument(
        "--uri",
        default=os.environ.get("TILED_URI", "http://127.0.0.1:8010"),
        help="Tiled server base URI",
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("TILED_API_KEY")
            or os.environ.get("TILED_LOCAL_API_KEY")
            or ""
        ),
        help="API key (or set TILED_API_KEY in backend/.env)",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"Source directory not found: {args.source}", file=sys.stderr)
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required (pip install pillow)", file=sys.stderr)
        return 1

    from tiled.client import from_uri

    pairs = _load_pairs(args.source)[: max(0, args.limit)]
    if not pairs:
        print(f"No PNG+TXT pairs found under {args.source}", file=sys.stderr)
        return 1

    kwargs = {}
    if args.api_key.strip():
        kwargs["api_key"] = args.api_key.strip()

    client = from_uri(args.uri.rstrip("/"), **kwargs)
    browse = _ensure_container(client, "browse", metadata={"kind": "browse_seed"})
    target = _ensure_container(browse, "generated_data", metadata={"source": str(args.source.resolve())})

    uploaded = 0
    for png_path, txt_path in pairs:
        stem = png_path.stem
        try:
            raw = txt_path.read_text(encoding="utf-8")
            meta = _json_safe(json.loads(raw))
        except Exception as exc:
            print(f"Skip {txt_path}: invalid JSON ({exc})", file=sys.stderr)
            continue

        try:
            img = Image.open(png_path).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
        except Exception as exc:
            print(f"Skip {png_path}: cannot read image ({exc})", file=sys.stderr)
            continue

        try:
            target.write_array(arr, key=stem, metadata=meta, dims=["y", "x", "channel"])
            print(f"Wrote browse/generated_data/{stem} ({arr.shape})")
            uploaded += 1
        except Exception as exc:
            print(f"Skip {stem}: Tiled write failed ({exc})", file=sys.stderr)

    print(f"Done. Uploaded {uploaded} dataset(s). Browse API will use browse/generated_data when present.")
    return 0 if uploaded else 2


if __name__ == "__main__":
    raise SystemExit(main())
