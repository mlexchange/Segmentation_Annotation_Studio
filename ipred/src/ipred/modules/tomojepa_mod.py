"""TomoJEPA / Mark encoder module (torch; ONNX when available)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ipred import feature_setups, tomojepa_embed
from ipred.modules.base import ChannelBlock, ModuleContext, ModuleMeta


def _resolve_weights(params: dict[str, Any]) -> str | None:
    """Resolve Mark25/11 weights from params."""
    explicit = params.get("weights_path")
    if explicit and str(explicit) != "(missing)":
        p = Path(str(explicit)).expanduser()
        if p.is_file():
            return str(p.resolve())
    weights_id = str(params.get("weights_id") or "mark25").lower()
    if weights_id in ("mark11", "tomojepa11", "11"):
        return feature_setups.resolve_tomojepa_weights_path(
            "tomojepa11.pth", env_var="TOMOJEPA11_WEIGHTS"
        )
    return feature_setups.resolve_tomojepa_weights_path(
        "tomojepa25.pth", env_var="TOMOJEPA_WEIGHTS"
    )


def _encoder_name(params: dict[str, Any], path: str | None) -> str:
    wid = str(params.get("weights_id") or "").lower()
    if wid in ("mark11", "tomojepa11", "11"):
        return "mark11"
    if path and "tomojepa11" in Path(path).name:
        return "mark11"
    return "mark25"


def _weights_format(name: str) -> str:
    if name == "mark11":
        return feature_setups.WEIGHTS_TOMOJEPA_MARK11
    return feature_setups.WEIGHTS_TOMOJEPA_MARK25


class TomoJepaModule:
    """Dense TomoJEPA embeddings (torch; prefers ONNX when present)."""

    meta = ModuleMeta(
        id="tomojepa",
        name="TomoJEPA",
        description="Mark25/Mark11 dense projector (torch or ONNX)",
        runtime="torch",
        accepts_input_from=True,
        produces_channels=False,
        produces_embedding=True,
        params_schema={
            "weights_id": {"type": "string", "default": "mark25", "enum": ["mark25", "mark11"]},
            "weights_path": {"type": "string", "default": None},
            "input_size": {"type": "integer", "default": 512},
            "resize": {"type": "boolean", "default": True},
        },
    )

    def ready(self) -> bool:
        if self._onnx_path({"weights_id": "mark25"}) or self._onnx_path(
            {"weights_id": "mark11"}
        ):
            return True
        for fname, env in (
            ("tomojepa25.pth", "TOMOJEPA_WEIGHTS"),
            ("tomojepa11.pth", "TOMOJEPA11_WEIGHTS"),
        ):
            p = feature_setups.resolve_tomojepa_weights_path(fname, env_var=env)
            if p and tomojepa_embed.encoder_available(p):
                return True
        return False

    def preview_labels(self, params: dict[str, Any]) -> list[str]:
        del params
        return []

    def _onnx_path(self, params: dict[str, Any]) -> Path | None:
        import os

        explicit = params.get("onnx_path")
        if explicit and Path(str(explicit)).expanduser().is_file():
            return Path(str(explicit)).expanduser().resolve()
        wid = str(params.get("weights_id") or "mark25").lower()
        fname = "tomojepa11.onnx" if wid in ("mark11", "11") else "tomojepa25.onnx"
        env = "TOMOJEPA11_ONNX" if "11" in fname else "TOMOJEPA_ONNX"
        env_p = os.getenv(env)
        if env_p and Path(env_p).expanduser().is_file():
            return Path(env_p).expanduser().resolve()
        # modules/tomojepa_mod.py → …/ipred/src/ipred/modules → parents[3]=ipred/
        here = Path(__file__).resolve()
        for c in (
            here.parents[3] / "models" / fname,
            here.parents[4] / "ipred" / "models" / fname,
        ):
            if c.is_file():
                return c.resolve()
        return None

    def run(self, ctx: ModuleContext) -> ChannelBlock:
        src = ctx.input_image if ctx.input_image is not None else ctx.gray
        p = ctx.params
        resize = bool(p.get("resize", True))
        input_size = int(p.get("input_size", 512))
        onnx_path = self._onnx_path(p)
        use_onnx = False
        if onnx_path is not None:
            from ipred import tomojepa_onnx

            use_onnx = tomojepa_onnx.onnx_matches_input_size(
                onnx_path, input_size
            )
        if use_onnx and onnx_path is not None:
            from ipred import tomojepa_onnx

            emb, orig_hw, reshaped_hw = tomojepa_onnx.encode_dense_embeddings(
                src,
                weights_path=str(onnx_path),
                input_size=input_size,
                resize=resize,
            )
            runtime = "onnx"
            path_s = str(onnx_path)
        else:
            wpath = _resolve_weights(p)
            if not tomojepa_embed.encoder_available(wpath):
                raise ValueError(
                    "TomoJEPA weights not available (torch/.pth or ONNX)"
                )
            emb, orig_hw, reshaped_hw = tomojepa_embed.encode_dense_embeddings(
                src,
                weights_path=wpath,
                input_size=input_size,
                resize=resize,
            )
            runtime = "torch"
            path_s = wpath
        enc_name = _encoder_name(p, path_s)
        meta = {
            "orig_hw": list(orig_hw),
            "reshaped_hw": list(reshaped_hw),
            "weights_path": path_s,
            "encoder": enc_name,
            "input_size": input_size,
            "resize": resize,
            "weights_format": _weights_format(enc_name),
            "runtime": runtime,
            "model_input_range": [-1.0, 1.0],
            "intensity_norm": "minmax_01_then_linear_m11",
        }
        return ChannelBlock(emb=emb, emb_meta=meta)
