"""Module registry — discover and list FeatureModules."""

from __future__ import annotations

from typing import Any

from ipred.modules.base import FeatureModule, ModuleMeta
from ipred.modules.clahe_mod import ClaheModule
from ipred.modules.pca_mod import PcaModule
from ipred.modules.skimage_mod import SkimageMultiscaleModule
from ipred.modules.slimsam_mod import SlimSamModule
from ipred.modules.tomojepa_mod import TomoJepaModule

_REGISTRY: dict[str, FeatureModule] | None = None


def _build_registry() -> dict[str, FeatureModule]:
    mods: list[FeatureModule] = [
        SkimageMultiscaleModule(),
        ClaheModule(),
        SlimSamModule(),
        TomoJepaModule(),
        PcaModule(),
    ]
    return {m.meta.id: m for m in mods}


def get_registry() -> dict[str, FeatureModule]:
    """Return module id → instance map."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_module(module_id: str) -> FeatureModule:
    """Lookup module or raise KeyError."""
    reg = get_registry()
    if module_id not in reg:
        raise KeyError(f"unknown module {module_id}")
    return reg[module_id]


def list_module_catalog() -> list[dict[str, Any]]:
    """JSON-serializable catalog for GET /modules."""
    out: list[dict[str, Any]] = []
    for mod in get_registry().values():
        meta: ModuleMeta = mod.meta
        # Prefer ONNX in catalog only when graph matches default input_size
        runtime = meta.runtime
        if meta.id == "tomojepa":
            from ipred import tomojepa_onnx

            onnx = TomoJepaModule()._onnx_path({"weights_id": "mark25"})
            if (
                onnx is not None
                and tomojepa_onnx.onnx_matches_input_size(onnx, 512)
            ):
                runtime = "onnx"
        out.append(
            {
                "id": meta.id,
                "name": meta.name,
                "description": meta.description,
                "runtime": runtime,
                "ready": bool(mod.ready()),
                "accepts_input_from": meta.accepts_input_from,
                "produces_channels": meta.produces_channels,
                "produces_embedding": meta.produces_embedding,
                "params_schema": meta.params_schema,
            }
        )
    return out
