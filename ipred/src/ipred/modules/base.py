"""Feature module protocol — typed channel producers for compositions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class ModuleMeta:
    """Catalog entry for a feature module."""

    id: str
    name: str
    description: str
    runtime: str  # onnx | torch | numpy
    params_schema: dict[str, Any] = field(default_factory=dict)
    accepts_input_from: bool = False
    produces_channels: bool = True
    produces_embedding: bool = False


@dataclass
class ChannelBlock:
    """One module's contribution to the feature bank."""

    float_stack: np.ndarray | None = None  # HxWxC
    labels: list[str] = field(default_factory=list)
    emb: np.ndarray | None = None  # Hp×Wp×D dense grid
    emb_meta: dict[str, Any] | None = None
    # Optional 2-D image for downstream encoder nodes (HxW float)
    image_2d: np.ndarray | None = None


@dataclass
class ModuleContext:
    """Shared state while running a composition on one slice."""

    raw: np.ndarray
    gray: np.ndarray
    params: dict[str, Any]
    # Outputs of nodes already run, keyed by node id
    node_outputs: dict[str, ChannelBlock] = field(default_factory=dict)
    # Resolved input image for modules that accept input_from
    input_image: np.ndarray | None = None


class FeatureModule(Protocol):
    """Runnable feature module."""

    meta: ModuleMeta

    def ready(self) -> bool:
        """True when dependencies/weights are available."""
        ...

    def run(self, ctx: ModuleContext) -> ChannelBlock:
        """Produce channels and/or an embedding."""
        ...

    def preview_labels(self, params: dict[str, Any]) -> list[str]:
        """Labels that would appear in the bank for these params (no compute)."""
        ...
