"""DINOv3 backbone loading + hand-rolled LoRA for the Train tab.

Model *code* is vendored (``backend/vendor/dinov3/``) rather than loaded via
``torch.hub.load()`` against upstream — ``torch.hub`` executes the real
repo's ``hubconf.py``, which eagerly imports the *entire* research codebase
(detectors, segmentors, depthers, eval utilities) just to expose a backbone
constructor, pulling in ``torchvision``, ``torchmetrics``, ``termcolor``, and
more as a side effect (confirmed empirically — this is not a hypothetical).
Vendoring just the ~15 self-contained files needed for
``DinoVisionTransformer`` (see ``backend/vendor/dinov3/hub_backbones.py``'s
module docstring) avoids that entirely and needs only ``torch``. Every
vendored file records its exact upstream source/commit in a header comment.

Weights are always loaded from a local ``.pth`` file the operator has already
placed on disk — this app never downloads a checkpoint from Meta's servers.

LoRA is hand-rolled (~1 module) rather than pulling in ``peft`` — this repo
keeps optional ML dependencies to exactly what's needed (``torch``), and LoRA
itself is a small, well-understood wrapper around ``nn.Linear``.

Distributed under Meta's DINOv3 License (research/non-commercial terms) —
this module only ever loads a checkpoint the operator has already placed on
disk; it never affects DINOv3's own licensing (see backend/vendor/dinov3/LICENSE.md).
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Pinned commit the vendored files under backend/vendor/dinov3/ were copied
# from (main, 2026-07-15) — see each file's header for its exact source URL.
DINOV3_REF = "6876159a11b4df116f30f667f8c9888617df0751"
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _ensure_vendor_on_path() -> None:
    """Make ``import dinov3...`` resolve to backend/vendor/dinov3/."""
    vendor_str = str(_VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


# arch -> embed_dim, the backbone's token width (needed to size the linear
# segmentation head). Patch size is 16 for all of these (all our supported
# archs are the "...16" variants). ConvNeXt is out of scope (per product
# decision) — the ViT family, including the 7B giant, is fully supported.
VIT_ARCHS: dict[str, dict[str, int]] = {
    "vits16": {"embed_dim": 384},
    "vits16plus": {"embed_dim": 384},
    "vitb16": {"embed_dim": 768},
    "vitl16": {"embed_dim": 1024},
    "vith16plus": {"embed_dim": 1280},
    "vit7b16": {"embed_dim": 4096},
}
PATCH_SIZE = 16

# Longest-match-first so "vits16plus" isn't truncated to "vits16" — though the
# `_pretrain_` anchor immediately after the arch group makes backtracking find
# the right alternative regardless; ordering here is defensive, not load-bearing.
_ARCH_ALTERNATION = "|".join(sorted(VIT_ARCHS, key=len, reverse=True))
_CHECKPOINT_RE = re.compile(
    rf"^dinov3_(?P<arch>{_ARCH_ALTERNATION})_pretrain_(?P<weights>[a-z0-9]+)-(?P<hash>[0-9a-f]{{8}})\.pth$"
)


def models_dir() -> Path:
    """Server-owned directory holding pretrained DINOv3 checkpoints."""
    from train_common import models_dir as _models_dir  # avoid a circular import at module load

    return _models_dir()


def list_checkpoints() -> list[dict[str, Any]]:
    """Scan :func:`models_dir` for recognised DINOv3 checkpoint files.

    Files that don't match ``dinov3_<arch>_pretrain_<weights>-<hash>.pth`` for
    one of :data:`VIT_ARCHS`, or that live in a subdirectory, are silently
    skipped — this never raises so the capability probe stays best-effort.
    """
    d = models_dir()
    if not d.exists():
        return []
    results: list[dict[str, Any]] = []
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        m = _CHECKPOINT_RE.match(f.name)
        if not m:
            continue
        arch = m.group("arch")
        results.append(
            {
                "file_name": f.name,
                "arch": arch,
                "weights": m.group("weights"),
                "embed_dim": VIT_ARCHS[arch]["embed_dim"],
                "size_bytes": f.stat().st_size,
            }
        )
    return results


def resolve_checkpoint(arch: str, file_name: str) -> Path:
    """Validate *file_name* against the checkpoint naming convention and arch,
    then return its path under :func:`models_dir` — never a path outside it.

    Raises:
        HTTPException: 400 if the name is unsafe/doesn't match, or the arch
            in the filename disagrees with the requested *arch*; 404 if the
            file doesn't exist.
    """
    if "/" in file_name or "\\" in file_name or "\x00" in file_name:
        raise HTTPException(400, "Invalid checkpoint file name")
    m = _CHECKPOINT_RE.match(file_name)
    if not m or m.group("arch") != arch:
        raise HTTPException(400, f"Checkpoint {file_name!r} does not match arch {arch!r}")
    path = (models_dir() / file_name).resolve()
    if not path.is_relative_to(models_dir()) or not path.exists():
        raise HTTPException(404, f"Checkpoint not found: {file_name!r}")
    return path


def load_backbone(arch: str, ckpt_path: Path, device: str) -> Any:
    """Load a frozen DINOv3 backbone for *arch* from a local checkpoint.

    Constructs the architecture from the vendored model code (see module
    docstring), then loads the checkpoint's state dict directly — entirely
    offline, no network access of any kind.
    """
    import torch  # noqa: PLC0415 — optional dependency

    _ensure_vendor_on_path()
    from dinov3.hub_backbones import build_untrained_backbone  # noqa: PLC0415

    try:
        backbone = build_untrained_backbone(arch, ckpt_path.name, device)
        # map_location="cpu" (not `device`): load_state_dict copies each tensor
        # into the model's already-allocated device parameters one at a time, so
        # loading straight onto `device` briefly holds the checkpoint AND the
        # model in device memory at once — for the largest arch (vit7b16, ~27GB
        # of weights) that is ~54GB of peak device memory just to load it.
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict, strict=True)
    except Exception as exc:  # noqa: BLE001 — surfaced as a job error, not a crash
        raise RuntimeError(f"Failed to load DINOv3 backbone {arch!r}: {exc}") from exc

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear:
    """Mixin-free LoRA wrapper factory — see :func:`_make_lora_linear`.

    Kept as a factory function (not a class defined at import time) so this
    module stays importable without torch; the real ``nn.Module`` subclass is
    created lazily the first time :func:`inject_lora` runs.
    """


def _make_lora_linear_class() -> type:
    import math

    import torch
    import torch.nn as nn

    class _LoRALinear(nn.Module):
        """Wraps a frozen ``nn.Linear`` with a trainable low-rank delta:
        ``y = base(x) + (x @ A^T @ B^T) * (alpha / rank)``.

        ``B`` is zero-initialized so the adapter starts as an exact no-op —
        training begins from the frozen backbone's original behaviour.
        """

        def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.rank = rank
            self.scale = alpha / rank
            # Expose these so the wrapped module stays a transparent drop-in
            # replacement for nn.Linear — DINOv3's own attention code reads
            # `self.qkv.in_features` directly (not just `self.qkv(x)`), so
            # simply delegating calls via forward() isn't enough.
            self.in_features = base.in_features
            self.out_features = base.out_features
            device, dtype = base.weight.device, base.weight.dtype
            self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            base_out = self.base(x)
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            return base_out + lora_out * self.scale

    return _LoRALinear


_LORA_TARGET_SUFFIXES = (".attn.qkv", ".attn.proj")


def inject_lora(backbone: Any, rank: int, alpha: float) -> dict[str, Any]:
    """Wrap every attention qkv/proj ``nn.Linear`` in *backbone* with LoRA.

    Only ``nn.Linear`` (and subclasses — DINOv3's masked-bias qkv variant
    included) whose dotted module name ends in one of
    :data:`_LORA_TARGET_SUFFIXES` are wrapped; every other parameter stays
    frozen. Returns the injected modules by dotted name.

    Raises:
        ValueError: if zero modules matched — almost certainly means the
            backbone's internal naming changed upstream; fail loudly rather
            than silently train nothing.
    """
    import torch.nn as nn

    LoRALinearImpl = _make_lora_linear_class()
    injected: dict[str, Any] = {}

    targets = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(_LORA_TARGET_SUFFIXES)
    ]
    if not targets:
        available = [n for n, m in backbone.named_modules() if isinstance(m, nn.Linear)][:20]
        raise ValueError(
            "No LoRA injection targets found (expected names ending in "
            f"{_LORA_TARGET_SUFFIXES}); available nn.Linear modules include: {available}"
        )

    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        base_linear = getattr(parent, attr)
        wrapped = LoRALinearImpl(base_linear, rank=rank, alpha=alpha)
        setattr(parent, attr, wrapped)
        injected[name] = wrapped

    return injected


def lora_state_dict(lora_modules: dict[str, Any]) -> dict[str, Any]:
    """Extract just the trainable LoRA A/B tensors (not the frozen base)."""
    return {
        name: {"lora_A": module.lora_A.detach().cpu(), "lora_B": module.lora_B.detach().cpu()}
        for name, module in lora_modules.items()
    }


def load_lora_state_dict(lora_modules: dict[str, Any], state: dict[str, Any]) -> int:
    """Load previously-saved LoRA A/B tensors back into injected modules.

    Returns the number of modules actually loaded. Names missing from *state*
    are skipped rather than raising — but callers should check the count, since
    a total mismatch (wrong arch) otherwise looks identical to a clean load
    while leaving every module at its freshly-initialised value.
    """
    import torch

    n_loaded = 0
    for name, module in lora_modules.items():
        saved = state.get(name)
        if saved is None:
            continue
        with torch.no_grad():
            module.lora_A.copy_(saved["lora_A"].to(module.lora_A.device))
            module.lora_B.copy_(saved["lora_B"].to(module.lora_B.device))
        n_loaded += 1
    return n_loaded


def build_model(
    arch: str,
    ckpt_path: Path,
    n_classes: int,
    rank: int,
    alpha: float,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the backbone, inject LoRA, and attach a linear segmentation head.

    Returns ``(backbone, head, lora_modules)``. Trainable parameters are the
    LoRA A/B tensors plus the head — call ``build_trainable_params`` to
    collect them for the optimizer.
    """
    import torch.nn as nn

    backbone = load_backbone(arch, ckpt_path, device)
    lora_modules = inject_lora(backbone, rank=rank, alpha=alpha)
    embed_dim = VIT_ARCHS[arch]["embed_dim"]
    head = nn.Conv2d(embed_dim, n_classes, kernel_size=1).to(device)
    return backbone, head, lora_modules


def build_trainable_params(head: Any, lora_modules: dict[str, Any]) -> list[Any]:
    """Collect the parameters that should actually receive gradients."""
    params = list(head.parameters())
    for module in lora_modules.values():
        params.extend([module.lora_A, module.lora_B])
    return params


def load_model(
    model_config: dict[str, Any],
    hyperparams: dict[str, Any],
    adapter_state: dict[str, Any],
    n_classes: int,
    device: str,
) -> tuple[Any, Any]:
    """Rebuild a fine-tuned backbone+head for inference from a saved run.

    Args:
        model_config: This run's ``config.json["model_config"]`` (``arch``,
            ``checkpoint``).
        hyperparams: This run's ``config.json["hyperparams"]`` (``lora_rank``,
            ``lora_alpha``) — must match training exactly to reconstruct the
            same LoRA shapes.
        adapter_state: This run's saved ``adapter.pt`` (``{"lora", "head"}``).
        n_classes: Number of classes the head was trained with
            (``len(config.json["classes"])``).
    """
    import torch.nn as nn

    arch = model_config["arch"]
    ckpt_path = resolve_checkpoint(arch, model_config["checkpoint"])
    backbone = load_backbone(arch, ckpt_path, device)
    lora_modules = inject_lora(backbone, rank=hyperparams["lora_rank"], alpha=hyperparams["lora_alpha"])
    load_lora_state_dict(lora_modules, adapter_state["lora"])

    embed_dim = VIT_ARCHS[arch]["embed_dim"]
    head = nn.Conv2d(embed_dim, n_classes, kernel_size=1).to(device)
    head.load_state_dict(adapter_state["head"])
    head.eval()
    return backbone, head


def make_forward_fn(backbone: Any, head: Any):
    """Return ``forward(batch_images) -> logits`` for :func:`train_common.run_training_loop`.

    ``batch_images`` is ``(B, 3, H, W)``; the backbone is frozen (only LoRA
    deltas + the head train), but stays in ``eval()`` mode throughout — DINOv3
    has no dropout/batchnorm in the frozen path that training mode would
    otherwise toggle, and eval mode avoids any surprise from stateful norm
    layers picking up batch statistics from a tiny fine-tune batch.
    """
    import torch.nn.functional as F

    def _forward(batch_images: Any) -> Any:
        tokens = backbone.get_intermediate_layers(batch_images, n=1, reshape=True, norm=True)[0]
        logits = head(tokens)
        return F.interpolate(logits, size=batch_images.shape[-2:], mode="bilinear", align_corners=False)

    return _forward


def make_to_tensor_fn():
    """Return ``(rgb_uint8_hwc) -> float CPU tensor (3,H,W)`` with ImageNet normalisation.

    Stays on CPU deliberately: :func:`train_common.run_training_loop` batches
    several of these per-sample tensors with ``torch.stack`` before doing one
    device transfer for the whole batch, rather than many small H2D copies.
    """
    import numpy as np
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _to_tensor(rgb_uint8: "np.ndarray") -> Any:
        t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(2, 0, 1).float() / 255.0
        return (t - mean) / std

    return _to_tensor
