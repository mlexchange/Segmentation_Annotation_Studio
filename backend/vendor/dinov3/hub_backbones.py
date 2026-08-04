# Minimal replacement for facebookresearch/dinov3's dinov3/hub/backbones.py.
#
# Architecture kwargs below are copied verbatim from each `dinov3_<arch>()`
# function in upstream's dinov3/hub/backbones.py @ 6876159a11b4df116f30f667f8c9888617df0751
# (https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py).
# Distributed under Meta's DINOv3 License (see backend/vendor/dinov3/LICENSE.md).
#
# What's deliberately NOT vendored: `torch.hub`'s own weight-download/URL
# machinery (`Weights` enum, `_make_dinov3_vit_model_url`,
# `_safe_load_state_dict_from_url`) — this app always loads weights from a
# local checkpoint file the operator already placed on disk (see
# dino_runtime.load_backbone), so none of that is exercised. Only the
# architecture-dimension kwargs (embed_dim, depth, ffn_layer, etc.) are needed,
# and those are reproduced exactly as upstream defines them per arch.

from __future__ import annotations

from dinov3.models.vision_transformer import DinoVisionTransformer

# Shared across every ViT arch below (verbatim from upstream's per-arch kwargs).
_COMMON_KWARGS = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    pos_embed_rope_base=100,
    pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32",
    qkv_bias=True,
    drop_path_rate=0.0,
    layerscale_init=1.0e-05,
    norm_layer="layernormbf16",
    ffn_bias=True,
    proj_bias=True,
    n_storage_tokens=4,
    mask_k_bias=True,
)

# arch -> kwargs that differ per architecture (embed_dim/depth/heads/ffn).
_ARCH_KWARGS: dict[str, dict] = {
    "vits16": dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, ffn_layer="mlp"),
    "vits16plus": dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=6, ffn_layer="swiglu"),
    "vitb16": dict(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4, ffn_layer="mlp"),
    "vitl16": dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, ffn_layer="mlp"),
    "vith16plus": dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6.0, ffn_layer="swiglu"),
    # ViT-7B is the one arch that overrides a _COMMON_KWARGS value (qkv_bias),
    # and the only one using swiglu64. `ffn_ratio=3` with SwiGLU's 2/3 rule and
    # 64-alignment gives the checkpoint's 8192-wide FFN: int(3*4096*2/3) = 8192.
    "vit7b16": dict(
        embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3, qkv_bias=False, ffn_layer="swiglu64"
    ),
}

# vitl16 is the one arch upstream ships two structurally different pretrained
# variants for (LVD1689M vs SAT493M) — SAT493M sets `untie_global_and_local_cls_norm
# =True`. Match on the checkpoint filename's 8-hex-char hash suffix, exactly
# like upstream's `dinov3_vitl16()` does for a string `weights` argument.
_VITL16_SAT493M_HASH = "eadcf0ff"


def build_untrained_backbone(arch: str, checkpoint_file_name: str, device) -> DinoVisionTransformer:
    """Construct a randomly-initialised DinoVisionTransformer for *arch*.

    Caller loads real weights afterward (see dino_runtime.load_backbone) —
    this only fixes the architecture's shape, matching upstream's per-arch
    hardcoded kwargs exactly so a checkpoint's state_dict loads with
    ``strict=True``.
    """
    if arch not in _ARCH_KWARGS:
        raise ValueError(f"Unsupported DINOv3 arch: {arch!r}")
    kwargs = {**_COMMON_KWARGS, **_ARCH_KWARGS[arch], "device": device}
    if arch == "vitl16" and _VITL16_SAT493M_HASH in checkpoint_file_name:
        kwargs["untie_global_and_local_cls_norm"] = True
    # Upstream's `dinov3_vit7b16()` sets this unconditionally, for both the
    # LVD1689M and SAT493M weights — and indeed both checkpoints carry
    # `local_cls_norm.*`, which only exists when the norms are untied.
    if arch == "vit7b16":
        kwargs["untie_global_and_local_cls_norm"] = True
    return DinoVisionTransformer(**kwargs)
