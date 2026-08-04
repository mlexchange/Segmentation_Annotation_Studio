# Minimal replacement for facebookresearch/dinov3's dinov3/models/__init__.py.
#
# Upstream's version imports `convnext` and defines training-config-driven
# `build_model()` helpers this app never calls — this repo only vendors the
# ViT backbone (`vision_transformer.py`), so this file intentionally stays
# empty rather than pulling in code (and its own import chain) we don't need.
