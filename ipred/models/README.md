# ipred/models

Local model weights for iPred's feature modules. Gitignored — nothing here ships
with the repo.

## SlimSAM (ONNX)

Auto-vendored by the Magic tool's fetch script — no separate step needed here.
`slimsam_mod.py` resolves it from `frontend/public/models/slimsam-77-uniform/onnx/vision_encoder.onnx`:

```
node frontend/scripts/fetch-sam-model.mjs
```

## TomoJEPA (Mark25 / Mark11)

**Not fetched automatically — no public source exists.** These are private
trained checkpoints (~365 MB each), not published on HuggingFace or anywhere
else; `github.com/phzwart/tomojepa` is the training toolkit only and ships no
checkpoint files.

To enable TomoJEPA-based compositions, place the checkpoint(s) here:

```
ipred/models/tomojepa25.pth   # Mark25
ipred/models/tomojepa11.pth   # Mark11
```

or point at them via `TOMOJEPA_WEIGHTS` / `TOMOJEPA11_WEIGHTS` env vars.
ONNX exports (`tomojepa25.onnx` / `tomojepa11.onnx`, or `TOMOJEPA_ONNX` /
`TOMOJEPA11_ONNX`) are preferred when present — see
`python -m ipred.scripts.export_tomojepa_onnx`.

Until a checkpoint is present, `GET /modules` reports `tomojepa.ready == false`
and the frontend greys out compositions that depend on it.
