# 7. 3D volume view

The **3D** tab renders the open dataset's reconstruction and mask layers
directly in 3D, streamed straight from Tiled — there is no export step. It
uses WebGPU, so it needs a modern Chromium-based browser (Chrome or Edge
113+) served over a secure context (`https://`, or `localhost`/`127.0.0.1`).

!!! note "This is optional"
    Nothing else in the tool requires the 3D view — it's there for inspecting
    results in context, not for annotating.

## No volume yet?

A dataset ingested as individual 2D slices has no 3D pyramid until one is
built. If you see **"No 3D volume for this dataset yet"**, the panel shows the
source shape/dtype and the pyramid it will build, then a **Build 3D volume**
button. This reads every slice once; full resolution is **not** copied — it
stays where it is, and the 3D view never loads a level too large for the GPU.

If a volume already exists but needs refreshing (e.g. after a fidelity setting
changed), a small **Rebuild volume** button sits in the bottom-left corner of
the 3D view itself.

---

## The render HUD

A docked sidebar on the right controls how the volume looks:

| Panel | Controls |
| --- | --- |
| **Data** | Resolution/LOD picker, voxel size, high-res ROI streaming. |
| **Transfer Function** | Colormap, opacity curve (drag/add/remove points on the histogram), color range (with **Auto**/**Equalize**), clip limit. Also has a **Bands** mode for multiple independent intensity ranges. |
| **Slices** | Axis-aligned slice planes through the volume. |
| **Crop** | An ROI crop box. |
| **Measure** | On-canvas measurement between points in the volume. |
| **Annotations** | The built-in mask-layer controls (this app also has its own, described below — either produces the identical visual result). |
| **Presets** | Save/apply/delete named transfer-function presets. |

Pan with space+drag, shift/middle/right-wheel zoom to the cursor; ++p++/++ctrl+click++
picks a point, ++l++ cycles LOD, ++o++ toggles open/collapsed.

---

## Mask layers

A small panel in the top-left corner controls two independent, fixed mask
layers:

- **Fast (iPred)** — the Annotate tab's quick pixel classifier / manual
  annotations.
- **Deep (dlsia)** — a [Train tab](train.md) model's output.

Each has its own **Load** button, and (once loaded) an opacity slider and
per-class visibility toggles (click a class row to show/hide it).

### Live vs. Tiled (Fast slot only)

The Fast slot defaults to **Live** mode — it rasterizes your **current**
annotation shapes directly in the browser and loads instantly, with **no
Tiled sync required first**. Switch to **Tiled** to load the precise,
backend-rasterized result from `Push masks to Tiled` in the Annotate tab
instead. The Deep slot has no Live equivalent — it's always the saved output
of a from-scratch-trained model, loaded from the `<source>__masks_deep`
container [Train's inference panel](train.md#inference) writes.

!!! tip "Loading a large Deep mask can take a while"
    A real Tiled-backed mask (hundreds of slices, freshly written) can
    legitimately take a while to fetch over the network. The **Load** button
    shows *"Loading…"* then *"Still loading…"* rather than failing outright —
    if it's genuinely still not done after a very generous wait, a **Check
    again** button appears rather than making you re-fetch from scratch.

### Getting here from elsewhere

- Annotate's iPred panel has its own **Push to Tiled** (stays on the tab) and
  **View in 3D** (navigates here, loading the Fast layer) — split into two
  independent actions so pushing doesn't force you into the 3D view, and
  viewing doesn't force a fresh push.
- Train's inference panel's **Write masks to Tiled** writes the Deep layer
  independently, so Fast and Deep can show two genuinely different results
  side by side for comparison.

---

## Isolating a feature by intensity (Sampler → 3D bridge)

If you've used the Annotate tab's Sampler/Threshold-lasso tool (see
[Annotate → The Magic tool](annotate.md#the-magic-tool-smart-ai-classic) for
the sibling Magic tool, or the Threshold Brush's own **Set band from a
region**) to fit an intensity band around a small, materially-distinct
feature, its readout gains a **View band in 3D** button (for a plain,
non-projected fit only). Clicking it isolates that intensity band here:
opaque inside the band, transparent outside, via the same **Transfer
Function** the HUD's own panel controls.

!!! note "This isolates by intensity, not by traced region"
    It shows *everything* in that density range across the whole volume, not
    only the specific spot you traced — genuinely useful when the feature's
    density is distinct from its surroundings (the same property the 2D
    threshold tool already relies on), less useful if it overlaps
    similar-density material elsewhere.

---

## Connection issues

If the dataset can't be resolved because Tiled dropped (rather than because
no volume has been built yet), the view shows **"Couldn't reach the Tiled
server"** with a **Go to Connect** button, instead of leaving you on an
indefinite spinner.

---

Back to [Train a deep model ←](train.md), or return to the
[recommended reading order](index.md#recommended-reading-order).
