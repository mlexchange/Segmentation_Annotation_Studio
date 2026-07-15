# 5. Export & download

This is the payoff: turning your annotations into a downloadable dataset. You can
export as a **COCO dataset** (for SAM3 fine-tuning) or in the **DINOv3 / Lightly**
semantic-segmentation layout. Either way the primary path produces a `.zip` file
with images and masks.

There are two related actions in the same dialog:

- **Export** → build a COCO dataset and **download a `.zip`** to your computer.
- **Push masks to Tiled** → write masks back into Tiled (server-side, **no** local download).

---

## Downloading a COCO dataset

### Step 1 — Prepare (optional but recommended)

Before exporting, you can shape *what* gets included:

- **Mark negative slices** in the [Slice panel](annotate.md#navigating-slices)
  with **Mark as negative** — these export as images with zero annotations.
- **Rate samples** with stars in [Browse](browse.md#the-samples-column) — ratings
  drive the *"★ and above"* export scopes.

### Step 2 — Open the export dialog

In the Annotate sidebar, click **Export**. This opens the **Download
Dataset** modal (where you also choose the export format).

### Step 3 — Choose a scope

Under **Export scope**, pick which samples to include:

| Scope | Includes |
| --- | --- |
| **Current slice only** | Just the slice open in the viewer. |
| **Current sample only** | All annotations for the open sample *(default)*. |
| **All annotated samples** | Every sample with at least one annotation this session. |
| **★ and above** | Annotated samples rated 1 star or higher. |
| **★★ and above** | Annotated samples rated 2 stars or higher. |
| **★★★ only** | Only samples rated 3 stars. |

The dialog previews how many samples match, e.g. *"3 sample(s) will be
exported."* (or *"No samples match."*).

### Step 4 — Set options

- **Format** — choose the dataset layout:
    - **COCO (SAM3)** *(default)* — RLE masks + images, for SAM3 fine-tuning.
    - **DINOv3 / Lightly** — `images/` + `masks/` label PNGs with matching
      filenames + a `classes.json`, for [LightlyTrain semantic
      segmentation](https://docs.lightly.ai/train/stable/semantic_segmentation.html).
      See [the layout below](#dinov3-lightly-format).
- **Annotator** — your name (placeholder *"Your name (recorded in the export)"*).
  It's stamped into the export folder name, the COCO `info` block, and
  `manifest.json`.
- **Include polygon copy in COCO** — optional checkbox (COCO format only). RLE
  masks are always exact; enable this only if an external viewer needs polygon
  geometry (it's slower).

### Step 5 — Export and download

1. Click **Export**. The button shows **Exporting…** and a progress bar tracks
   *"{done}/{total} slices"* with a live log.
2. On success you'll see a message like *"Saved to {path} (Tiled). Use Download
   .zip to save images + masks to your computer."*
3. Click **Download .zip** — your browser's save dialog picks where to store it.

!!! tip "Filename"
    The zip is named after the annotator and source, e.g.
    `yourname__samplename_20260714T....zip`.

---

## What's inside the `.zip`

The export is split into `train/`, `valid/`, and `test/` folders (an 80/10/10
split is applied automatically to slices marked `auto`, using a fixed seed for
reproducibility). Each split contains:

```text
manifest.json
train/
  _annotations.coco.json          # COCO JSON: info, images, categories, annotations
  <source>_0001.png               # rendered slice image (uses your Display settings)
  masks/
    semantic/<source>_0001.png    # label map: 0 = background, 1..N = class index
    <class_name>/<source>_0001.png  # per-class binary mask (0 / 255)
    legend.json                   # [{ id, name, color }, ...]
valid/
  ...
test/
  ...
```

### The COCO JSON

Each annotation record contains:

| Field | Notes |
| --- | --- |
| `segmentation` | **Always** present — compressed RLE (pycocotools), exact including holes. |
| `segmentation_poly` | Only if you checked **Include polygon copy in COCO**. |
| `bbox`, `area`, `category_id` | Standard COCO fields. |
| `iscrowd` | Always `0`. |

The `info` block also records the **render** settings used to produce the PNG
images, so exports are reproducible. `categories[].name` is the SAM3 concept
phrase (the class label).

## DINOv3 / Lightly format

Choosing **DINOv3 / Lightly** in Step 4 writes the layout expected by
LightlyTrain's semantic-segmentation trainer instead of COCO:

```text
classes.json                    # { "0": "background", "1": "<class>", ... }
manifest.json
train/
  images/<source>_0001.png      # rendered slice (uses your Display settings)
  masks/<source>_0001.png       # single-channel label map, pixel = class index (0 = bg)
val/                            # our "valid" split → Lightly's "val"
  images/... masks/...
test/                           # kept as a held-out split (optional to use)
  images/... masks/...
```

- **Matching filenames**: each `masks/<name>.png` shares the exact stem of its
  `images/<name>.png`, which is how Lightly pairs them.
- **Masks are index label maps** (grayscale PNG, `0` = background, `1..N` = class
  index — the same indices as `classes.json`), not colorized.
- Point Lightly's config at the folders, e.g.
  `data = { "train": {"images": ".../train/images", "masks": ".../train/masks"},
  "val": {...}, "classes": ".../classes.json" }` (add `"ignore_classes": [0]` to
  skip background).

!!! note "Where the files are written on the server"
    Before you download, the backend writes the dataset under `EXPORT_ROOT`
    (or `~/data/exports`). The **Download .zip** button streams that folder to
    your browser as a zip.

---

## Alternative: Push masks to Tiled

If your source is a **Tiled** dataset, you can write masks back into Tiled
instead of (or in addition to) downloading.

1. In the same **Download Dataset** modal, click **Push masks to Tiled**.
2. The backend rasterizes the selected scope and writes stacked mask volumes
   into a sibling Tiled container named `<source>__masks`:
    - `semantic` — a `uint8 (n, H, W)` class-index volume.
    - `<class_name>` — a `uint8 (n, H, W)` binary (0/255) volume per class.
3. Success message: *"Masks merged into Tiled: {container} ({n} slices total,
   {updated} updated)."*

!!! warning
    **Push masks to Tiled** is disabled for local-folder sources (its tooltip
    says *"Only available for Tiled sources"*). It produces **no** downloadable
    file — the result lives in Tiled.

---

## What is *not* a data export

A couple of other buttons say "export" but do something different:

| Button | Location | What it produces |
| --- | --- | --- |
| **Save version** | Annotate sidebar | A server-side version snapshot (not a dataset). See [Saving](annotate.md#saving-your-work). |
| **Export** | Reference tab | `annotation-guide.json` — class descriptions only. See [Annotation guide](reference-guide.md). |

The **only** way to download annotated segmentation data is the COCO **Export →
Download .zip** flow described above.

---

## Importing a dataset back

The backend can also re-import a COCO dataset directory into the editor (via its
import endpoint), useful for reviewing or continuing a previous export. This is
a backend capability rather than a prominent UI button.

---

Done! You've installed the tool, annotated images, and exported a COCO dataset.
For a faster refresher, see the [Quick start](../getting-started/quick-start.md),
or speed up your workflow with the [keyboard shortcuts](../reference/shortcuts.md).
