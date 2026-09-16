# 6. Train a deep model

The **Train** tab fine-tunes a real deep-learning segmentation model (a
**dlsia TUNet**) on this session's annotated samples, then runs it across a
whole volume — a different, slower, more accurate path than the Annotate
tab's own fast in-browser pixel classifier (the **Predict** stage under
Assist/Predict). Use Train when you want a model that generalizes well beyond
the slices you've personally annotated, or that you plan to reuse across
sessions.

!!! note "This is optional"
    Nothing else in the tool requires the Train tab. If the fast pixel
    classifier already gives you good enough results, you never need to open
    this page.

## Is training available?

At the top of the tab, a status banner reports readiness:

- **Ready**: *"torch {version} · {device}"* (e.g. `mps`, `cuda`, or `cpu`) and
  *"dlsia (TUNet): available"*.
- **Unavailable**: *"Training is unavailable on this server."* — the server
  wasn't started with ML dependencies installed (the Docker image
  intentionally omits them to stay lightweight; run via `start_all.sh` with
  `INSTALL_ML=1` — the default on Apple Silicon — on a machine that can
  install them).
- If a job is already running, the banner adds *"A training/inference job is
  currently running."* — only one training or inference job runs at a time.

---

## Training data

The **Training data** panel lists every sample you've annotated **this
session**, with its shape count, as checkboxes:

- If nothing shows up: *"No annotated samples yet this session. Annotate a
  few slices in the Annotate tab, then come back here."*
- Check the samples to include; the header shows *"N of M selected."*

!!! tip
    More annotated slices across more samples generally makes for a better
    model — this is real deep-learning training, not the fast classifier's
    per-sample fit.

### Train on denoised input (optional)

If you've tuned a denoise filter in the Annotate tab's Display panel, a
**"Train on denoised input"** checkbox appears, naming the exact filter (e.g.
*"(median, 40%)"*) rather than offering a second, separate copy of the
controls. This is **off by default** and changes what the model actually
*learns* — unlike every other denoise control in the app, which is
display-only. The setting is recorded on the saved run, and inference
automatically reapplies the same filter, so training and prediction can never
disagree about what the model is looking at.

If no denoise filter is set (or the current one can't be used for training),
the checkbox is disabled with an explanation instead of silently doing
nothing.

---

## Hyperparameters (advanced)

Collapsed by default — sensible defaults are supplied, so most users never
need to open this. When you do:

| Field | What it controls |
| --- | --- |
| **Run name** | Optional label; auto-generated if left blank. |
| **Epochs**, **Learning rate** | Standard training controls. |
| **Batch size** | How many patches process at once. Click **Estimate max** to have the server run real training steps at increasing batch sizes and find the largest that fits your GPU/memory — this needs the device to itself, so it's disabled while another job is running. |
| **Patch/Image size (px)** | The window size the model trains on. |
| **Random flip augmentation** | Cheap data augmentation. |
| **Tile large images** | **On** (default): cuts native-resolution patches with 25% overlap and blends predictions back together — keeps fine detail on images larger than the patch size. **Off**: shrinks each whole slice to the patch size before training — faster, but loses detail on large images. |
| **Depth**, **Base channels**, **Growth rate** | TUNet architecture parameters. |

---

## Start training

Click **Start training**. A progress bar tracks epochs and (once available)
mIoU; cancel is cooperative — a slice already inside a GPU forward pass
finishes before honoring cancel.

## Saved runs

Every completed (or cancelled-but-partial) run appears under **Saved runs**:
model family, timestamp, mIoU, epochs completed, whether it was trained on
denoised input, and its class list. Select a run's radio button to use it for
inference below. Click the trash icon to permanently delete a run's saved
weights (asks for confirmation, naming the run precisely since two runs
trained close together can otherwise look identical).

---

## Inference

With a saved run selected and a sample open (in Browse/Annotate), pick a
scope:

- **Current slice** — just the slice open in the viewer.
- **Slice range** — a start/end slice.
- **All slices** — the whole volume.

Click **Run inference**. Progress shows *"{done}/{total} slices"* with a live
log of region counts per slice, and **the preview slider becomes usable as
soon as the first slice finishes** — it keeps extending as more slices
complete while the job is still running, not just after it finishes.
**Cancel** stops a running job cooperatively.

Once done (or, for slices already predicted, while still running), you get:

- **Import as annotations** — vectorizes the predicted regions into real,
  editable shapes back in the Annotate tab.
- **Write masks to Tiled** *(Tiled sources only)* — writes the prediction
  directly into a `<source>__masks_deep` container in Tiled, independent of
  whatever the Annotate tab's own "Push masks to Tiled" wrote — so the 3D
  view's **Deep** mask layer and **Fast** mask layer can show two genuinely
  different results side by side. See [3D volume view](volume.md#mask-layers).

!!! tip "Large inference jobs can take a while to write"
    A big "Write masks to Tiled" write (hundreds of slices) can legitimately
    take a while over the network. If the 3D view's mask panel shows *"Still
    loading…"* rather than an error, it's still working — a **Check again**
    button appears if it's still not done after a very generous wait.

---

Next: [3D volume view →](volume.md)
