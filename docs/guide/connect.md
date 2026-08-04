# 1. Connect to data

The **Connect** tab (page title *"Connect to Dataset"*) is where you tell the
app which images to work with. It's the first screen you see on launch.

At the top, a toggle chooses the data source:

- **Tiled Server** — data managed by a Tiled catalog (supports metadata browsing and ingest).
- **Local Folder** — a folder of image files on the machine running the backend.

---

## Tiled Server mode

Tiled mode has two sections.

### Connect to Tiled

1. **Server** — pick a server from the dropdown (it starts on *"— select server —"*).
2. **Dataset to view (optional)** — expand this to point Browse at a specific
   collection. Use the breadcrumb (starting at **root**) and the
   **Browse "…"** button to drill into containers. Leave it blank to let the
   app auto-detect samples.
3. Click **Connect**. While it checks the connection you'll see *"Verifying…"*.
4. On success the status reads *"Connected — N sample(s) found"* and a
   **Go to Browse** button appears.

!!! tip "Re-verify"
    If a connection was interrupted, the button becomes **Re-verify** — click it
    to re-check without changing your selection.

### Load / Ingest Datasets

Below the connection section, a drop zone lets you upload images into Tiled. See
[Ingesting data](#ingesting-data-tiled-only) below.

---

## Local Folder mode

1. Under **Grant access to a folder**, type an absolute path
   (placeholder: `/absolute/path/to/data`) and click **Grant**.
2. Browse into subfolders and choose **Use "…" as dataset folder**.
3. Click **Connect** — this navigates straight to [Browse](browse.md).

Local mode supports the same annotation-status and star-rating filters in
Browse, but has no metadata columns or ingest.

---

## Ingesting data (Tiled only)

The **Ingest data into this server** drop zone uploads new images into the
connected Tiled server.

| Field | What it does |
| --- | --- |
| **Classes / keywords (comma-separated, optional)** | Placeholder `e.g. air, sample, void, pore`. Each entry becomes a pre-created annotation class **and** a searchable tag in Browse. |
| Drop zone | *"Drag an image file or folder of images here"* — or use **choose files** / **choose a folder**. |
| **Save uploaded images to (optional)** | Expandable; placeholder `browse/my_dataset` sets the destination container. |

Supported formats: **TIFF, PNG, JPG, NPY**.

When ingest finishes, you get shortcut buttons such as **Browse this dataset**,
**Open in Annotate**, or **Annotate first image**.

!!! note "Why set keywords at ingest time?"
    Keywords do double duty: they seed your class list in the Annotate tab (so
    you can start drawing immediately) and they become metadata tags you can
    filter on in Browse.

---

Next: [Browse & select a sample →](browse.md)
