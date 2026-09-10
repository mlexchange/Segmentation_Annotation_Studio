# Using the tool

This section is a complete walkthrough of the Segmentation Annotation Studio interface,
tab by tab. If you just want the fastest path to a dataset, read the
[Quick start](../getting-started/quick-start.md) instead.

## The Finch shell

Every screen shares the same layout:

- **Left sidebar** — the icon rail that switches between tabs.
- **Header** — the ALS logo and the app title, *"Segmentation Annotation Tool"*.
- **Main area** — the current tab's content.

The sidebar contains six tabs by default:

| Tab | Icon | What it's for |
| --- | --- | --- |
| **Connect** | plug | Choose a Tiled dataset or local folder. |
| **Browse** | magnifier | Filter and pick a sample to annotate. |
| **Reference** | book | Write class descriptions (the annotation guide). |
| **Annotate** | pencil | Draw masks, manage versions, and export. |
| **3D** | cube | View the reconstruction and mask layers in 3D. |
| **Train** | brain | Train a dlsia deep-learning segmentation model and run it on a whole volume. |

A small indicator in the header shows live Tiled connection status (green
"Tiled connected" / red "Tiled disconnected" — click it to jump back to
Connect if it drops).

!!! tip "Customize which tabs you see"
    Click the floating **Customize Layout** button (top-right) to open
    *"Customize Your Layout"* and hide or show tabs. Your choice is saved in
    the browser. Click **Apply Changes** to confirm.

## Recommended reading order

1. [Connect to data](connect.md) — get the app pointed at your images.
2. [Browse & select](browse.md) — find and open a sample.
3. [Annotate](annotate.md) — the core drawing workflow.
4. [Annotation guide](reference-guide.md) — keep multi-annotator projects consistent.
5. [Export & download](export.md) — produce a COCO dataset.
6. [Train a deep model](train.md) — optional, for a volume-wide model beyond the fast in-tab classifier.
7. [3D volume view](volume.md) — optional, for viewing the reconstruction and mask layers in 3D.
