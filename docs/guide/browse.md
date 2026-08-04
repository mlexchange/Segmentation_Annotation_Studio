# 2. Browse & select

The **Browse** tab is where you find a specific image (a *sample*) and open it
for annotation. A banner at the top shows the connected dataset and sample
count, with a **Change** link back to [Connect](connect.md).

!!! note "Not connected?"
    If you haven't connected yet, Browse shows *"No dataset connected."* with a
    **Go to Connect** button.

The layout depends on your data source.

---

## Tiled: the metadata browser

Tiled data uses a Miller-column browser (labelled **Metadata Browser**) — a set
of columns you read left to right. The toolbar controls:

| Control | Purpose |
| --- | --- |
| **Server** | Switch the active Tiled server. |
| **Annotation** | Filter by annotation status: *All samples*, *With annotations*, *Without annotations*. |
| **Refresh** | Reload the current listing. |
| **All samples** | Show every dataset regardless of filters. |
| **Add column** | Add a metadata facet column to filter by. |
| **Open in Annotate** | Open the currently selected sample. |

The columns flow: **facet filters → Samples → (Slices, for volumes) → detail panel**.

### The Samples column

Each sample row can show:

- **Star ratings** (click to rate 1–3 stars). Use the **Min rating** filter
  (**All**, ★, ★★, ★★★) to narrow the list.
- An **annotated** badge if it already has annotations.
- A slice-count badge for multi-slice volumes.
- A pencil icon to open it directly in Annotate.

!!! tip "Ratings drive export scopes"
    Star ratings you set here power the *"★ and above"* export scopes later.
    Rate your best annotations so you can export only those.

### The detail panel

Selecting a sample shows a preview (*"Loading preview…"* while it loads) plus
metadata grouped into sections: **Annotation, Identity, Experiment, Geometry,
Thin Film, Chemistry, Other**. A footer **Open in Annotate** button opens it.

If there's nothing to filter by yet, you'll see hints like *"Click 'Add column'
to filter, or 'All samples' to view everything."*

---

## Local: the flat file list

Local folders show a simple list (**Local Samples**) with the folder path. It
offers the same **Annotation** and **Min rating** filters. Hover a row to reveal
its **Annotate** button.

---

## Opening a sample

When you open a sample (from either browser, or straight from ingest), the app:

1. Loads the image metadata.
2. Restores any local draft autosave for that sample.
3. Seeds the class list from ingest keywords if you have no saved classes yet.
4. Switches to the [Annotate](annotate.md) tab.

---

Next: [Annotate →](annotate.md)
