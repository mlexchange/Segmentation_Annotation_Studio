# 3. Annotate

The **Annotate** tab is the heart of the tool. Its left sidebar stacks a series
of panels; the right side is the image canvas.

!!! note "Nothing loaded?"
    If you open Annotate without a sample, you'll see *"No sample loaded. Pick a
    sample to annotate."* and a **Go to Browse** button. Open a sample from
    [Browse](browse.md) first.

The sidebar panels, top to bottom:

**Classes → Tools → Display → Slice → Cross-slice → Measure → Save / Insights / Export**

---

## Classes

Everything you draw belongs to a **class**. The **CLASSES** panel manages them.

- **Add a class**: click **+**, enter a **Class label**, pick a **Color**, click **Add**.
- **Quick add** chips let you add common classes in one click (from your
  annotation guide, or defaults like `air`, `sample`, `void`, `pore`,
  `background`, `substrate`).
- **Activate** a class by clicking its row — new shapes use the active class.

Each class row has actions:

| Action | Effect |
| --- | --- |
| Click row | Make it the active class. |
| :material-information: Info | Show its guide description and example crops. |
| :material-eye: / :material-eye-off: | **Show class** / **Hide class** on the canvas. |
| :material-pencil: Pencil | Edit the class: rename it inline **and** change its color via the swatch (now a color picker). Click the :material-check: check to finish. |
| :material-delete: Trash | Delete the class (asks to confirm — this removes its annotations). |

!!! tip
    Press ++1++–++9++ to activate the first nine classes by their order in the list.

If you have no classes, the Tools panel warns *"Add a class above to start
annotating."*

### Colorblind-safe colors

The **CLASSES** panel has a **Colorblind-safe colors** checkbox. When checked,
**newly added** classes are colored from a colorblind-safe palette (the
Okabe–Ito scheme, extended with Paul Tol's muted colors). Uncheck it to color
new classes from the standard palette.

!!! note
    This is a display/authoring preference stored in your browser. To avoid
    silently changing already-annotated datasets, it only affects the colors
    auto-assigned to **new** classes — your existing classes keep their colors.
    You can always recolor a class yourself when adding it, and class colors are
    recorded per class in saved versions and exports as usual.

---

## Tools

The **TOOLS** panel is your drawing palette. Each tool has a keyboard shortcut.

| Tool | Key | What it does |
| --- | --- | --- |
| **Pan** | hold ++space++ | Drag the canvas around. |
| **Select** | ++s++ | Click or marquee-select shapes; edit vertices. |
| **Polygon** | ++p++ | Click to place vertices; double-click to finish. |
| **Magnetic** | ++m++ | Click along an edge to trace it; double-click to finish. |
| **Magic** | ++g++ | AI or classic region selection (see below). |
| **Rect** | ++e++ | Drag a rectangle. |
| **Ellipse** | ++l++ | Drag an ellipse. |
| **Brush** | ++b++ | Paint freehand strokes. |
| **Fill** | ++f++ | Flood-fill by intensity threshold. |
| **Eraser** | ++r++ | Erase from brush shapes. |

**Undo** (++cmd+z++) and **Redo** (++cmd+shift+z++) buttons sit in the toolbar,
with up to 200 steps of history.

### The Magic tool (Smart AI + Classic)

The **Magic** tool has two engines, chosen with a toggle:

=== "Smart (AI)"

    Uses a SAM (Segment Anything) model running in your browser.

    - On first use you'll see *"Loading model… (first time only)"*, then
      *"Encoding slice…"*, then a readiness message like *"SAM ready · WebGPU"*
      (or *"CPU"*).
    - **Drag a box** around an object, or **click** it, then commit with the
      **Add** button or ++enter++.
    - ++shift+click++ adds to the selection; ++alt+click++ (++opt+click++ on Mac)
      excludes a region.
    - Tune it with the **Detail** buttons (*auto*, *fine*, *medium*, *coarse*),
      the **Tightness** and **Edge smoothing** sliders, and the **Avoid
      other-class regions** checkbox.

    !!! note
        Changing **Display** brightness/contrast re-encodes the slice for SAM,
        so adjust the display first if you plan to use Smart mode heavily.

=== "Classic"

    A classic intensity-based magic wand — no model required.

    - Choose **Connected** (contiguous region) or **All similar** (every matching pixel).
    - Tune **Tolerance**, **Edge stop** (contiguous only), and **Edge smoothing**.

### Tool-specific and global options

- **Brush / Eraser**: a **Brush radius (px)** slider (1–500).
- **Fill**: a **Fill threshold** slider (0–100%).
- **Global**: **Annotation opacity**, **Clip to other classes**, and **Merge
  overlapping same class**.

---

## Working on the canvas

- **Zoom**: mouse wheel zooms toward the cursor; the current zoom shows bottom-right.
- **Fit**: press ++t++ to fit the image to the screen.
- **Pan**: hold ++space++ and drag (releases back to your previous tool).

### Selecting and editing shapes

With the **Select** tool active, the canvas hint reads *"Click a shape · drag a
box to select many · Shift-drag to add more · Shift-click to toggle"*. When
shapes are selected, a selection toolbar appears:

| Control | Action |
| --- | --- |
| **Class:** dropdown | Reassign selected shapes to another class. |
| **Copy** / **Paste** | ++cmd+c++ / ++cmd+v++. |
| **Invert** | ++i++ — invert a single selected shape. |
| **Delete** | ++delete++ or ++backspace++. |
| **Thickness** | Adjust a selected brush shape's stroke width. |
| **Region:** ops | **Merge**, **Grow**, **Shrink**, **Remove islands** — then **Apply** (or **Cancel**). |

To add a vertex to a polygon, double-click its edge with the Select tool. Press
++n++ to start a new brush instance.

---

## Display (view-only)

The **DISPLAY** panel changes how the image *looks* while you work — it does
**not** change exported pixels (except that it defines the render used for
export images). Controls include **Brightness**, **Contrast**, a histogram
levels window, **Colormap** (gray, viridis, magma, inferno), **Gamma**,
**Auto-contrast**, and **Sharpen**. Use the reset button to restore defaults.

---

## Navigating slices

For multi-slice volumes, the **Slice** panel shows *"Slice N / total"* with:

- A slider and **Previous slice** / **Next slice** buttons (++left++ / ++right++, or ++x++ for next).
- A **Jump to annotated…** dropdown listing slices that already have shapes.
- **Mark as negative** — flags a slice as a deliberate negative example
  (exported as an image with zero annotations).

### Cross-slice

For volumes, the **CROSS-SLICE** panel acts on the active class. Its main action
is **Copy '{class}' → next slice**, handy for propagating a mask through a stack.

---

## Measure

Select one or more regions and the **MEASURE** panel reports **Regions, Area,
Perimeter, Centroid, Bounds**, and **Intensity (raw)** (mean ± SD, min/max,
pixel count). If you set a **Pixel size** and unit (default **µm**),
measurements convert to physical units.

---

## Saving your work

The tool saves at two levels:

- **Autosave** — a local draft is saved roughly every 1.5 seconds for crash
  recovery. This is *not* a versioned save.
- **Save version** — an explicit, restorable snapshot stored on the server.

Click **Save** (the button reads **Saved** or **Saving…** depending on state) to
open the **Save version** modal:

1. Review the **Preview** thumbnail.
2. Fill in **Who annotated this** (placeholder *"Your name or initials"*).
3. Add optional **Notes**.
4. Click **Save version**.

### Version history

The version-count button opens **Version History**. For each version you can
**Preview** (scrub versions on the canvas without changing your work) or
**Restore** (load it into the editor to save as a new version). A preview bar
across the canvas shows *"Previewing v{N} (latest)"* with **Restore this
version** / **Exit** and a version slider.

---

## Insights (quality checks)

Click **Insights** to open **Dataset Insights**, a QA dashboard showing:

- Counts: **Annotated slices, Negative slices, Empty unmarked, Samples this session**.
- A **Class balance** bar chart.
- **Quality checks** flags: *Tiny regions, Self-intersecting polygons,
  Cross-class overlaps, Empty, not marked negative*.

Click any flag to jump straight to the offending slice or region on the canvas.

---

## Export

When your annotations are ready, click **Export COCO** to open the download
dialog. That's covered in full on the next page.

Next: [Annotation guide →](reference-guide.md) or jump to [Export & download →](export.md)
