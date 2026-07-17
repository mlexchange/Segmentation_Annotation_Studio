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
| **Select** | ++s++ | Click or marquee-select shapes; move, edit vertices, run region ops. |
| **Polygon** | ++p++ | Click to place vertices; double-click to finish. |
| **Magnetic** | ++m++ | Click along an edge and it snaps to it (livewire); double-click to finish. |
| **Magic** | ++g++ | AI (SAM) or classic intensity region selection (see below). |
| **Rect** | ++e++ | Drag a rectangle. |
| **Ellipse** | ++l++ | Drag an ellipse. |
| **Brush** | ++b++ | Paint freehand; disconnected dabs become separate shapes. |
| **Fill** | ++f++ | Flood-fill a region by intensity similarity. |
| **Eraser** | ++r++ | Carve pixels out of any shape (see below). |

**Undo** (++cmd+z++) and **Redo** (++cmd+shift+z++) buttons sit in the toolbar,
with up to 200 steps of history.

### How each tool works

All shapes are stored in **image-pixel coordinates**, independent of zoom/pan, so
they stay pixel-accurate at any magnification.

- **Polygon** — each click drops a vertex; the dashed rubber-band line follows the
  cursor. Double-click (or ++enter++ is not used here) closes the ring. While
  drafting, ++cmd+z++/++ctrl+z++ removes the **last vertex** (not the whole shape);
  if you close too early, one ++cmd+z++/++ctrl+z++ reopens the polygon in edit mode
  with its last node removed so you can continue.
- **Magnetic (livewire)** — computes an edge-cost map of the current view and traces
  the **least-cost path** from your last click to the cursor, so the line hugs
  contrast edges. Click to lock each segment; double-click to finish. ++cmd+z++
  pops the last locked node, and (like Polygon) reopens the trace if you just closed
  it.
- **Rectangle / Ellipse** — press-drag to size; released as a shape.
- **Brush** — freehand round-capped strokes. Strokes that touch build up **one**
  shape; a stroke drawn in a **disconnected** area starts a **new** shape, so each
  blob is independently selectable. Press ++n++ to force a new brush instance.
- **Fill** — flood-fills the connected region around your click whose intensity is
  within the **Fill threshold** of the clicked pixel (a paint-bucket by brightness
  similarity).
- **Eraser** — carves pixels out of **any** shape kind (polygon, rectangle, ellipse,
  brush), not just brushes. It is **radius-aware**: the brush disk erases as soon as
  it grazes a shape's edge — the cursor center need not be inside. Erasing a polygon
  **rebuilds its vertices** to match the carved outline (no stray invisible nodes),
  can **split** one shape into several, and can open **holes**. Undo restores the
  shape in one step.

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

- **Brush / Eraser**: a **Brush radius (px)** slider (1–500) with a matching
  number box for exact values.
- **Eraser scope**: radios — **Erase selected class** (default) or **Erase all
  classes** (carve whatever visible shape the stroke crosses).
- **Select**: **Select this class** / **Select all classes** radios; ++cmd+a++/++ctrl+a++
  selects every shape on the slice per that scope.
- **Fill**: a **Fill threshold** slider (0–100%).
- **Global** (below the class list / in Tools): **Annotation opacity**, **Clip to
  other classes** (on by default), and **Merge overlapping same class**.

!!! info "Clip and Merge use exact geometry"
    **Clip to other classes** subtracts neighbouring classes from a new shape so
    regions tile **flush with no gap**; **Merge overlapping same class** unions a
    new shape with overlapping same-class shapes. Both use true polygon boolean
    operations at full resolution, so **existing vertices are preserved** — only
    the seam/cut changes, and repeated edits don't erode a region.

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

To add a vertex to a polygon, double-click its edge with the Select tool;
double-click a vertex to delete it. Drag the white outer vertices or the amber
hole vertices to reshape. ++cmd+a++/++ctrl+a++ selects every shape on the slice
(scoped by the **Select this class / all classes** radios). Press ++n++ to start
a new brush instance.

---

## Display (view-only)

The **DISPLAY** panel changes how the image *looks* while you work — it does
**not** change exported pixels (except that it defines the render used for
export images). Controls include **Brightness**, **Contrast**, a histogram
levels window, **Colormap** (gray, viridis, magma, inferno), **Gamma**,
**CLAHE**, and **Sharpen**. Use the reset button to restore defaults.

- **CLAHE** is *adaptive* (local) contrast: it equalizes each image tile's
  histogram with a clip limit and blends the tiles, so faint local features pop
  without blowing out the whole frame. When on, it is applied **before** the
  brightness/contrast/levels chain, and the levels histogram updates to reflect
  it. (It replaced the older global "Auto-contrast" stretch.)
- The display preprocessors also change what the **tools see** — SAM, the magic
  wand, and the magnetic edge map all operate on the enhanced view — but never
  affect the exported pixels.

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

When your annotations are ready, click **Export** to open the download
dialog. That's covered in full on the next page.

Next: [Annotation guide →](reference-guide.md) or jump to [Export & download →](export.md)
