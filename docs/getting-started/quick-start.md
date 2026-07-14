# Quick start

This is the fast path: from a running app to a downloaded, annotated dataset in
a few minutes. Each step links to the in-depth guide if you want more detail.

!!! note "Before you begin"
    Make sure the app is running and open at <http://127.0.0.1:5173>. If not,
    see [Installation](installation.md).

---

## 1. Connect to your data

You land on the **Connect** tab. Choose one of the two modes at the top:

=== "Tiled Server"

    1. Pick a server from the **Server** dropdown.
    2. *(Optional)* expand **Dataset to view (optional)** to point Browse at a specific collection.
    3. Click **Connect**. On success you'll see *"Connected — N sample(s) found"*.
    4. Click **Go to Browse**.

=== "Local Folder"

    1. Enter an absolute path (e.g. `/absolute/path/to/data`) and click **Grant**.
    2. Navigate to the folder you want and choose **Use "…" as dataset folder**.
    3. Click **Connect** — this takes you straight to Browse.

→ More detail: [Connect to data](../guide/connect.md)

---

## 2. Open an image to annotate

On the **Browse** tab, find a sample and open it:

- **Tiled**: use the metadata columns to filter, or click **All samples**, then click the pencil icon (or **Open in Annotate**).
- **Local**: hover a row and click **Annotate**.

→ More detail: [Browse & select](../guide/browse.md)

---

## 3. Add a class

In the **Annotate** tab's left sidebar, find the **CLASSES** panel:

1. Click **+** to add a class.
2. Type a **Class label** (e.g. `pore`), pick a **Color**, and click **Add**.
3. Click the class row to make it active (highlighted).

!!! tip
    If you ingested data with keyword tags, those appear as one-click **quick add** chips.

---

## 4. Draw a mask

Pick a tool from the **TOOLS** panel and draw on the image:

| Tool | Key | How |
| --- | --- | --- |
| **Brush** | ++b++ | Paint freehand strokes. |
| **Polygon** | ++p++ | Click vertices, double-click to finish. |
| **Magic → Smart (AI)** | ++g++ | Drag a box around an object; press ++enter++ to commit. |

Zoom with the mouse wheel and press ++t++ to fit the image to the screen. Undo
with ++cmd+z++ (or ++ctrl+z++).

→ More detail: [Annotate](../guide/annotate.md)

---

## 5. Save your work

Click **Save** in the sidebar to open the **Save version** modal, add your name
and optional notes, then click **Save version**.

!!! note "Autosave vs. Save version"
    The app autosaves a local draft every ~1.5 seconds for crash recovery.
    **Save version** creates an explicit, restorable version on the server.

---

## 6. Export a COCO dataset

1. Click **Export COCO** in the Annotate sidebar.
2. In the **Download COCO Dataset** modal, choose an **Export scope** (e.g. *Current sample only*).
3. Enter your **Annotator** name.
4. Click **Export**, wait for the progress bar, then click **Download .zip**.

You'll get a `.zip` containing rendered images, a COCO JSON file, and per-class
mask PNGs.

→ More detail: [Export & download](../guide/export.md)

---

## Where to go next

<div class="finch-grid" markdown>

<div class="finch-card" markdown>
### :material-draw: Master annotation
Every tool, class management, slices, versions, and QA insights.

[Annotate guide →](../guide/annotate.md){ .md-button }
</div>

<div class="finch-card" markdown>
### :material-keyboard: Keyboard shortcuts
Work faster with the full shortcut reference.

[Shortcuts →](../reference/shortcuts.md){ .md-button }
</div>

</div>
