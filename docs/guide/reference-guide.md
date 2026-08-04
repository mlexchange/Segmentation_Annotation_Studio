# 4. Annotation guide

The **Reference** tab (title *"Annotation Guide"*) is an optional but valuable
step for consistency, especially when several people annotate the same dataset.
It lets you describe each class in plain language so everyone labels the same
way.

!!! note "Requires an open sample"
    Reference works on the currently open dataset. If none is open, you'll see a
    **Go to Browse** button.

The tab explains its purpose at the top:

> *"Describe each class so annotators label consistently. These classes appear
> as one-click suggestions in the Annotate tab. Saved automatically with this
> dataset."*

---

## What you can do here

| Feature | Description |
| --- | --- |
| **Task notes** | A free-text area for overall labeling instructions. |
| **Per-class description** | A **Class label**, color, and a description (placeholder: *"What is this class? How does it look? When should it (not) be used?"*). |
| **Example crops** | Generate labelled example images per class. |
| **Generate example crops from** | Choose **Current annotation** or a saved version, then click **Generate**. |
| **Add class** | Add a new class to the guide. |
| **Import current classes** | Pull in the classes you already created in Annotate. |
| **Import** / **Export** | Load or save the guide as an `annotation-guide.json` file. |

---

## How it connects to Annotate

- Classes defined here appear as **quick add** chips in the Annotate
  [Classes panel](annotate.md#classes).
- Their descriptions and example crops show up behind the :material-information:
  **Info** icon on each class row.
- The guide is saved automatically with the dataset — no manual save needed.

!!! tip "Sharing across a team"
    Use **Export** to download `annotation-guide.json` and share it, then have
    teammates use **Import** to load the same class definitions. This keeps a
    multi-annotator project aligned.

!!! warning "This is not a data export"
    The **Export** button here downloads only the *guide* (class descriptions and
    example crops) — not segmentation masks or a training dataset. For that, see
    [Export & download](export.md).

---

Next: [Export & download →](export.md)
