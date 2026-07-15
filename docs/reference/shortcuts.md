# Keyboard shortcuts

Shortcuts work on the Annotate canvas. They're ignored while you're typing in a
text input, text area, or dropdown.

!!! note "Mac vs. Windows/Linux"
    ++cmd++ on macOS is ++ctrl++ on Windows/Linux. Both are shown as
    ++cmd++/++ctrl++ below.

## Tools

| Key | Tool |
| --- | --- |
| ++s++ | Select |
| ++p++ | Polygon |
| ++m++ | Magnetic |
| ++g++ | Magic |
| ++e++ | Rect |
| ++l++ | Ellipse |
| ++b++ | Brush |
| ++f++ | Fill |
| ++r++ | Eraser |
| hold ++space++ | Pan (reverts to previous tool on release) |

## Canvas & navigation

| Key | Action |
| --- | --- |
| ++t++ | Fit image to screen |
| Mouse wheel | Zoom toward cursor |
| ++x++ | Next slice |
| ++left++ / ++right++ | Previous / next slice |
| ++1++–++9++ | Activate class 1–9 |
| ++n++ | New brush instance |

## Editing

| Key | Action |
| --- | --- |
| ++cmd+z++ | Undo. **While drafting a polygon/magnetic shape**, removes the last node instead; right after an accidental close, reopens the shape to edit mode. |
| ++cmd+shift+z++ / ++ctrl+y++ | Redo |
| ++cmd+a++ / ++ctrl+a++ | Select all shapes on the slice (Select tool; scoped by the this-class / all-classes radios) |
| ++cmd+c++ | Copy selected shapes (Select tool) |
| ++cmd+v++ | Paste shapes (Select tool) |
| ++i++ | Invert a single selected shape |
| ++delete++ / ++backspace++ | Delete selected shapes |
| ++esc++ | Cancel an in-progress polygon / magic / fill draft |
| ++enter++ | Commit a Magic/Fill selection, or apply an active region op |

## Mouse modifiers (Magic / Fill / SAM)

| Action | Effect |
| --- | --- |
| ++shift++ + click | Add to the selection |
| ++alt++ + click (++opt++ on Mac) | Exclude a region ("not" point) |
| Double-click | Finish a Polygon or Magnetic shape |
| Double-click an edge | Add a vertex (Select tool, on a polygon) |
| Double-click a vertex | Delete that vertex (Select tool, on a polygon) |
