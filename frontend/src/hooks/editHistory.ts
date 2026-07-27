/**
 * editHistory — makes class deletion a real, ordered, redoable Ctrl/Cmd+Z action.
 *
 * Region edits already live on zundo's temporal (undo) stack. Class *entries* live in a
 * separate, non-temporal store, so this module keeps a small journal in lockstep with
 * zundo: one entry per tracked edit (via annotationStore's `setEditListener`), tagged as
 * either a plain region edit (`null`) or a class deletion. All undo/redo call sites route
 * through `undo()`/`redo()` here, which drive zundo AND replay the paired class op — so a
 * class delete undoes/redoes as one step, correctly ordered with every other edit.
 *
 * Because each journal entry maps 1:1 to a zundo entry, `undoOps.length` stays equal to
 * `pastStates.length`; we re-trim after each edit to mirror zundo's `limit`.
 */
import { useAnnotationStore, setEditListener } from '@/stores/annotationStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { buildSourceKey } from '@/lib/sourceKey';

interface ClassDeleteEntry {
  kind: 'classDelete';
  sourceKey: string;
  cls: AnnotationClass;
  index: number;
}
/** A journal entry: a class deletion to replay, or `null` for a plain edit (region edit or
 *  a polygon/lasso draft node/close — those are fully handled by zundo's tracked `draft`). */
type Entry = ClassDeleteEntry | null;

let undoOps: Entry[] = [];
let redoOps: Entry[] = [];

// Set just before the annotation-store edit it describes, so the next tracked edit is
// tagged with it. Consumed by onTrackedEdit().
let pending: Entry = null;

/** Current sample key, or null when nothing is open. */
function currentSourceKey(): string | null {
  const { source, kind, serverUri } = useDatasetStore.getState();
  return source && kind ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri) : null;
}

/** Tag the next tracked edit as a class deletion (call immediately before the edit). */
export function markClassDelete(sourceKey: string, cls: AnnotationClass, index: number) {
  pending = { kind: 'classDelete', sourceKey, cls, index };
}

// Matches the annotationStore temporal `limit` so the journal drops its oldest entry in
// lockstep with zundo dropping its oldest pastState.
const LIMIT = 200;

/** Fired by annotationStore after each tracked edit — records one journal entry.
 *  NOTE: zundo invokes onSave BEFORE it appends the new pastState, so we cannot compare
 *  against pastStates.length here (it's still the pre-edit count) — just cap at LIMIT. */
function onTrackedEdit() {
  const entry = pending;
  pending = null;
  undoOps.push(entry);
  redoOps = [];
  if (undoOps.length > LIMIT) undoOps.shift();
}

// Register once when this module first loads (imported by the undo/redo call sites).
setEditListener(onTrackedEdit);

/** Undo one step: zundo region undo + (if the entry is a class delete) re-insert the class. */
export function undo() {
  const temporal = useAnnotationStore.temporal.getState();
  if (temporal.pastStates.length === 0) return;
  const entry = undoOps.length ? undoOps.pop()! : null;
  temporal.undo();
  if (entry?.kind === 'classDelete' && entry.sourceKey === currentSourceKey()) {
    useClassStore.getState().insertClass(entry.cls, entry.index);
  }
  redoOps.push(entry);
}

/** Redo one step: zundo region redo + (if the entry is a class delete) re-remove the class. */
export function redo() {
  const temporal = useAnnotationStore.temporal.getState();
  if (temporal.futureStates.length === 0) return;
  const entry = redoOps.length ? redoOps.pop()! : null;
  temporal.redo();
  if (entry?.kind === 'classDelete' && entry.sourceKey === currentSourceKey()) {
    useClassStore.getState().deleteClass(entry.cls.classId);
  }
  undoOps.push(entry);
}

/** Reset both the region (zundo) history and the class-delete journal. Called on sample
 *  switch so undo/redo is scoped per sample. */
export function clearHistory() {
  // Drop any in-progress draft too (its nodes lived on the temporal stack we're clearing),
  // so no draft survives a sample switch. The set is wiped by clear() immediately below.
  useAnnotationStore.getState().clearDraft();
  undoOps = [];
  redoOps = [];
  useAnnotationStore.temporal.getState().clear();
}
