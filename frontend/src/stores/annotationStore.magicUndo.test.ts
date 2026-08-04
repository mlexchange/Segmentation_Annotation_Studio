/**
 * Does undo reverse a Magic-tool commit?
 *
 * The Magic/Fill tools batch their whole selection into one `setShapes` (see
 * AnnotationCanvas's commitMagic → commitShapes), so a committed selection must
 * come back off the stack in a single step. Undo in the UI routes through
 * editHistory, not zundo directly, so this drives that same entry point.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { useAnnotationStore } from './annotationStore';
import * as editHistory from '@/hooks/editHistory';
import type { Shape } from './annotationStore';

const KEY = 'tiled:http://x:browse/sample';

function poly(id: string, classId = 1): Shape {
  return { id, classId, kind: 'polygon', points: [0, 0, 10, 0, 10, 10] };
}

const shapesOn = (slice = 0) => useAnnotationStore.getState().byImage[KEY]?.[String(slice)] ?? [];
const pastCount = () => useAnnotationStore.temporal.getState().pastStates.length;

describe('undo after a Magic-tool commit', () => {
  beforeEach(() => {
    useAnnotationStore.getState().reset();
    useAnnotationStore.temporal.getState().clear();
    editHistory.clearHistory();
  });

  it('reverses a committed magic selection in one step', () => {
    // commitMagic maps every preview polygon into one batched setShapes.
    useAnnotationStore.getState().setShapes(KEY, 0, [poly('m1'), poly('m2'), poly('m3')]);
    expect(shapesOn()).toHaveLength(3);

    editHistory.undo();

    expect(shapesOn()).toHaveLength(0);
  });

  it('keeps shapes that were already on the slice before the magic commit', () => {
    useAnnotationStore.getState().setShapes(KEY, 0, [poly('existing')]);
    useAnnotationStore.getState().setShapes(KEY, 0, [poly('existing'), poly('magic')]);

    editHistory.undo();

    expect(shapesOn().map((s) => s.id)).toEqual(['existing']);
  });

  it('redoes the magic commit', () => {
    useAnnotationStore.getState().setShapes(KEY, 0, [poly('m1')]);
    editHistory.undo();
    expect(shapesOn()).toHaveLength(0);

    editHistory.redo();

    expect(shapesOn().map((s) => s.id)).toEqual(['m1']);
  });

  it('a paused clearDraft does not desync editHistory from the zundo stack', () => {
    // The canvas clears the draft on every slice/sample change with temporal
    // paused; if that ever recorded a journal entry without a matching zundo
    // entry, undo would pop the wrong one and appear to do nothing.
    useAnnotationStore.getState().setShapes(KEY, 0, [poly('m1')]);
    const before = pastCount();

    const temporal = useAnnotationStore.temporal.getState();
    temporal.pause();
    useAnnotationStore.getState().clearDraft();
    temporal.resume();

    expect(pastCount()).toBe(before);

    editHistory.undo();
    expect(shapesOn()).toHaveLength(0);
  });

  it('clearing an already-empty draft does not consume an undo press', () => {
    // Regression: zundo records on every set() unless given an equality fn, so a
    // no-op clearDraft used to become a real undo step. The canvas calls
    // clearDraft on every tool switch, so switching to Magic — or away from it
    // after committing — silently ate the user's next Undo press.
    useAnnotationStore.getState().setShapes(KEY, 0, [poly('m1')]);
    const before = pastCount();

    useAnnotationStore.getState().clearDraft(); // tool switch, draft already empty
    expect(pastCount()).toBe(before);

    editHistory.undo();

    expect(shapesOn()).toHaveLength(0); // the very first press reverses the commit
  });

  it('still records clearing a REAL draft, so undo reopens it', () => {
    useAnnotationStore.getState().addPolyNode(KEY, '0', 5, 5);
    expect(useAnnotationStore.getState().draft.poly).toEqual([5, 5]);

    useAnnotationStore.getState().clearDraft(); // Escape / real tool switch

    expect(useAnnotationStore.getState().draft.poly).toEqual([]);
    editHistory.undo();
    expect(useAnnotationStore.getState().draft.poly).toEqual([5, 5]);
  });

  it('still records touchHistory, which forces a no-content entry on purpose', () => {
    const before = pastCount();
    useAnnotationStore.getState().touchHistory();
    expect(pastCount()).toBe(before + 1);
  });

  it('deliberately stays reference-based, so a rebuilt-but-identical write still records', () => {
    // Documents the boundary of the fix. setShapes spreads byImage, so writing
    // the same shapes twice yields a new reference and is still recorded. That is
    // intentional: deep-comparing every annotation on each set would cost more
    // than the phantom step it saves, and real edits always carry new content.
    // The phantom steps that actually hurt (clearDraft on an empty draft) keep
    // every reference intact, which is exactly what the equality fn catches.
    const same = [poly('m1')];
    useAnnotationStore.getState().setShapes(KEY, 0, same);
    const after = pastCount();
    useAnnotationStore.getState().setShapes(KEY, 0, same);
    expect(pastCount()).toBe(after + 1);
  });
});
