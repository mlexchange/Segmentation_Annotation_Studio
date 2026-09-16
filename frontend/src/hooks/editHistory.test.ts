import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { clearHistory, markClassDelete, redo, undo } from './editHistory';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { useDatasetStore } from '@/stores/datasetStore';

const SOURCE_KEY = 'local:x.tif';

function openSample() {
  useDatasetStore.getState().setDataset('local', 'x.tif', null, {
    nSlices: 3,
    height: 10,
    width: 10,
    dtype: 'uint8',
    isRgb: false,
    valueRange: [0, 255],
  });
}

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useAnnotationStore.temporal.getState().clear();
  useClassStore.setState({ classes: [] });
  useDatasetStore.getState().reset();
});

afterEach(() => {
  clearHistory();
});

describe('editHistory', () => {
  it('undo()/redo() are no-ops when there is no history', () => {
    expect(() => undo()).not.toThrow();
    expect(() => redo()).not.toThrow();
    expect(useAnnotationStore.temporal.getState().pastStates).toHaveLength(0);
  });

  it('undo reverts a plain region edit (not tagged as a class delete)', () => {
    openSample();
    useAnnotationStore.getState().addShape(SOURCE_KEY, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2,
    });
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0']).toHaveLength(1);

    undo();
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0'] ?? []).toHaveLength(0);

    redo();
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0']).toHaveLength(1);
  });

  it('markClassDelete + undo re-inserts the class at its original index when sourceKey matches', () => {
    openSample();
    const cls: AnnotationClass = { classId: 5, label: 'Pore', color: '#f00', isVisible: true };
    useClassStore.setState({ classes: [cls] });

    markClassDelete(SOURCE_KEY, cls, 0);
    useClassStore.getState().deleteClass(5);
    // The class-delete journal only records on the NEXT tracked annotationStore edit
    // (onTrackedEdit fires from annotationStore, not classStore), so we need one.
    useAnnotationStore.getState().touchHistory();

    expect(useClassStore.getState().classes).toHaveLength(0);

    undo();
    expect(useClassStore.getState().classes).toEqual([cls]);

    redo();
    expect(useClassStore.getState().classes).toHaveLength(0);
  });

  it('does not replay a class-delete entry when the current sourceKey differs', () => {
    openSample();
    const cls: AnnotationClass = { classId: 5, label: 'Pore', color: '#f00', isVisible: true };
    useClassStore.setState({ classes: [cls] });

    markClassDelete(SOURCE_KEY, cls, 0);
    useClassStore.getState().deleteClass(5);
    useAnnotationStore.getState().touchHistory();
    expect(useClassStore.getState().classes).toHaveLength(0);

    // Switch to a different sample before undoing.
    useDatasetStore.getState().setDataset('local', 'other.tif', null, {
      nSlices: 1, height: 1, width: 1, dtype: 'uint8', isRgb: false, valueRange: [0, 255],
    });

    undo();
    // Region history still undoes (temporal stack is shared), but the class is
    // NOT re-inserted because currentSourceKey() no longer matches the entry.
    expect(useClassStore.getState().classes).toHaveLength(0);
  });

  it('redo/undo stacks stay ordered across a mix of plain and class-delete edits', () => {
    openSample();
    const cls: AnnotationClass = { classId: 5, label: 'Pore', color: '#f00', isVisible: true };
    useClassStore.setState({ classes: [cls] });

    // 1: plain region edit
    useAnnotationStore.getState().addShape(SOURCE_KEY, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1,
    });
    // 2: class delete
    markClassDelete(SOURCE_KEY, cls, 0);
    useClassStore.getState().deleteClass(5);
    useAnnotationStore.getState().touchHistory();

    // Undo class delete first (LIFO)
    undo();
    expect(useClassStore.getState().classes).toEqual([cls]);
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0']).toHaveLength(1);

    // Undo plain region edit
    undo();
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0'] ?? []).toHaveLength(0);

    // Redo both back in order
    redo();
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0']).toHaveLength(1);
    redo();
    expect(useClassStore.getState().classes).toHaveLength(0);
  });

  it('a new edit after undo clears the redo stack', () => {
    openSample();
    useAnnotationStore.getState().addShape(SOURCE_KEY, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1,
    });
    undo();
    // A fresh edit (not via redo) should drop the stale redo entry.
    useAnnotationStore.getState().addShape(SOURCE_KEY, 0, {
      id: 's2', classId: 1, kind: 'rectangle', x: 1, y: 1, w: 1, h: 1,
    });
    redo(); // should be a no-op now (no future states)
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0']).toHaveLength(1);
    expect(useAnnotationStore.getState().byImage[SOURCE_KEY]?.['0']?.[0].id).toBe('s2');
  });

  it('clearHistory wipes the draft, journal, and temporal stack', () => {
    openSample();
    useAnnotationStore.getState().addShape(SOURCE_KEY, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1,
    });
    expect(useAnnotationStore.temporal.getState().pastStates.length).toBeGreaterThan(0);

    clearHistory();
    expect(useAnnotationStore.temporal.getState().pastStates).toHaveLength(0);
    expect(useAnnotationStore.temporal.getState().futureStates).toHaveLength(0);
    // undo/redo are now no-ops (journal cleared alongside temporal stack)
    expect(() => undo()).not.toThrow();
    expect(() => redo()).not.toThrow();
  });
});
