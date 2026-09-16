import { describe, it, expect, beforeEach } from 'vitest';
import { useAnnotationStore } from './annotationStore';
import type { Shape, BrushShape } from './annotationStore';

const SK = 'tiled::sample';

const rect = (id: string, classId: number, overrides: Partial<Shape> = {}): Shape => ({
  id, classId, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2, ...overrides,
} as Shape);

const poly = (id: string, classId: number): Shape => ({
  id, classId, kind: 'polygon', points: [0, 0, 10, 0, 10, 10],
});

const brush = (id: string, classId: number): BrushShape => ({
  id, classId, kind: 'brush', strokes: [],
});

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useAnnotationStore.temporal.getState().clear();
});

describe('addShape / addShapes / addShapesAcrossSlices', () => {
  it('addShape appends a single shape to the (source, slice)', () => {
    useAnnotationStore.getState().addShape(SK, 0, rect('a', 1));
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(1);
    expect(useAnnotationStore.getState().byImage[SK]['0'][0].id).toBe('a');
  });

  it('addShapes appends several shapes in one call', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 2)]);
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(2);
  });

  it('addShapesAcrossSlices distributes shapes to multiple slices in one update', () => {
    useAnnotationStore.getState().addShapesAcrossSlices(SK, {
      0: [rect('a', 1)],
      2: [rect('b', 1), rect('c', 1)],
    });
    const byImage = useAnnotationStore.getState().byImage[SK];
    expect(byImage['0']).toHaveLength(1);
    expect(byImage['2']).toHaveLength(2);
  });

  it('addShapesAcrossSlices skips slices with an empty shape list and is a no-op if all empty', () => {
    useAnnotationStore.getState().addShapesAcrossSlices(SK, { 0: [], 1: [] });
    expect(useAnnotationStore.getState().byImage[SK]).toBeUndefined();
  });

  it('addShapesAcrossSlices merges into existing slice shapes rather than replacing them', () => {
    useAnnotationStore.getState().addShape(SK, 0, rect('a', 1));
    useAnnotationStore.getState().addShapesAcrossSlices(SK, { 0: [rect('b', 1)] });
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['a', 'b']);
  });
});

describe('removeShape / removeShapes', () => {
  beforeEach(() => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 2), rect('c', 1)]);
  });

  it('removeShape drops one shape by id', () => {
    useAnnotationStore.getState().removeShape(SK, 0, 'b');
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['a', 'c']);
  });

  it('removeShapes drops several shapes by id in one call', () => {
    useAnnotationStore.getState().removeShapes(SK, 0, ['a', 'c']);
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['b']);
  });
});

describe('updateShape', () => {
  it('replaces only the targeted shape via the updater fn', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 2)]);
    useAnnotationStore.getState().updateShape(SK, 0, 'a', (sh) => ({ ...sh, classId: 9 } as Shape));
    const shapes = useAnnotationStore.getState().byImage[SK]['0'];
    expect(shapes.find((s) => s.id === 'a')?.classId).toBe(9);
    expect(shapes.find((s) => s.id === 'b')?.classId).toBe(2);
  });
});

describe('setClassForShapes', () => {
  it('reassigns only the listed shape ids to the new class', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 1), rect('c', 1)]);
    useAnnotationStore.getState().setClassForShapes(SK, 0, ['a', 'c'], 7);
    const shapes = useAnnotationStore.getState().byImage[SK]['0'];
    expect(shapes.find((s) => s.id === 'a')?.classId).toBe(7);
    expect(shapes.find((s) => s.id === 'b')?.classId).toBe(1);
    expect(shapes.find((s) => s.id === 'c')?.classId).toBe(7);
  });
});

describe('replaceClassShapesOnSlice', () => {
  it('replaces all shapes of a class on the slice, leaving other classes untouched', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 2)]);
    useAnnotationStore.getState().replaceClassShapesOnSlice(SK, 0, 1, [rect('new', 1)]);
    const shapes = useAnnotationStore.getState().byImage[SK]['0'];
    expect(shapes.map((s) => s.id).sort()).toEqual(['b', 'new']);
  });
});

describe('copySliceShapes', () => {
  beforeEach(() => {
    useAnnotationStore.getState().addShapes(SK, 0, [poly('a', 1), poly('b', 2)]);
  });

  it('clones shapes from the source slice into each target slice with fresh ids', () => {
    useAnnotationStore.getState().copySliceShapes(SK, 0, [1, 2]);
    const s1 = useAnnotationStore.getState().byImage[SK]['1'];
    const s2 = useAnnotationStore.getState().byImage[SK]['2'];
    expect(s1).toHaveLength(2);
    expect(s2).toHaveLength(2);
    expect(s1.map((s) => s.id)).not.toContain('a');
  });

  it('filters to one class when classId is given', () => {
    useAnnotationStore.getState().copySliceShapes(SK, 0, [1], 1);
    const s1 = useAnnotationStore.getState().byImage[SK]['1'];
    expect(s1).toHaveLength(1);
    expect(s1[0].classId).toBe(1);
  });

  it('skips a target slice equal to the source slice', () => {
    useAnnotationStore.getState().copySliceShapes(SK, 0, [0, 1]);
    // Slice 0 keeps its originals only (no duplicate append onto itself).
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(2);
    expect(useAnnotationStore.getState().byImage[SK]['1']).toHaveLength(2);
  });

  it('is a no-op when the source slice has nothing to copy', () => {
    useAnnotationStore.getState().copySliceShapes(SK, 5, [6]);
    expect(useAnnotationStore.getState().byImage[SK]['6']).toBeUndefined();
  });

  it('merges copies with shapes already on the target slice', () => {
    useAnnotationStore.getState().addShape(SK, 1, rect('existing', 3));
    useAnnotationStore.getState().copySliceShapes(SK, 0, [1]);
    expect(useAnnotationStore.getState().byImage[SK]['1']).toHaveLength(3);
  });
});

describe('removeShapesByClassId', () => {
  it('removes the class across every source and slice, pruning emptied slices/sources', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 2)]);
    useAnnotationStore.getState().addShapes('other::src', 0, [rect('c', 1)]);
    useAnnotationStore.getState().removeShapesByClassId(1);
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['b']);
    // The other source had only class-1 shapes, so it's pruned entirely.
    expect(useAnnotationStore.getState().byImage['other::src']).toBeUndefined();
  });

  it('prunes an emptied slice but keeps sibling slices with other classes', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().addShapes(SK, 1, [rect('b', 2)]);
    useAnnotationStore.getState().removeShapesByClassId(1);
    expect(useAnnotationStore.getState().byImage[SK]['0']).toBeUndefined();
    expect(useAnnotationStore.getState().byImage[SK]['1']).toHaveLength(1);
  });
});

describe('removeShapesByClassIdInSource', () => {
  it('removes the class only within the given source, leaving other sources alone', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1), rect('b', 2)]);
    useAnnotationStore.getState().addShapes('other::src', 0, [rect('c', 1)]);
    useAnnotationStore.getState().removeShapesByClassIdInSource(SK, 1);
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['b']);
    expect(useAnnotationStore.getState().byImage['other::src']['0']).toHaveLength(1);
  });

  it('deletes the source entirely when its last class is removed', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().removeShapesByClassIdInSource(SK, 1);
    expect(useAnnotationStore.getState().byImage[SK]).toBeUndefined();
  });

  it('is a no-op for a source with no data', () => {
    useAnnotationStore.getState().removeShapesByClassIdInSource('nope', 1);
    expect(useAnnotationStore.getState().byImage['nope']).toBeUndefined();
  });
});

describe('removeShapesByOrigin', () => {
  it('removes shapes of the given origin, treating missing origin as human', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [
      rect('human1', 1),
      { ...rect('pred1', 1), origin: 'predicted' } as Shape,
    ]);
    useAnnotationStore.getState().removeShapesByOrigin(SK, 'predicted');
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['human1']);
  });

  it('removes human-origin shapes (undefined origin) when asked', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [
      rect('human1', 1),
      { ...rect('pred1', 1), origin: 'predicted' } as Shape,
    ]);
    useAnnotationStore.getState().removeShapesByOrigin(SK, 'human');
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['pred1']);
  });

  it('deletes an emptied source and is a no-op for missing sources', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().removeShapesByOrigin(SK, 'human');
    expect(useAnnotationStore.getState().byImage[SK]).toBeUndefined();

    useAnnotationStore.getState().removeShapesByOrigin('nope', 'human');
    expect(useAnnotationStore.getState().byImage['nope']).toBeUndefined();
  });
});

describe('appendBrushStroke / appendEraseStroke', () => {
  it('appendBrushStroke appends a stroke to a brush shape and is a no-op for non-brush shapes', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [brush('br', 1), rect('rc', 1)]);
    const stroke = { points: [1, 1], radius: 3, mode: 'paint' as const };
    useAnnotationStore.getState().appendBrushStroke(SK, 0, 'br', stroke);
    useAnnotationStore.getState().appendBrushStroke(SK, 0, 'rc', stroke);
    const shapes = useAnnotationStore.getState().byImage[SK]['0'];
    const b = shapes.find((s) => s.id === 'br') as BrushShape;
    expect(b.strokes).toHaveLength(1);
    const r = shapes.find((s) => s.id === 'rc');
    expect((r as any).strokes).toBeUndefined();
  });

  it('appendEraseStroke adds an erase-mode brush stroke for brush shapes', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [brush('br', 1)]);
    useAnnotationStore.getState().appendEraseStroke(SK, 0, 'br', { points: [1, 1], radius: 2 });
    const b = useAnnotationStore.getState().byImage[SK]['0'][0] as BrushShape;
    expect(b.strokes).toHaveLength(1);
    expect(b.strokes[0].mode).toBe('erase');
  });

  it('appendEraseStroke adds an `erased` carve-out entry for vector shapes', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('rc', 1)]);
    useAnnotationStore.getState().appendEraseStroke(SK, 0, 'rc', { points: [1, 1], radius: 2 });
    const shape = useAnnotationStore.getState().byImage[SK]['0'][0];
    expect(shape.erased).toHaveLength(1);
    expect(shape.erased?.[0].radius).toBe(2);
  });
});

describe('setShapes', () => {
  it('replaces all shapes for the (source, slice)', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().setShapes(SK, 0, [rect('b', 2), rect('c', 2)]);
    expect(useAnnotationStore.getState().byImage[SK]['0'].map((s) => s.id)).toEqual(['b', 'c']);
  });
});

describe('setSplitForSlice / toggleNegativeSlice', () => {
  it('sets the split value for a slice', () => {
    useAnnotationStore.getState().setSplitForSlice(SK, 3, 'valid');
    expect(useAnnotationStore.getState().splitBySlice[SK]['3']).toBe('valid');
  });

  it('toggleNegativeSlice adds then removes a slice from the negative list', () => {
    useAnnotationStore.getState().toggleNegativeSlice(SK, 2);
    expect(useAnnotationStore.getState().negativeSlices[SK]).toEqual(['2']);
    useAnnotationStore.getState().toggleNegativeSlice(SK, 2);
    expect(useAnnotationStore.getState().negativeSlices[SK]).toEqual([]);
  });
});

describe('loadFromDraft / mergeSourceDraft', () => {
  it('loadFromDraft replaces all annotation data wholesale', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().loadFromDraft({
      byImage: { 'new::src': { '0': [rect('x', 1)] } },
      splitBySlice: {},
      negativeSlices: {},
    });
    expect(useAnnotationStore.getState().byImage[SK]).toBeUndefined();
    expect(useAnnotationStore.getState().byImage['new::src']['0']).toHaveLength(1);
  });

  it('mergeSourceDraft merges one source without touching others', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().mergeSourceDraft(
      'other::src',
      { '0': [rect('b', 2)] },
      { '0': 'train' },
      ['0'],
    );
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(1);
    expect(useAnnotationStore.getState().byImage['other::src']['0']).toHaveLength(1);
    expect(useAnnotationStore.getState().splitBySlice['other::src']['0']).toBe('train');
    expect(useAnnotationStore.getState().negativeSlices['other::src']).toEqual(['0']);
  });
});

describe('reset', () => {
  it('clears shapes, splits, negative slices, and the draft', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().setSplitForSlice(SK, 0, 'test');
    useAnnotationStore.getState().toggleNegativeSlice(SK, 0);
    useAnnotationStore.getState().addPolyNode(SK, '0', 1, 1);
    useAnnotationStore.getState().reset();
    const s = useAnnotationStore.getState();
    expect(s.byImage).toEqual({});
    expect(s.splitBySlice).toEqual({});
    expect(s.negativeSlices).toEqual({});
    expect(s.draft.tool).toBeNull();
  });
});

describe('touchHistory', () => {
  it('creates a shallow-cloned byImage but leaves the data equal', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    const before = useAnnotationStore.getState().byImage;
    useAnnotationStore.getState().touchHistory();
    const after = useAnnotationStore.getState().byImage;
    expect(after).not.toBe(before);
    expect(after).toEqual(before);
  });

  it('records a temporal (undo) entry even though content is unchanged', () => {
    useAnnotationStore.temporal.getState().clear();
    expect(useAnnotationStore.temporal.getState().pastStates.length).toBe(0);
    useAnnotationStore.getState().touchHistory();
    expect(useAnnotationStore.temporal.getState().pastStates.length).toBeGreaterThan(0);
  });
});

describe('draft: addPolyNode / addMagneticNode / clearDraft / commitDraftShapes', () => {
  it('addPolyNode starts a fresh draft when context changes and appends within the same context', () => {
    useAnnotationStore.getState().addPolyNode(SK, '0', 1, 2);
    expect(useAnnotationStore.getState().draft.poly).toEqual([1, 2]);
    useAnnotationStore.getState().addPolyNode(SK, '0', 3, 4);
    expect(useAnnotationStore.getState().draft.poly).toEqual([1, 2, 3, 4]);
  });

  it('addPolyNode resets the draft when the slice/source context changes', () => {
    useAnnotationStore.getState().addPolyNode(SK, '0', 1, 2);
    useAnnotationStore.getState().addPolyNode(SK, '1', 5, 6);
    expect(useAnnotationStore.getState().draft.poly).toEqual([5, 6]);
    expect(useAnnotationStore.getState().draft.sliceKey).toBe('1');
  });

  it('addMagneticNode accumulates path points for the same context and reseeds otherwise', () => {
    useAnnotationStore.getState().addMagneticNode(SK, '0', [1, 1, 2, 2], { x: 1, y: 1 });
    useAnnotationStore.getState().addMagneticNode(SK, '0', [3, 3], { x: 3, y: 3 });
    const draft = useAnnotationStore.getState().draft;
    expect(draft.magnetic).toEqual([1, 1, 2, 2, 3, 3]);
    expect(draft.magneticSeed).toEqual({ x: 3, y: 3 });
  });

  it('clearDraft resets to the empty draft', () => {
    useAnnotationStore.getState().addPolyNode(SK, '0', 1, 2);
    useAnnotationStore.getState().clearDraft();
    expect(useAnnotationStore.getState().draft.tool).toBeNull();
    expect(useAnnotationStore.getState().draft.poly).toEqual([]);
  });

  it('commitDraftShapes writes the slice shapes and clears the draft atomically', () => {
    useAnnotationStore.getState().addPolyNode(SK, '0', 1, 2);
    useAnnotationStore.getState().commitDraftShapes(SK, 0, [poly('finished', 1)]);
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(1);
    expect(useAnnotationStore.getState().draft.tool).toBeNull();
  });
});

describe('undo/redo via zundo temporal middleware', () => {
  it('undo restores the prior byImage state and redo re-applies the edit', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(1);

    useAnnotationStore.getState().addShapes(SK, 0, [rect('b', 1)]);
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(2);

    useAnnotationStore.temporal.getState().undo();
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(1);
    expect(useAnnotationStore.getState().byImage[SK]['0'][0].id).toBe('a');

    useAnnotationStore.temporal.getState().redo();
    expect(useAnnotationStore.getState().byImage[SK]['0']).toHaveLength(2);
  });

  it('undo can walk back past multiple edits to the initial empty state', () => {
    useAnnotationStore.getState().addShapes(SK, 0, [rect('a', 1)]);
    useAnnotationStore.getState().addShapes(SK, 1, [rect('b', 1)]);
    useAnnotationStore.getState().removeShape(SK, 0, 'a');

    useAnnotationStore.temporal.getState().undo();
    useAnnotationStore.temporal.getState().undo();
    useAnnotationStore.temporal.getState().undo();
    expect(useAnnotationStore.getState().byImage[SK]).toBeUndefined();
  });
});
