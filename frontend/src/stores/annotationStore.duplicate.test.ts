import { describe, it, expect, beforeEach } from 'vitest';
import { useAnnotationStore } from './annotationStore';
import type { Shape } from './annotationStore';

const SK = 'tiled::sample';
const poly = (id: string, classId: number): Shape => ({
  id, classId, kind: 'polygon', points: [0, 0, 10, 0, 10, 10],
});

describe('duplicateClassShapes', () => {
  beforeEach(() => useAnnotationStore.getState().reset());

  it('clones all shapes of a class across slices into the new class with fresh ids', () => {
    const st = useAnnotationStore.getState();
    st.addShapes(SK, 0, [poly('a', 1), poly('b', 2)]);
    st.addShapes(SK, 1, [poly('c', 1)]);

    st.duplicateClassShapes(SK, 1, 5);

    const s0 = useAnnotationStore.getState().byImage[SK]['0'];
    const s1 = useAnnotationStore.getState().byImage[SK]['1'];
    // Originals untouched.
    expect(s0.filter((s) => s.classId === 1)).toHaveLength(1);
    expect(s0.filter((s) => s.classId === 2)).toHaveLength(1);
    // One copy of each class-1 shape now exists under class 5, per slice.
    const copies0 = s0.filter((s) => s.classId === 5);
    const copies1 = s1.filter((s) => s.classId === 5);
    expect(copies0).toHaveLength(1);
    expect(copies1).toHaveLength(1);
    // Fresh ids, same geometry.
    expect(copies0[0].id).not.toBe('a');
    expect((copies0[0] as { points: number[] }).points).toEqual([0, 0, 10, 0, 10, 10]);
  });

  it('is a no-op for an unknown source key', () => {
    const st = useAnnotationStore.getState();
    st.duplicateClassShapes('nope', 1, 2);
    expect(useAnnotationStore.getState().byImage['nope']).toBeUndefined();
  });
});
