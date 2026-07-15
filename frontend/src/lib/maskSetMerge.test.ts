import { describe, expect, it } from 'vitest';
import { mergeMaskSetIntoAnnotations } from './maskSetMerge';
import type { Shape } from '@/stores/annotationStore';

describe('mergeMaskSetIntoAnnotations', () => {
  it('replace_class swaps that class and keeps others', () => {
    const ann: Shape[] = [
      { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 8, h: 8 },
      { id: 'b', classId: 2, kind: 'rectangle', x: 0, y: 0, w: 4, h: 4 },
    ];
    const set: Shape[] = [
      { id: 'c', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 4, h: 4 },
    ];
    const out = mergeMaskSetIntoAnnotations(ann, set, 16, 16, 'replace_class');
    expect(out.filter((s) => s.classId === 2)).toHaveLength(1);
    expect(out.some((s) => s.classId === 1)).toBe(true);
    // Old class-1 rect at origin should be gone (replaced)
    const c1 = out.filter((s) => s.classId === 1);
    expect(c1.every((s) => s.id !== 'a')).toBe(true);
  });

  it('union enlarges class coverage', () => {
    const ann: Shape[] = [
      { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 4, h: 4 },
    ];
    const set: Shape[] = [
      { id: 'c', classId: 1, kind: 'rectangle', x: 8, y: 8, w: 4, h: 4 },
    ];
    const out = mergeMaskSetIntoAnnotations(ann, set, 16, 16, 'union');
    expect(out.filter((s) => s.classId === 1).length).toBeGreaterThanOrEqual(1);
  });

  it('delete_where_set punches holes / removes overlap', () => {
    const ann: Shape[] = [
      { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 12, h: 12 },
    ];
    const set: Shape[] = [
      { id: 'c', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 12, h: 12 },
    ];
    const out = mergeMaskSetIntoAnnotations(ann, set, 16, 16, 'delete_where_set');
    expect(out.filter((s) => s.classId === 1)).toHaveLength(0);
  });
});
