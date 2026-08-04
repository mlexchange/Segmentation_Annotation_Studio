import { describe, expect, it } from 'vitest';
import { gatherTrainingSources, listAnnotatedSourceKeys } from './gatherTrainingSources';
import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

const tissue: AnnotationClass[] = [
  { classId: 1, label: 'Tissue', color: '#111111', isVisible: true },
];
const shape: Shape = {
  id: 'shape-1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10,
};

describe('listAnnotatedSourceKeys', () => {
  it('includes only sourceKeys with at least one shape', () => {
    const byImage = {
      'local:a.tif': { '0': [shape] },
      'local:b.tif': { '0': [] },
    };
    expect(listAnnotatedSourceKeys(byImage).map((c) => c.sourceKey)).toEqual(['local:a.tif']);
  });

  it('reports the total shape count across all slices', () => {
    const byImage = { 'local:a.tif': { '0': [shape], '1': [shape, shape] } };
    expect(listAnnotatedSourceKeys(byImage)[0].shapeCount).toBe(3);
  });

  it('sorts results by sourceKey', () => {
    const byImage = { 'local:b.tif': { '0': [shape] }, 'local:a.tif': { '0': [shape] } };
    expect(listAnnotatedSourceKeys(byImage).map((c) => c.sourceKey)).toEqual(['local:a.tif', 'local:b.tif']);
  });
});

describe('gatherTrainingSources', () => {
  it('throws when no sources are selected', () => {
    expect(() => gatherTrainingSources([], {}, {}, {}, {}, null, [])).toThrow(/select at least one/i);
  });

  it('builds one source item per selected sourceKey, parsed from the key', () => {
    const byImage = { 'local:a.tif': { '0': [shape] } };
    const result = gatherTrainingSources(
      ['local:a.tif'], byImage, {}, {}, { 'local:a.tif': tissue }, null, [],
    );
    expect(result.sources).toEqual([{
      kind: 'local', source: 'a.tif', server_uri: null,
      slices: { '0': [shape] }, split_by_slice: {}, negative_slices: [],
    }]);
    expect(result.classes).toEqual(tissue);
  });

  it('includes the currently-open sample\'s live classes when it is among the selection', () => {
    const byImage = { 'local:a.tif': { '0': [shape] } };
    const liveClasses: AnnotationClass[] = [{ ...tissue[0], color: '#abcdef' }];
    const result = gatherTrainingSources(
      ['local:a.tif'], byImage, {}, {}, {}, 'local:a.tif', liveClasses,
    );
    expect(result.classes).toEqual(liveClasses);
  });

  it('propagates a taxonomy mismatch error across selected samples', () => {
    const byImage = { 'local:a.tif': { '0': [shape] }, 'local:b.tif': { '0': [shape] } };
    const mismatched: AnnotationClass[] = [{ classId: 1, label: 'Defect', color: '#222', isVisible: true }];
    expect(() => gatherTrainingSources(
      ['local:a.tif', 'local:b.tif'], byImage, {}, {},
      { 'local:a.tif': tissue, 'local:b.tif': mismatched }, null, [],
    )).toThrow(/different class taxonomies/i);
  });
});
