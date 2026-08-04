import { describe, expect, it } from 'vitest';
import { validateSharedTaxonomy } from './exportValidation';
import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

const tissue: AnnotationClass[] = [
  { classId: 1, label: 'Tissue', color: '#111111', isVisible: true },
];
const defect: AnnotationClass[] = [
  { classId: 1, label: 'Defect', color: '#222222', isVisible: true },
];
const shape: Shape = {
  id: 'shape-1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10,
};

describe('validateSharedTaxonomy', () => {
  it('accepts a multi-sample export only when every taxonomy is equivalent', () => {
    expect(() => validateSharedTaxonomy(
      ['a', 'b'],
      { a: tissue, b: [{ ...tissue[0], color: '#ffffff', isVisible: false }] },
      { a: { '0': [shape] }, b: { '0': [shape] } },
    )).not.toThrow();
  });

  it('rejects samples that reuse a class id for different labels', () => {
    expect(() => validateSharedTaxonomy(
      ['a', 'b'],
      { a: tissue, b: defect },
      { a: { '0': [shape] }, b: { '0': [shape] } },
    )).toThrow(/different class taxonomies/i);
  });

  it('rejects shapes whose class id is absent from the sample taxonomy', () => {
    expect(() => validateSharedTaxonomy(
      ['a'],
      { a: tissue },
      { a: { '0': [{ ...shape, classId: 99 }] } },
    )).toThrow(/class 99/i);
  });

  it('rejects multi-sample exports when a source has no recorded taxonomy', () => {
    expect(() => validateSharedTaxonomy(
      ['a', 'b'],
      { a: tissue },
      { a: { '0': [shape] }, b: { '0': [shape] } },
    )).toThrow(/taxonomy.*b/i);
  });
});
