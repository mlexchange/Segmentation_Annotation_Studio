import { describe, expect, it } from 'vitest';
import { summarizeSliceAnnotations, type AnnotationPayload } from './sliceAnnotationSummary';
import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

const cls = (classId: number, label: string, color = '#111111'): AnnotationClass => ({
  classId, label, color, isVisible: true,
});

const poly = (id: string, classId: number): Shape => ({
  id, classId, kind: 'polygon', points: [0, 0, 10, 0, 5, 10],
});

const brush = (id: string, classId: number): Shape => ({
  id, classId, kind: 'brush', strokes: [{ points: [0, 0, 5, 5], radius: 3, mode: 'paint' }],
});

describe('summarizeSliceAnnotations', () => {
  it('returns an empty summary when no payloads are present', () => {
    const result = summarizeSliceAnnotations([{ payload: null, sliceKey: '3' }]);
    expect(result).toEqual({ classes: [], totalShapes: 0, isNegative: false, split: null });
  });

  it('groups a slice\'s shapes by class, ignoring other slices', () => {
    const payload: AnnotationPayload = {
      classes: [cls(1, 'background'), cls(2, 'leaf')],
      slices: {
        '3': [poly('a', 1), poly('b', 2), brush('c', 2)],
        '4': [poly('other', 1)],
      },
    };

    const result = summarizeSliceAnnotations([{ payload, sliceKey: '3' }]);

    expect(result.totalShapes).toBe(3);
    expect(result.classes).toHaveLength(2);
    expect(result.classes[0]).toMatchObject({ classId: 1, label: 'background' });
    expect(result.classes[0].shapes).toEqual([{ id: 'a', kind: 'polygon' }]);
    expect(result.classes[1].shapes.map((s) => s.kind)).toEqual(['polygon', 'brush']);
  });

  it('omits classes that have no shapes on this slice', () => {
    const payload: AnnotationPayload = {
      classes: [cls(1, 'background'), cls(2, 'unused')],
      slices: { '0': [poly('a', 1)] },
    };

    const result = summarizeSliceAnnotations([{ payload, sliceKey: '0' }]);

    expect(result.classes.map((c) => c.label)).toEqual(['background']);
  });

  it('merges the volume-key and standalone-key payloads for one slice', () => {
    // Same physical slice: index 3 of the volume, or slice "0" of its own array.
    const volume: AnnotationPayload = {
      classes: [cls(1, 'leaf')],
      slices: { '3': [poly('from-volume', 1)] },
    };
    const standalone: AnnotationPayload = {
      classes: [cls(1, 'leaf'), cls(5, 'pore', '#ff0000')],
      slices: { '0': [poly('from-standalone', 5)] },
    };

    const result = summarizeSliceAnnotations([
      { payload: volume, sliceKey: '3' },
      { payload: standalone, sliceKey: '0' },
    ]);

    expect(result.totalShapes).toBe(2);
    expect(result.classes.map((c) => c.label)).toEqual(['leaf', 'pore']);
    expect(result.classes[1].color).toBe('#ff0000');
  });

  it('de-duplicates a shape present under both keys', () => {
    const shared = poly('same-id', 1);
    const a: AnnotationPayload = { classes: [cls(1, 'leaf')], slices: { '2': [shared] } };
    const b: AnnotationPayload = { classes: [cls(1, 'leaf')], slices: { '0': [shared] } };

    const result = summarizeSliceAnnotations([
      { payload: a, sliceKey: '2' },
      { payload: b, sliceKey: '0' },
    ]);

    expect(result.totalShapes).toBe(1);
    expect(result.classes[0].shapes).toHaveLength(1);
  });

  it('reports a shape whose class is missing from the class list', () => {
    const payload: AnnotationPayload = { classes: [], slices: { '0': [poly('orphan', 9)] } };

    const result = summarizeSliceAnnotations([{ payload, sliceKey: '0' }]);

    expect(result.totalShapes).toBe(1);
    expect(result.classes[0]).toMatchObject({ classId: 9, label: 'Class 9' });
  });

  it('surfaces the negative-example flag for this slice only', () => {
    const payload: AnnotationPayload = { classes: [], slices: {}, negative_slices: ['3', '7'] };

    expect(summarizeSliceAnnotations([{ payload, sliceKey: '3' }]).isNegative).toBe(true);
    expect(summarizeSliceAnnotations([{ payload, sliceKey: '4' }]).isNegative).toBe(false);
  });

  it('surfaces an explicit split but treats auto as unassigned', () => {
    const explicit: AnnotationPayload = { split_by_slice: { '3': 'valid' } };
    const auto: AnnotationPayload = { split_by_slice: { '3': 'auto' } };

    expect(summarizeSliceAnnotations([{ payload: explicit, sliceKey: '3' }]).split).toBe('valid');
    expect(summarizeSliceAnnotations([{ payload: auto, sliceKey: '3' }]).split).toBeNull();
  });

  it('tolerates payloads with missing fields', () => {
    const result = summarizeSliceAnnotations([{ payload: {}, sliceKey: '0' }]);
    expect(result).toEqual({ classes: [], totalShapes: 0, isNegative: false, split: null });
  });
});
