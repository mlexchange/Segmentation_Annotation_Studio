import { describe, expect, it } from 'vitest';
import { remapPredictedShapes, type RunClass } from './importPredictions';
import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

const shape = (id: string, classId: number): Shape => ({
  id, classId, kind: 'polygon', points: [0, 0, 10, 0, 5, 10],
});

describe('remapPredictedShapes', () => {
  it('maps a run class to an existing current class by case-insensitive label', () => {
    const runClasses: RunClass[] = [{ classId: 1, label: 'Pore', color: '#ff0000' }];
    const current: AnnotationClass[] = [{ classId: 7, label: 'pore', color: '#000000', isVisible: true }];
    const slices = { '0': [shape('s1', 1)] };

    const result = remapPredictedShapes(runClasses, current, slices);

    expect(result.classes).toEqual(current); // no new class appended
    expect(result.slices['0'][0].classId).toBe(7); // remapped to the existing class's id
  });

  it('appends a run class with no label match, keeping its color', () => {
    const runClasses: RunClass[] = [{ classId: 1, label: 'Void', color: '#00ff00' }];
    const current: AnnotationClass[] = [{ classId: 3, label: 'Pore', color: '#000000', isVisible: true }];
    const slices = { '0': [shape('s1', 1)] };

    const result = remapPredictedShapes(runClasses, current, slices);

    expect(result.classes).toHaveLength(2);
    const appended = result.classes[1];
    expect(appended.label).toBe('Void');
    expect(appended.color).toBe('#00ff00');
    expect(appended.classId).toBe(4); // max existing (3) + 1
    expect(result.slices['0'][0].classId).toBe(4);
  });

  it('assigns classId 1 when the current class list is empty', () => {
    const runClasses: RunClass[] = [{ classId: 1, label: 'Pore', color: '#ff0000' }];
    const result = remapPredictedShapes(runClasses, [], { '0': [shape('s1', 1)] });
    expect(result.classes[0].classId).toBe(1);
  });

  it('handles multiple run classes, some matched and some appended', () => {
    const runClasses: RunClass[] = [
      { classId: 1, label: 'Pore', color: '#111111' },
      { classId: 2, label: 'Crack', color: '#222222' },
    ];
    const current: AnnotationClass[] = [{ classId: 5, label: 'pore', color: '#000000', isVisible: true }];
    const slices = { '0': [shape('s1', 1), shape('s2', 2)] };

    const result = remapPredictedShapes(runClasses, current, slices);

    expect(result.classes).toHaveLength(2);
    expect(result.slices['0'].find((s) => s.id === 's1')?.classId).toBe(5); // matched existing
    expect(result.slices['0'].find((s) => s.id === 's2')?.classId).toBe(6); // appended
  });

  it('preserves shape geometry, only remapping classId', () => {
    const runClasses: RunClass[] = [{ classId: 1, label: 'Pore', color: '#ff0000' }];
    const current: AnnotationClass[] = [];
    const result = remapPredictedShapes(runClasses, current, { '0': [shape('s1', 1)] });
    expect(result.slices['0'][0]).toMatchObject({ id: 's1', kind: 'polygon', points: [0, 0, 10, 0, 5, 10] });
  });

  it('stamps every returned shape origin: "predicted"', () => {
    const runClasses: RunClass[] = [{ classId: 1, label: 'Pore', color: '#ff0000' }];
    const current: AnnotationClass[] = [];
    const result = remapPredictedShapes(runClasses, current, {
      '0': [shape('s1', 1), shape('s2', 1)],
    });
    expect(result.slices['0'].every((s) => s.origin === 'predicted')).toBe(true);
  });
});
