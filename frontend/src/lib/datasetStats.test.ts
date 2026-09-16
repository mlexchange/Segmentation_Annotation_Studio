import { describe, expect, it } from 'vitest';
import { computeSampleStats } from './datasetStats';
import type { AnnotationClass } from '@/stores/classStore';

const CLASSES: AnnotationClass[] = [
  { classId: 1, label: 'Cell', color: '#f00', isVisible: true },
  { classId: 2, label: 'Wall', color: '#0f0', isVisible: true },
];

describe('computeSampleStats', () => {
  it('reports zero stats for a source with no slices at all', () => {
    const stats = computeSampleStats({}, CLASSES, 3, [], 100, 100);
    expect(stats.coverage.annotatedSlices).toBe(0);
    expect(stats.classStats.every((c) => c.shapeCount === 0)).toBe(true);
  });

  it('counts shapes and pixel area per class', () => {
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 }] },
      CLASSES, 3, [], 100, 100,
    );
    const cell = stats.classStats.find((c) => c.classId === 1)!;
    expect(cell.shapeCount).toBe(1);
    expect(cell.pixelArea).toBeGreaterThan(0);
    expect(stats.coverage.annotatedSlices).toBe(1);
  });

  it('flags a tiny (sliver) shape', () => {
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 }] },
      CLASSES, 3, [], 100, 100,
    );
    expect(stats.flags.some((f) => f.kind === 'sliver')).toBe(true);
  });

  it('does not flag a normal-sized shape as a sliver', () => {
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 50, h: 50 }] },
      CLASSES, 3, [], 100, 100,
    );
    expect(stats.flags.some((f) => f.kind === 'sliver')).toBe(false);
  });

  it('flags a self-intersecting polygon', () => {
    // A "bowtie" shape: crosses itself in the middle.
    const bowtie = [0, 0, 100, 100, 100, 0, 0, 100];
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'polygon', points: bowtie }] },
      CLASSES, 3, [], 200, 200,
    );
    expect(stats.flags.some((f) => f.kind === 'self-intersection')).toBe(true);
  });

  it('does not flag a simple (non-intersecting) polygon', () => {
    const square = [0, 0, 50, 0, 50, 50, 0, 50];
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'polygon', points: square }] },
      CLASSES, 3, [], 200, 200,
    );
    expect(stats.flags.some((f) => f.kind === 'self-intersection')).toBe(false);
  });

  it('flags an overlap between two different classes on the same slice', () => {
    const stats = computeSampleStats(
      {
        '0': [
          { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 50, h: 50 },
          { id: 's2', classId: 2, kind: 'rectangle', x: 25, y: 25, w: 50, h: 50 },
        ],
      },
      CLASSES, 3, [], 200, 200,
    );
    expect(stats.flags.some((f) => f.kind === 'overlap')).toBe(true);
  });

  it('does not flag disjoint shapes of different classes', () => {
    const stats = computeSampleStats(
      {
        '0': [
          { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 20, h: 20 },
          { id: 's2', classId: 2, kind: 'rectangle', x: 150, y: 150, w: 20, h: 20 },
        ],
      },
      CLASSES, 3, [], 200, 200,
    );
    expect(stats.flags.some((f) => f.kind === 'overlap')).toBe(false);
  });

  it('flags empty-unmarked slices (not annotated, not negative)', () => {
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 50, h: 50 }] },
      CLASSES, 3, [], 100, 100,
    );
    expect(stats.coverage.emptyUnmarked).toEqual([1, 2]);
    expect(stats.flags.filter((f) => f.kind === 'empty-unmarked')).toHaveLength(2);
  });

  it('does not flag a slice marked as negative', () => {
    const stats = computeSampleStats({}, CLASSES, 2, ['0', '1'], 100, 100);
    expect(stats.coverage.emptyUnmarked).toEqual([]);
    expect(stats.coverage.negativeSlices).toBe(2);
  });

  it('suppresses empty-unmarked flags entirely when there are too many (>20)', () => {
    const stats = computeSampleStats({}, CLASSES, 25, [], 100, 100);
    expect(stats.coverage.emptyUnmarked).toHaveLength(25);
    expect(stats.flags.filter((f) => f.kind === 'empty-unmarked')).toHaveLength(0);
  });

  it('computes area for an ellipse shape', () => {
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'ellipse', cx: 50, cy: 50, rx: 20, ry: 10 }] },
      CLASSES, 1, [], 100, 100,
    );
    const cell = stats.classStats.find((c) => c.classId === 1)!;
    expect(cell.pixelArea).toBeGreaterThan(0);
  });

  it('computes area for a brush stroke', () => {
    const stats = computeSampleStats(
      {
        '0': [{
          id: 's1', classId: 1, kind: 'brush',
          strokes: [{ mode: 'paint', points: [0, 0, 50, 50], radius: 5 }],
        }],
      },
      CLASSES, 1, [], 100, 100,
    );
    const cell = stats.classStats.find((c) => c.classId === 1)!;
    expect(cell.pixelArea).toBeGreaterThan(0);
  });

  it('ignores an erase-mode brush stroke for area purposes', () => {
    const stats = computeSampleStats(
      {
        '0': [{
          id: 's1', classId: 1, kind: 'brush',
          strokes: [{ mode: 'erase', points: [0, 0, 50, 50], radius: 5 }],
        }],
      },
      CLASSES, 1, [], 100, 100,
    );
    const cell = stats.classStats.find((c) => c.classId === 1)!;
    // Shape still counts, but analytic area contribution from an erase stroke is 0.
    expect(cell.shapeCount).toBe(1);
  });

  it('a polygon hole reduces its net area', () => {
    const outer = [0, 0, 100, 0, 100, 100, 0, 100];
    const hole = [25, 25, 75, 25, 75, 75, 25, 75];
    const statsWithHole = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'polygon', points: outer, holes: [hole] }] },
      CLASSES, 1, [], 200, 200,
    );
    const statsNoHole = computeSampleStats(
      { '0': [{ id: 's1', classId: 1, kind: 'polygon', points: outer }] },
      CLASSES, 1, [], 200, 200,
    );
    const withHoleArea = statsWithHole.classStats.find((c) => c.classId === 1)!.pixelArea;
    const noHoleArea = statsNoHole.classStats.find((c) => c.classId === 1)!.pixelArea;
    expect(withHoleArea).toBeLessThan(noHoleArea);
  });

  it('ignores an empty-array slice entry (not annotated)', () => {
    const stats = computeSampleStats({ '0': [] }, CLASSES, 1, [], 100, 100);
    expect(stats.coverage.annotatedSlices).toBe(0);
  });

  it('labels a flag for a class not in the provided class list generically', () => {
    const stats = computeSampleStats(
      { '0': [{ id: 's1', classId: 99, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 }] },
      CLASSES, 1, [], 100, 100,
    );
    const sliver = stats.flags.find((f) => f.kind === 'sliver')!;
    expect(sliver.message).toContain('class 99');
  });
});
