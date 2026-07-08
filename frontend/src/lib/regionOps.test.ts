import { describe, it, expect } from 'vitest';
import { computeRegionOps } from './regionOps';
import type { Shape } from '@/stores/annotationStore';

const W = 200, H = 200;

function rect(id: string, classId: number, x: number, y: number, w: number, h: number): Shape {
  return { id, classId, kind: 'rectangle', x, y, w, h };
}

/** Rough area (px) of a returned polygon via its bounding box (adequate for sanity). */
function bboxArea(points: number[]): number {
  const xs = points.filter((_, i) => i % 2 === 0);
  const ys = points.filter((_, i) => i % 2 === 1);
  return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
}

describe('computeRegionOps', () => {
  it('merge unions two overlapping rects of a class into one region', () => {
    const shapes = [rect('a', 1, 20, 20, 60, 60), rect('b', 1, 60, 60, 60, 60)];
    const out = computeRegionOps(shapes, W, H, 'merge', 0);
    expect(out.length).toBe(1);
    expect(out[0].classId).toBe(1);
  });

  it('grow enlarges, shrink reduces the bounding area', () => {
    const shapes = [rect('a', 1, 60, 60, 40, 40)];
    const base = bboxArea(computeRegionOps(shapes, W, H, 'merge', 0)[0].points);
    const grown = bboxArea(computeRegionOps(shapes, W, H, 'grow', 8)[0].points);
    const shrunk = bboxArea(computeRegionOps(shapes, W, H, 'shrink', 8)[0].points);
    expect(grown).toBeGreaterThan(base);
    expect(shrunk).toBeLessThan(base);
  });

  it('remove-islands drops components smaller than the threshold', () => {
    const shapes = [rect('big', 1, 40, 40, 80, 80), rect('speck', 1, 5, 5, 3, 3)];
    const out = computeRegionOps(shapes, W, H, 'islands', 500); // 500 px² min
    expect(out.length).toBe(1); // the speck is removed
  });

  it('keeps classes separate', () => {
    const shapes = [rect('a', 1, 20, 20, 30, 30), rect('b', 2, 120, 120, 30, 30)];
    const out = computeRegionOps(shapes, W, H, 'merge', 0);
    expect(new Set(out.map((r) => r.classId))).toEqual(new Set([1, 2]));
  });
});
