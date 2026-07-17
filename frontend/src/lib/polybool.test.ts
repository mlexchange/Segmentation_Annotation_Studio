import { describe, it, expect } from 'vitest';
import { unionShapesToPolygons, shapeToMultiPolygon, multiPolygonToShapes, eraseStampToMultiPolygon, subtractFromShape } from './polybool';
import type { Shape } from '@/stores/annotationStore';

const W = 200, H = 200;

function poly(id: string, classId: number, points: number[]): Shape {
  return { id, classId, kind: 'polygon', points };
}

/** Does a flat ring contain the exact vertex (x,y)? */
function hasVertex(points: number[], x: number, y: number): boolean {
  for (let i = 0; i + 1 < points.length; i += 2) {
    if (Math.abs(points[i] - x) < 1e-6 && Math.abs(points[i + 1] - y) < 1e-6) return true;
  }
  return false;
}

describe('unionShapesToPolygons', () => {
  it('merges two overlapping squares into one polygon', () => {
    const a = poly('a', 1, [0, 0, 100, 0, 100, 100, 0, 100]);
    const b = poly('b', 1, [50, 50, 150, 50, 150, 150, 50, 150]);
    const out = unionShapesToPolygons([a, b], 1, W, H);
    expect(out).not.toBeNull();
    expect(out!.length).toBe(1);
    // The union's non-overlapping outer corners survive as exact vertices.
    expect(hasVertex(out![0].points, 0, 0)).toBe(true);
    expect(hasVertex(out![0].points, 150, 150)).toBe(true);
  });

  it('preserves a single shape\'s exact vertices (no resampling)', () => {
    const pts = [10, 10, 90, 20, 80, 95, 15, 80];
    const out = unionShapesToPolygons([poly('a', 2, pts)], 2, W, H);
    expect(out).not.toBeNull();
    expect(out!.length).toBe(1);
    for (let i = 0; i < pts.length; i += 2) {
      expect(hasVertex(out![0].points, pts[i], pts[i + 1])).toBe(true);
    }
  });

  it('keeps two disjoint shapes as two polygons', () => {
    const a = poly('a', 1, [0, 0, 20, 0, 20, 20, 0, 20]);
    const b = poly('b', 1, [100, 100, 140, 100, 140, 140, 100, 140]);
    const out = unionShapesToPolygons([a, b], 1, W, H);
    expect(out).not.toBeNull();
    expect(out!.length).toBe(2);
  });

  it('subtracts an erase stamp while preserving the far vertices', () => {
    const square = poly('s', 1, [0, 0, 100, 0, 100, 100, 0, 100]);
    // A stroke biting the right edge near the middle.
    const stamp = eraseStampToMultiPolygon([100, 50, 130, 50], 12, W, H);
    const out = subtractFromShape(square, stamp, W, H);
    expect(out).not.toBeNull();
    expect(out!.length).toBeGreaterThanOrEqual(1);
    // The untouched left corners survive as exact vertices.
    expect(hasVertex(out![0].points, 0, 0)).toBe(true);
    expect(hasVertex(out![0].points, 0, 100)).toBe(true);
  });

  it('returns [] when the erase stamp fully covers the shape', () => {
    const small = poly('s', 1, [40, 40, 60, 40, 60, 60, 40, 60]);
    const stamp = eraseStampToMultiPolygon([30, 50, 70, 50], 40, W, H); // big disk over it
    const out = subtractFromShape(small, stamp, W, H);
    expect(out).not.toBeNull();
    expect(out!.length).toBe(0);
  });

  it('round-trips a holed polygon through geometry conversion', () => {
    const ring: Shape = {
      id: 'r', classId: 1, kind: 'polygon',
      points: [20, 20, 180, 20, 180, 180, 20, 180],
      holes: [[80, 80, 120, 80, 120, 120, 80, 120]],
    };
    const mp = shapeToMultiPolygon(ring, W, H);
    const back = multiPolygonToShapes(mp, 1);
    expect(back.length).toBe(1);
    expect(back[0].holes?.length).toBe(1);
  });
});
