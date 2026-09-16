import { describe, expect, it } from 'vitest';
import { measureRegion } from './measure';
import type { Shape } from '@/stores/annotationStore';

describe('measureRegion', () => {
  it('returns an all-zero/null measurement for no shapes', () => {
    const m = measureRegion([], 100, 100);
    expect(m).toEqual({ count: 0, areaPx: 0, perimeterPx: null, centroid: null, bbox: null });
  });

  it('measures a single rectangle: area, perimeter, centroid, bbox', () => {
    const rect: Shape = { id: 'r1', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 20, h: 10 };
    const m = measureRegion([rect], 100, 100);
    expect(m.count).toBe(1);
    // area ~ 20*10 = 200 (rasterized, so allow a little slack)
    expect(m.areaPx).toBeGreaterThan(150);
    expect(m.areaPx).toBeLessThanOrEqual(231);
    // analytic perimeter = 2*(20+10) = 60
    expect(m.perimeterPx).toBe(60);
    expect(m.centroid).not.toBeNull();
    expect(m.centroid!.x).toBeCloseTo(20, 0); // center x ~ 10+20/2
    expect(m.centroid!.y).toBeCloseTo(15, 0);
    expect(m.bbox).not.toBeNull();
    // bbox is (maxX-minX+1)*scale, so it can run 1px wider than the nominal size.
    expect(m.bbox!.w).toBeGreaterThanOrEqual(20);
    expect(m.bbox!.w).toBeLessThanOrEqual(21);
    expect(m.bbox!.h).toBeGreaterThanOrEqual(10);
    expect(m.bbox!.h).toBeLessThanOrEqual(11);
  });

  it('sums perimeter and area across multiple shapes (count reflects shapes, not area)', () => {
    const r1: Shape = { id: 'r1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 };
    const r2: Shape = { id: 'r2', classId: 1, kind: 'rectangle', x: 50, y: 50, w: 10, h: 10 };
    const m = measureRegion([r1, r2], 100, 100);
    expect(m.count).toBe(2);
    expect(m.perimeterPx).toBe(80); // 40 + 40
  });

  it('computes ellipse perimeter via the Ramanujan approximation', () => {
    const ellipse: Shape = { id: 'e1', classId: 1, kind: 'ellipse', cx: 50, cy: 50, rx: 10, ry: 10 };
    const m = measureRegion([ellipse], 100, 100);
    // A circle of radius 10: circumference = 2*pi*10 ≈ 62.83
    expect(m.perimeterPx).toBeCloseTo(2 * Math.PI * 10, 0);
  });

  it('computes polygon perimeter as the closed-ring length, including holes', () => {
    // A 10x10 square polygon (ring perimeter 40).
    const square: Shape = {
      id: 'p1', classId: 1, kind: 'polygon',
      points: [0, 0, 10, 0, 10, 10, 0, 10],
    };
    const m1 = measureRegion([square], 100, 100);
    expect(m1.perimeterPx).toBeCloseTo(40, 5);

    // Same square with a 4x4 hole adds the hole's ring perimeter (16).
    const withHole: Shape = {
      id: 'p2', classId: 1, kind: 'polygon',
      points: [0, 0, 10, 0, 10, 10, 0, 10],
      holes: [[3, 3, 7, 3, 7, 7, 3, 7]],
    };
    const m2 = measureRegion([withHole], 100, 100);
    expect(m2.perimeterPx).toBeCloseTo(40 + 16, 5);
  });

  it('returns null perimeter when any shape lacks a defined perimeter (brush)', () => {
    const rect: Shape = { id: 'r1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 };
    const brushShape: Shape = {
      id: 'b1', classId: 1, kind: 'brush',
      strokes: [{ points: [20, 20, 30, 30], radius: 3, mode: 'paint' }],
    };
    const m = measureRegion([rect, brushShape], 100, 100);
    expect(m.perimeterPx).toBeNull();
    // Geometry (area/centroid/bbox) is still computed from the union of both shapes.
    expect(m.count).toBe(2);
    expect(m.areaPx).toBeGreaterThan(0);
    expect(m.bbox).not.toBeNull();
  });

  it('treats overlapping shapes as a union for area/centroid/bbox (not double-counted)', () => {
    const a: Shape = { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 };
    const b: Shape = { id: 'b', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 }; // identical, fully overlapping
    const single = measureRegion([a], 100, 100);
    const overlapping = measureRegion([a, b], 100, 100);
    expect(overlapping.areaPx).toBeCloseTo(single.areaPx, 5);
    // count still reflects the number of shapes passed in, not the union region count.
    expect(overlapping.count).toBe(2);
  });

  it('degenerate zero-size rectangle yields a definite (non-crashing) result with zero perimeter contribution area', () => {
    const zero: Shape = { id: 'z', classId: 1, kind: 'rectangle', x: 5, y: 5, w: 0, h: 0 };
    const m = measureRegion([zero], 100, 100);
    expect(m.count).toBe(1);
    expect(m.perimeterPx).toBe(0);
  });
});
