import { describe, it, expect } from 'vitest';
import { shapeBBox, unionBBox, bboxIntersects, bboxNear, type BBox } from './geometry';
import { rasterizeShapes, gridFor } from './rasterize';
import type { Shape } from '@/stores/annotationStore';

/** Every set pixel of `shape`, rasterized, must fall inside `box`. This is the
 *  property the pre-filters depend on: the bbox is a true superset. */
function bboxCoversRaster(shape: Shape, box: BBox, width = 200, height = 200): boolean {
  const { gw, gh, scale } = gridFor(width, height);
  const mask = rasterizeShapes([shape], gw, gh, scale);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      if (!mask[gy * gw + gx]) continue;
      // Grid cell (gx,gy) covers image pixels [gx*scale, (gx+1)*scale).
      const x0 = gx * scale, y0 = gy * scale;
      const x1 = x0 + scale, y1 = y0 + scale;
      // Allow one cell of slack — rasterization rounds outward.
      if (x1 < box.x - scale || x0 > box.x + box.w + scale) return false;
      if (y1 < box.y - scale || y0 > box.y + box.h + scale) return false;
    }
  }
  return true;
}

describe('shapeBBox', () => {
  it('bounds a rectangle, normalizing negative extents', () => {
    const s: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 30, y: 40, w: -10, h: 20 };
    expect(shapeBBox(s)).toEqual({ x: 20, y: 40, w: 10, h: 20 });
  });

  it('bounds an ellipse by its radii', () => {
    const s: Shape = { id: 'e', classId: 0, kind: 'ellipse', cx: 50, cy: 60, rx: 10, ry: 5 };
    expect(shapeBBox(s)).toEqual({ x: 40, y: 55, w: 20, h: 10 });
  });

  it('bounds a polygon by its vertices', () => {
    const s: Shape = { id: 'p', classId: 0, kind: 'polygon', points: [10, 10, 40, 12, 25, 35] };
    expect(shapeBBox(s)).toEqual({ x: 10, y: 10, w: 30, h: 25 });
  });

  it('includes polygon holes (superset is always safe)', () => {
    const s: Shape = {
      id: 'p', classId: 0, kind: 'polygon',
      points: [0, 0, 100, 0, 100, 100, 0, 100],
      holes: [[20, 20, 40, 20, 40, 40, 20, 40]],
    };
    const b = shapeBBox(s);
    expect(b).toEqual({ x: 0, y: 0, w: 100, h: 100 });
  });

  it('expands brush strokes by their radius', () => {
    const s: Shape = {
      id: 'b', classId: 0, kind: 'brush',
      strokes: [{ points: [50, 50, 60, 50], radius: 5, mode: 'paint' }],
    };
    expect(shapeBBox(s)).toEqual({ x: 45, y: 45, w: 20, h: 10 });
  });

  it('covers every rasterized pixel, for each shape kind', () => {
    const shapes: Shape[] = [
      { id: 'r', classId: 0, kind: 'rectangle', x: 20, y: 30, w: 60, h: 40 },
      { id: 'e', classId: 0, kind: 'ellipse', cx: 100, cy: 90, rx: 30, ry: 18 },
      { id: 'p', classId: 0, kind: 'polygon', points: [10, 10, 90, 20, 70, 80, 15, 60] },
      { id: 'b', classId: 0, kind: 'brush', strokes: [{ points: [30, 30, 120, 140, 60, 150], radius: 9, mode: 'paint' }] },
    ];
    for (const s of shapes) {
      expect(bboxCoversRaster(s, shapeBBox(s))).toBe(true);
    }
  });

  it('returns an empty box for a degenerate shape', () => {
    const s: Shape = { id: 'p', classId: 0, kind: 'polygon', points: [] };
    const b = shapeBBox(s);
    expect(b.w).toBeLessThan(0);
    expect(bboxNear(b, { x: 0, y: 0, w: 10, h: 10 })).toBe(false);
  });
});

describe('unionBBox', () => {
  it('covers all inputs', () => {
    const shapes: Shape[] = [
      { id: 'a', classId: 0, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 },
      { id: 'b', classId: 0, kind: 'rectangle', x: 90, y: 80, w: 10, h: 20 },
    ];
    expect(unionBBox(shapes)).toEqual({ x: 0, y: 0, w: 100, h: 100 });
  });

  it('is empty for no shapes, and skips degenerate members', () => {
    expect(unionBBox([]).w).toBeLessThan(0);
    const shapes: Shape[] = [
      { id: 'empty', classId: 0, kind: 'polygon', points: [] },
      { id: 'a', classId: 0, kind: 'rectangle', x: 5, y: 5, w: 10, h: 10 },
    ];
    expect(unionBBox(shapes)).toEqual({ x: 5, y: 5, w: 10, h: 10 });
  });
});

describe('bboxIntersects / bboxNear', () => {
  const a: BBox = { x: 0, y: 0, w: 10, h: 10 };

  it('detects overlap and separation', () => {
    expect(bboxIntersects(a, { x: 5, y: 5, w: 10, h: 10 })).toBe(true);
    expect(bboxIntersects(a, { x: 20, y: 0, w: 5, h: 5 })).toBe(false);
  });

  it('bboxNear accepts exactly-touching boxes that the strict test rejects', () => {
    const touching: BBox = { x: 10, y: 0, w: 5, h: 5 };
    expect(bboxIntersects(a, touching)).toBe(false);
    expect(bboxNear(a, touching, 1)).toBe(true);
  });

  it('bboxNear honours the pad', () => {
    const gap: BBox = { x: 13, y: 0, w: 5, h: 5 };
    expect(bboxNear(a, gap, 1)).toBe(false);
    expect(bboxNear(a, gap, 4)).toBe(true);
  });

  it('bboxNear rejects degenerate boxes', () => {
    expect(bboxNear(a, { x: 0, y: 0, w: -1, h: -1 })).toBe(false);
  });
});
