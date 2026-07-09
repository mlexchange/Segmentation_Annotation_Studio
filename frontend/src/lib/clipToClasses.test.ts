import { describe, it, expect } from 'vitest';
import { clipShapesToOthers } from './clipToClasses';
import { rasterizeShapes, gridFor } from './rasterize';
import type { Shape } from '@/stores/annotationStore';

const W = 200, H = 200;

function rect(id: string, classId: number, x: number, y: number, w: number, h: number): Shape {
  return { id, classId, kind: 'rectangle', x, y, w, h };
}

/** Foreground pixel count of a shape list on the standard grid. */
function area(shapes: Shape[]): number {
  const { gw, gh, scale } = gridFor(W, H);
  const m = rasterizeShapes(shapes, gw, gh, scale);
  let n = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) n++;
  return n;
}

describe('clipShapesToOthers', () => {
  it('removes the part of a shape that overlaps another class', () => {
    const classA = rect('a', 1, 0, 0, 100, 100);       // existing class A
    const newB = rect('b', 2, 50, 0, 100, 100);         // class B overlaps A in x∈[50,100]
    const clipped = clipShapesToOthers([newB], [classA], W, H);
    expect(clipped.length).toBeGreaterThanOrEqual(1);
    // No clipped-B pixel may fall inside A.
    const { gw, gh, scale } = gridFor(W, H);
    const aMask = rasterizeShapes([classA], gw, gh, scale);
    const bMask = rasterizeShapes(clipped, gw, gh, scale);
    let overlap = 0;
    for (let i = 0; i < aMask.length; i++) if (aMask[i] && bMask[i]) overlap++;
    expect(overlap).toBe(0);
    // Area is roughly halved (the non-overlapping right half remains).
    expect(area(clipped)).toBeLessThan(area([newB]) * 0.7);
    expect(area(clipped)).toBeGreaterThan(0);
  });

  it('leaves a non-overlapping shape essentially unchanged', () => {
    const classA = rect('a', 1, 0, 0, 40, 40);
    const newB = rect('b', 2, 120, 120, 40, 40);        // far from A
    const clipped = clipShapesToOthers([newB], [classA], W, H);
    expect(clipped.length).toBe(1);
    expect(area(clipped)).toBeGreaterThan(area([newB]) * 0.8);
  });

  it('drops a shape fully enclosed by another class', () => {
    const classA = rect('a', 1, 0, 0, 200, 200);        // A covers everything
    const newB = rect('b', 2, 60, 60, 40, 40);          // entirely inside A
    const clipped = clipShapesToOthers([newB], [classA], W, H);
    expect(clipped.length).toBe(0);
  });

  it('drops a fragment fully redundant with the SAME class (spurious SAM blob)', () => {
    const existingB = rect('b0', 2, 40, 40, 60, 60);    // already-labeled class B
    const blob = rect('b1', 2, 55, 55, 20, 20);          // new B blob entirely inside existing B
    const clipped = clipShapesToOthers([blob], [existingB], W, H);
    expect(clipped.length).toBe(0);
  });

  it('keeps a same-class fragment that extends beyond existing labels', () => {
    const existingB = rect('b0', 2, 40, 40, 40, 40);     // existing B
    const extend = rect('b1', 2, 60, 60, 60, 60);        // overlaps + extends past B
    const clipped = clipShapesToOthers([extend], [existingB], W, H);
    expect(clipped.length).toBe(1);
  });

  it('avoids a class enclosed by a holed shape of another class (union bug)', () => {
    // classC (1) is a big ring with a HOLE where classA (3) sits — the exact
    // result of an earlier clip. classA (3) fills that hole. A new class-B (2)
    // region over the center must still avoid BOTH: classA must not be erased
    // from the "other" union by classC's hole carve.
    const classC: Shape = {
      id: 'c', classId: 1, kind: 'polygon',
      points: [40, 40, 160, 40, 160, 160, 40, 160],
      holes: [[80, 80, 120, 80, 120, 120, 80, 120]],
    };
    const classA = rect('a', 3, 80, 80, 40, 40);         // fills classC's hole
    const newB = rect('b', 2, 85, 85, 30, 30);           // fully inside classA
    // classA listed before classC: the old shared-mask union let classC's hole
    // carve classA away, so newB wasn't clipped against it.
    const clipped = clipShapesToOthers([newB], [classA, classC], W, H);
    expect(clipped.length).toBe(0);                       // fully over classA → dropped
  });

  it('carves a hole when the new region encloses another class', () => {
    const classA = rect('a', 1, 90, 90, 20, 20);        // small A blob in the middle
    const newB = rect('b', 2, 40, 40, 120, 120);        // large B surrounds A
    const clipped = clipShapesToOthers([newB], [classA], W, H);
    expect(clipped.length).toBe(1);
    expect(clipped[0].holes && clipped[0].holes.length).toBeGreaterThanOrEqual(1);
    // Rasterizing the clipped B leaves A's center empty (hole carved).
    const { gw, gh, scale } = gridFor(W, H);
    const bMask = rasterizeShapes(clipped, gw, gh, scale);
    expect(bMask[Math.floor((100 / scale)) * gw + Math.floor(100 / scale)]).toBe(0); // A center → empty
    expect(bMask[Math.floor((50 / scale)) * gw + Math.floor(50 / scale)]).toBe(1);   // B body → filled
  });
});
