/**
 * clipToClasses — clip newly-drawn shapes so they can't overlap OTHER classes'
 * regions on the same slice (neighbor classes act as a hard boundary).
 *
 * Each new shape is rasterized, the union of other-class shapes is subtracted,
 * and the remainder is re-vectorized to polygon(s). Pure (no store / React) so
 * it's unit-testable and reusable across every commit site. Adjacent boundaries
 * clip exactly; a fully-enclosed neighbor would need a hole (maskToPolygons only
 * traces outer contours) — acceptable for v1.
 */
import { v4 as uuidv4 } from 'uuid';
import type { PolygonShape, Shape } from '@/stores/annotationStore';
import { gridFor, rasterizeShapes } from '@/lib/rasterize';
import { maskToPolygons } from '@/lib/magicwand';

/** Does the slice hold any shape of a different class than `classId`? */
export function hasOtherClass(sliceShapes: Shape[], classId: number): boolean {
  return sliceShapes.some((s) => s.classId !== classId);
}

/**
 * Clip `newShapes` against every same-slice shape of a *different* class,
 * returning replacement polygon shapes (image coords). Shapes with no remaining
 * area are dropped. Non-overlapping shapes still round-trip to a polygon.
 */
export function clipShapesToOthers(
  newShapes: Shape[],
  sliceShapes: Shape[],
  width: number,
  height: number,
): PolygonShape[] {
  const { gw, gh, scale } = gridFor(width, height);
  const out: PolygonShape[] = [];

  // Cache the other-class mask per classId (usually one active class → one mask).
  const otherMaskByClass = new Map<number, Uint8Array>();
  const otherMaskFor = (classId: number): Uint8Array => {
    let m = otherMaskByClass.get(classId);
    if (!m) {
      m = rasterizeShapes(sliceShapes.filter((s) => s.classId !== classId), gw, gh, scale);
      otherMaskByClass.set(classId, m);
    }
    return m;
  };

  for (const shape of newShapes) {
    const other = otherMaskFor(shape.classId);
    const mine = rasterizeShapes([shape], gw, gh, scale);
    for (let i = 0; i < mine.length; i++) if (other[i]) mine[i] = 0; // mine AND NOT other
    const polys = maskToPolygons(mine, gw, gh, { minRegion: 4, scale });
    for (let k = 0; k < polys.length; k++) {
      if (polys[k].length >= 6) {
        out.push({
          // Reuse the id when the shape maps to a single polygon; fresh ids on split.
          id: k === 0 ? shape.id : uuidv4(),
          classId: shape.classId,
          kind: 'polygon',
          points: polys[k],
        });
      }
    }
  }
  return out;
}
