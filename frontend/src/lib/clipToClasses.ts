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
import { gridFor, rasterizeShapes, rasterizeUnion } from '@/lib/rasterize';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';

/** Does the slice hold any shape of a different class than `classId`? */
export function hasOtherClass(sliceShapes: Shape[], classId: number): boolean {
  return sliceShapes.some((s) => s.classId !== classId);
}

/**
 * Clip `newShapes` against every same-slice shape of a *different* class,
 * returning replacement polygon shapes (image coords). Fragments left with no
 * new area — fully inside another class (subtracted away) OR entirely redundant
 * with the same class's existing labels (e.g. a spurious little SAM blob on top
 * of an already-labeled region) — are dropped. Fragments that add any new area
 * are kept whole (so legitimate extensions/merges are preserved).
 */
export function clipShapesToOthers(
  newShapes: Shape[],
  sliceShapes: Shape[],
  width: number,
  height: number,
): PolygonShape[] {
  const { gw, gh, scale } = gridFor(width, height);
  const out: PolygonShape[] = [];

  // Cached per classId: other = union of DIFFERENT-class shapes (subtracted);
  // same = union of SAME-class shapes (used to drop fully-redundant fragments).
  const otherMaskByClass = new Map<number, Uint8Array>();
  const sameMaskByClass = new Map<number, Uint8Array>();
  const otherMaskFor = (classId: number): Uint8Array => {
    let m = otherMaskByClass.get(classId);
    if (!m) {
      m = rasterizeUnion(sliceShapes.filter((s) => s.classId !== classId), gw, gh, scale);
      otherMaskByClass.set(classId, m);
    }
    return m;
  };
  const sameMaskFor = (classId: number): Uint8Array => {
    let m = sameMaskByClass.get(classId);
    if (!m) {
      m = rasterizeUnion(sliceShapes.filter((s) => s.classId === classId), gw, gh, scale);
      sameMaskByClass.set(classId, m);
    }
    return m;
  };

  for (const shape of newShapes) {
    const other = otherMaskFor(shape.classId);
    const same = sameMaskFor(shape.classId);
    const mine = rasterizeShapes([shape], gw, gh, scale);
    // Subtract other classes; track whether any surviving pixel adds NEW area
    // (outside both other and same-class existing labels).
    let hasNew = false;
    for (let i = 0; i < mine.length; i++) {
      if (other[i]) { mine[i] = 0; continue; } // mine AND NOT other
      if (mine[i] && !same[i]) hasNew = true;
    }
    // Fully covered by existing labels (other or same) → redundant fragment, drop.
    if (!hasNew) continue;
    const polys = maskToPolygonsWithHoles(mine, gw, gh, { minRegion: 4, scale });
    for (let k = 0; k < polys.length; k++) {
      if (polys[k].points.length >= 6) {
        out.push({
          // Reuse the id when the shape maps to a single polygon; fresh ids on split.
          id: k === 0 ? shape.id : uuidv4(),
          classId: shape.classId,
          kind: 'polygon',
          points: polys[k].points,
          ...(polys[k].holes.length ? { holes: polys[k].holes } : {}),
        });
      }
    }
  }
  return out;
}
