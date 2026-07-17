/**
 * clipToClasses — clip newly-drawn shapes so they can't overlap OTHER classes'
 * regions on the same slice (neighbor classes act as a hard boundary).
 *
 * Primary path: a true polygon boolean DIFFERENCE against the union of other-class
 * shapes. This makes the clipped edge coincide EXACTLY with the neighbor's edge, so
 * adjacent classes tile flush with no unlabeled gap (a mask round-trip left a ~1px
 * seam because the traced edge landed a pixel off the neighbor's polygon edge). A
 * fragment adding no area beyond the SAME class's existing labels is dropped
 * (spurious blob). Falls back per-shape to the mask implementation on any failure.
 */
import { v4 as uuidv4 } from 'uuid';
import polygonClipping, { type MultiPolygon } from 'polygon-clipping';
import type { PolygonShape, Shape } from '@/stores/annotationStore';
import { fullResGridFor, rasterizeShapes, rasterizeUnion } from '@/lib/rasterize';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import { shapeToMultiPolygon, multiPolygonToShapes, unionShapesToMultiPolygon } from '@/lib/polybool';

/** Does the slice hold any shape of a different class than `classId`? */
export function hasOtherClass(sliceShapes: Shape[], classId: number): boolean {
  return sliceShapes.some((s) => s.classId !== classId);
}

/**
 * Boolean clip of one shape: subtract other-class regions, then drop it if what
 * remains is fully redundant with the same class's existing labels. Returns the
 * clipped polygon(s) (possibly `[]` = dropped) or `null` if the boolean op can't
 * run, so the caller can fall back to the mask path.
 */
function clipOneBoolean(
  shape: Shape,
  otherMP: MultiPolygon,
  sameMP: MultiPolygon,
  width: number,
  height: number,
): PolygonShape[] | null {
  try {
    let mp = shapeToMultiPolygon(shape, width, height);
    if (mp.length === 0) return null;
    if (otherMP.length) mp = polygonClipping.difference(mp, otherMP);
    if (mp.length === 0) return []; // fully inside another class → drop
    // Redundant if the remainder adds nothing beyond existing same-class labels.
    if (sameMP.length && polygonClipping.difference(mp, sameMP).length === 0) return [];
    const polys = multiPolygonToShapes(mp, shape.classId);
    if (polys.length) polys[0].id = shape.id; // reuse id for the primary fragment
    return polys;
  } catch {
    return null;
  }
}

/**
 * Clip `newShapes` against every same-slice shape of a *different* class,
 * returning replacement polygon shapes (image coords). Fragments left with no
 * new area — fully inside another class, or entirely redundant with the same
 * class's existing labels — are dropped; fragments that add area are kept whole.
 */
export function clipShapesToOthers(
  newShapes: Shape[],
  sliceShapes: Shape[],
  width: number,
  height: number,
): PolygonShape[] {
  const out: PolygonShape[] = [];
  const otherByClass = new Map<number, MultiPolygon>();
  const sameByClass = new Map<number, MultiPolygon>();
  const otherMP = (c: number) => {
    let m = otherByClass.get(c);
    if (!m) { m = unionShapesToMultiPolygon(sliceShapes.filter((s) => s.classId !== c), width, height); otherByClass.set(c, m); }
    return m;
  };
  const sameMP = (c: number) => {
    let m = sameByClass.get(c);
    if (!m) { m = unionShapesToMultiPolygon(sliceShapes.filter((s) => s.classId === c), width, height); sameByClass.set(c, m); }
    return m;
  };

  for (const shape of newShapes) {
    const res = clipOneBoolean(shape, otherMP(shape.classId), sameMP(shape.classId), width, height);
    if (res === null) out.push(...clipShapesToOthersMask([shape], sliceShapes, width, height));
    else out.push(...res);
  }
  return out;
}

/**
 * Mask-based clip (fallback): rasterize each new shape, subtract the union of
 * other-class shapes, drop fully-redundant fragments, re-vectorize the remainder.
 * Robust for degenerate/self-intersecting geometry the boolean path can't handle.
 */
export function clipShapesToOthersMask(
  newShapes: Shape[],
  sliceShapes: Shape[],
  width: number,
  height: number,
): PolygonShape[] {
  const { gw, gh, scale } = fullResGridFor(width, height);
  const out: PolygonShape[] = [];

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
    let hasNew = false;
    for (let i = 0; i < mine.length; i++) {
      if (other[i]) { mine[i] = 0; continue; } // mine AND NOT other
      if (mine[i] && !same[i]) hasNew = true;
    }
    if (!hasNew) continue;
    const polys = maskToPolygonsWithHoles(mine, gw, gh, { minRegion: 4, scale });
    for (let k = 0; k < polys.length; k++) {
      if (polys[k].points.length >= 6) {
        out.push({
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
