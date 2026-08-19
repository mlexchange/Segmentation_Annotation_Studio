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
import { shapeToMultiPolygon, multiPolygonToShapes, unionShapesChecked } from '@/lib/polybool';

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
  upscale = 1,
): PolygonShape[] {
  const out: PolygonShape[] = [];

  // Unions are computed once per class and carry an `ok` flag. `ok: false` means
  // some shape could not be folded in, so the union UNDER-covers the other classes
  // — clipping against it would leave real overlap. That case must route to the
  // mask path, not proceed: an under-covering union is indistinguishable from
  // "nothing to clip against" if you only look at `mp.length`, which is precisely
  // how complex geometry (e.g. speckled Threshold Brush regions) can silently
  // disable clipping for every later annotation on the slice.
  const otherByClass = new Map<number, { mp: MultiPolygon; ok: boolean }>();
  const sameByClass = new Map<number, { mp: MultiPolygon; ok: boolean }>();
  const otherMP = (c: number) => {
    let m = otherByClass.get(c);
    if (!m) { m = unionShapesChecked(sliceShapes.filter((s) => s.classId !== c), width, height); otherByClass.set(c, m); }
    return m;
  };
  const sameMP = (c: number) => {
    let m = sameByClass.get(c);
    if (!m) { m = unionShapesChecked(sliceShapes.filter((s) => s.classId === c), width, height); sameByClass.set(c, m); }
    return m;
  };

  // Shapes the boolean path can't handle are batched and clipped together at the
  // end: `clipShapesToOthersMask` caches its class masks per CALL, so clipping
  // them one at a time would re-rasterize every shape on the slice per shape.
  const needsMask: Shape[] = [];

  // Group by class so a multi-shape commit can be clipped in ONE boolean pass.
  // This matters enormously for the Threshold Brush, which commits every in-band
  // region of a stroke at once — often 100+ polygons. Differencing them one at a
  // time re-walks the whole other-class union per shape, so commit time grew as
  // (new shapes × slice complexity) and reached seconds on a busy slice. Every
  // other tool commits a single shape and never noticed.
  const byClass = new Map<number, Shape[]>();
  for (const s of newShapes) {
    const arr = byClass.get(s.classId);
    if (arr) arr.push(s); else byClass.set(s.classId, [s]);
  }

  for (const [classId, group] of byClass) {
    const other = otherMP(classId);
    const same = sameMP(classId);
    // Any doubt about the unions → mask clip, which rasterizes every shape
    // independently and so cannot silently under-cover.
    if (!other.ok || !same.ok) { needsMask.push(...group); continue; }

    // A single shape keeps the original per-shape path, which preserves the
    // shape's id on its primary fragment. Batching regenerates ids, which is fine
    // for freshly-committed regions but not worth changing for the common case.
    if (group.length === 1) {
      const res = clipOneBoolean(group[0], other.mp, same.mp, width, height);
      if (res === null) needsMask.push(group[0]); else out.push(...res);
      continue;
    }

    const res = clipBatchBoolean(group, other.mp, same.mp, classId, width, height);
    if (res === null) needsMask.push(...group); else out.push(...res);
  }

  if (needsMask.length) {
    out.push(...clipShapesToOthersMask(needsMask, sliceShapes, width, height, upscale));
  }
  return out;
}

/**
 * Clip a whole group of same-class shapes in one boolean pass.
 *
 * The group is unioned first (disjoint regions — which is what a mask→polygon
 * pass produces — stay separate, so the shape count is unchanged), then a single
 * difference removes the other classes. Returns null on failure so the caller can
 * fall back to the mask path, exactly like `clipOneBoolean`.
 */
function clipBatchBoolean(
  shapes: Shape[],
  otherMP: MultiPolygon,
  sameMP: MultiPolygon,
  classId: number,
  width: number,
  height: number,
): PolygonShape[] | null {
  try {
    const geoms: MultiPolygon[] = [];
    for (const s of shapes) {
      const g = shapeToMultiPolygon(s, width, height);
      if (g.length) geoms.push(g);
    }
    if (geoms.length === 0) return [];

    let mp = geoms.length === 1 ? geoms[0] : polygonClipping.union(geoms[0], ...geoms.slice(1));
    if (otherMP.length) mp = polygonClipping.difference(mp, otherMP);
    if (mp.length === 0) return [];

    const polys = multiPolygonToShapes(mp, classId);
    if (!sameMP.length) return polys;

    // Drop fragments that add nothing beyond this class's existing labels. Done
    // per fragment because a fragment is kept WHOLE if any part of it is new.
    return polys.filter((p) => {
      try {
        return polygonClipping.difference(shapeToMultiPolygon(p, width, height), sameMP).length > 0;
      } catch {
        return true; // can't prove it's redundant — keep it
      }
    });
  } catch {
    return null;
  }
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
  upscale = 1,
): PolygonShape[] {
  // `upscale` keeps sub-pixel geometry (e.g. a Threshold Brush region traced at 2x)
  // from being re-snapped to the native pixel grid by this fallback round-trip.
  const { gw, gh, scale } = fullResGridFor(width, height, upscale);
  const out: PolygonShape[] = [];

  // See the note in `clipShapesToOthers`: the bounds pre-filter is reverted here
  // too, so both clip paths behave exactly as they did before the perf pass.
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
