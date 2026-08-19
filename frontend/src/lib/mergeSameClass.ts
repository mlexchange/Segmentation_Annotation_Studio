/**
 * mergeSameClass — union newly-committed shapes with existing shapes of the SAME
 * class that they overlap, so overlapping regions of one class become a single
 * merged shape instead of stacked duplicates.
 *
 * Per class: rasterize the new shapes, find existing same-class shapes whose masks
 * intersect them, then re-vectorize the union (new + those existing) into merged
 * polygons. Non-overlapping new shapes are kept as-is (native kind preserved).
 */
import type { Shape } from '@/stores/annotationStore';
import { gridFor, fullResGridFor, rasterizeShapes, rasterizeUnion } from '@/lib/rasterize';
import { shapeBBox, unionBBox, bboxNear, type BBox } from '@/lib/geometry';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import { unionShapesToPolygons } from '@/lib/polybool';
import { v4 as uuidv4 } from 'uuid';

export interface MergeResult {
  /** Shapes to add (merged polygons + untouched new shapes). */
  add: Shape[];
  /** Ids of existing same-class shapes consumed by a merge (to remove). */
  removeIds: string[];
}

/** True if two 0/1 masks share any set pixel. */
function masksIntersect(a: Uint8Array, b: Uint8Array): boolean {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] && b[i]) return true;
  return false;
}

/**
 * True if two 0/1 masks share a set pixel, scanning only the rows/columns where
 * both shapes' bounds overlap. Equivalent to `masksIntersect` — outside the
 * overlap rect at least one mask is empty by construction — but it avoids walking
 * a multi-megapixel grid to answer a question about a small region.
 */
function masksIntersectIn(
  a: Uint8Array, b: Uint8Array, gw: number, gh: number, rect: BBox, scale: number,
): boolean {
  const x0 = Math.max(0, Math.floor(rect.x / scale));
  const y0 = Math.max(0, Math.floor(rect.y / scale));
  const x1 = Math.min(gw - 1, Math.ceil((rect.x + rect.w) / scale));
  const y1 = Math.min(gh - 1, Math.ceil((rect.y + rect.h) / scale));
  for (let y = y0; y <= y1; y++) {
    const row = y * gw;
    for (let x = x0; x <= x1; x++) {
      const i = row + x;
      if (a[i] && b[i]) return true;
    }
  }
  return false;
}

/** Grow a box by `pad` on every side. */
function inflate(b: BBox, pad: number): BBox {
  return { x: b.x - pad, y: b.y - pad, w: b.w + pad * 2, h: b.h + pad * 2 };
}

/**
 * Intersection of two boxes, each first grown by `pad` (empty w/h < 0 when they
 * remain disjoint). The padding absorbs rasterization rounding: two shapes whose
 * boxes merely abut can still land set pixels in the same grid cell.
 */
function bboxOverlap(a: BBox, b: BBox, pad = 0): BBox {
  const A = inflate(a, pad), B = inflate(b, pad);
  const x = Math.max(A.x, B.x);
  const y = Math.max(A.y, B.y);
  const w = Math.min(A.x + A.w, B.x + B.w) - x;
  const h = Math.min(A.y + A.h, B.y + B.h) - y;
  return { x, y, w, h };
}

/**
 * Expand a seed selection to the transitive closure of same-class shapes that
 * overlap it — so selecting one region in an overlapping same-class cluster
 * yields the whole cluster (for merging on a selection).
 */
export function expandSameClassOverlap(
  seed: Shape[],
  all: Shape[],
  width: number,
  height: number,
): Shape[] {
  if (seed.length === 0) return seed;
  const { gw, gh, scale } = gridFor(width, height);
  const maskCache = new Map<string, Uint8Array>();
  const maskOf = (s: Shape): Uint8Array => {
    let m = maskCache.get(s.id);
    if (!m) { m = rasterizeShapes([s], gw, gh, scale); maskCache.set(s.id, m); }
    return m;
  };
  const boxCache = new Map<string, BBox>();
  const boxOf = (s: Shape): BBox => {
    let b = boxCache.get(s.id);
    if (!b) { b = shapeBBox(s); boxCache.set(s.id, b); }
    return b;
  };

  const chosen = new Map<string, Shape>(seed.map((s) => [s.id, s]));
  const classes = new Set(seed.map((s) => s.classId));

  for (const classId of classes) {
    const candidates = all.filter((s) => s.classId === classId && !chosen.has(s.id));
    let members = [...chosen.values()].filter((s) => s.classId === classId);
    let changed = true;
    while (changed && candidates.length > 0) {
      changed = false;
      // Union mask + bounds of the current cluster members for this class. The
      // bounds let a candidate be rejected without rasterizing it at all, which
      // matters because this loop is quadratic in cluster size.
      const union = new Uint8Array(gw * gh);
      for (const m of members) {
        const mm = maskOf(m);
        for (let i = 0; i < union.length; i++) if (mm[i]) union[i] = 1;
      }
      const unionBounds = unionBBox(members);
      for (let i = candidates.length - 1; i >= 0; i--) {
        const cand = candidates[i];
        if (!bboxNear(boxOf(cand), unionBounds, scale + 1)) continue;
        if (masksIntersect(maskOf(cand), union)) {
          const c = candidates.splice(i, 1)[0];
          chosen.set(c.id, c);
          members.push(c);
          changed = true;
        }
      }
    }
  }
  return [...chosen.values()];
}

/**
 * Merge each new shape with the existing same-class shapes it overlaps.
 *
 * @param newShapes   shapes about to be committed.
 * @param sliceShapes shapes already on the slice.
 * @param width,height image dimensions (px).
 */
export function mergeNewWithSameClass(
  newShapes: Shape[],
  sliceShapes: Shape[],
  width: number,
  height: number,
  upscale = 1,
): MergeResult {
  // Full resolution so the merged region preserves existing geometry (no erosion);
  // `upscale` additionally preserves sub-pixel detail from an upscaled working grid.
  const { gw, gh, scale } = fullResGridFor(width, height, upscale);
  const add: Shape[] = [];
  const removeIds: string[] = [];

  // Group the new shapes by class; other-class new shapes pass through untouched.
  const byClass = new Map<number, Shape[]>();
  for (const s of newShapes) {
    const arr = byClass.get(s.classId);
    if (arr) arr.push(s); else byClass.set(s.classId, [s]);
  }

  for (const [classId, group] of byClass) {
    const groupBounds = unionBBox(group);
    // Bounds pre-filter before ANY rasterization: a shape whose box is disjoint
    // from the new geometry cannot share a set pixel with it, so the mask test
    // below would always say no. Without this, committing one stroke rasterizes
    // every same-class shape on the slice into its own full-resolution buffer
    // (millions of cells each) purely to be told they don't touch.
    const existing = sliceShapes.filter(
      (s) => s.classId === classId && bboxNear(shapeBBox(s), groupBounds, scale + 1),
    );
    if (existing.length === 0) { add.push(...group); continue; }

    // Cheap mask test to decide WHICH of the remaining candidates to merge.
    // One reused scratch buffer instead of an allocation per candidate, and the
    // comparison only walks the rows/columns where the two boxes actually overlap.
    const newMask = rasterizeUnion(group, gw, gh, scale);
    const scratch = new Uint8Array(gw * gh);
    const overlapping = existing.filter((e) => {
      const rect = bboxOverlap(shapeBBox(e), groupBounds, scale + 1);
      if (rect.w < 0 || rect.h < 0) return false;
      scratch.fill(0);
      rasterizeShapes([e], gw, gh, scale, scratch);
      return masksIntersectIn(scratch, newMask, gw, gh, rect, scale);
    });
    if (overlapping.length === 0) { add.push(...group); continue; }

    // Combine via a true polygon boolean union so the existing shapes keep their
    // exact vertices (only the merge seam changes). Fall back to a rasterize→
    // re-vectorize union if the boolean op fails (self-intersecting input, etc.).
    let polys = unionShapesToPolygons([...group, ...overlapping], classId, width, height);
    if (!polys) {
      const mask = rasterizeShapes([...group, ...overlapping], gw, gh, scale);
      polys = maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
        .filter((p) => p.points.length >= 6)
        .map((p) => ({
          id: uuidv4(),
          classId,
          kind: 'polygon' as const,
          points: p.points,
          ...(p.holes.length ? { holes: p.holes } : {}),
        }));
    }

    if (polys.length === 0) { add.push(...group); continue; } // fallback: keep originals
    add.push(...polys);
    removeIds.push(...overlapping.map((e) => e.id));
  }

  return { add, removeIds };
}
