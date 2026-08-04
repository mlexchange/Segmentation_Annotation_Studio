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

  const chosen = new Map<string, Shape>(seed.map((s) => [s.id, s]));
  const classes = new Set(seed.map((s) => s.classId));

  for (const classId of classes) {
    const candidates = all.filter((s) => s.classId === classId && !chosen.has(s.id));
    let members = [...chosen.values()].filter((s) => s.classId === classId);
    let changed = true;
    while (changed && candidates.length > 0) {
      changed = false;
      // Union mask of the current cluster members for this class.
      const union = new Uint8Array(gw * gh);
      for (const m of members) {
        const mm = maskOf(m);
        for (let i = 0; i < union.length; i++) if (mm[i]) union[i] = 1;
      }
      for (let i = candidates.length - 1; i >= 0; i--) {
        if (masksIntersect(maskOf(candidates[i]), union)) {
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
): MergeResult {
  // Full resolution so the merged region preserves existing geometry (no erosion).
  const { gw, gh, scale } = fullResGridFor(width, height);
  const add: Shape[] = [];
  const removeIds: string[] = [];

  // Group the new shapes by class; other-class new shapes pass through untouched.
  const byClass = new Map<number, Shape[]>();
  for (const s of newShapes) {
    const arr = byClass.get(s.classId);
    if (arr) arr.push(s); else byClass.set(s.classId, [s]);
  }

  for (const [classId, group] of byClass) {
    const existing = sliceShapes.filter((s) => s.classId === classId);
    if (existing.length === 0) { add.push(...group); continue; }

    // Cheap mask test to decide WHICH existing shapes to merge (fast, tolerant).
    const newMask = rasterizeUnion(group, gw, gh, scale);
    const overlapping = existing.filter((e) => masksIntersect(rasterizeShapes([e], gw, gh, scale), newMask));
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
