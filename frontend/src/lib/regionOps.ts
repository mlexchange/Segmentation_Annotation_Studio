/**
 * regionOps — mask-level operations on a selection of shapes, grouped by class.
 *
 * Each op rasterizes the selected shapes of a class into one binary mask, applies
 * a morphology transform, and re-vectorizes to polygons (image coords). Pure (no
 * store / no React) so the select-tool "region ops" preview can be unit-tested.
 */
import type { Shape } from '@/stores/annotationStore';
import { gridFor, rasterizeShapes } from '@/lib/rasterize';
import { maskToPolygons, maskToPolygonsWithHoles } from '@/lib/magicwand';
import { dilate, erode, removeSmallComponents } from '@/lib/morphology';

export type RegionOp = 'merge' | 'grow' | 'shrink' | 'islands';

export interface RegionResult {
  classId: number;
  points: number[];
  /** Inner rings (unlabeled interior gaps) preserved as even-odd holes. */
  holes?: number[][];
}

/**
 * Apply `op` to the selected shapes, grouped by class, returning replacement
 * polygons (image coords). `param` is in image pixels: grow/shrink radius, or the
 * min island area (px²) for `islands`.
 */
export function computeRegionOps(
  shapes: Shape[],
  width: number,
  height: number,
  op: RegionOp,
  param: number,
): RegionResult[] {
  const { gw, gh, scale } = gridFor(width, height);
  const results: RegionResult[] = [];
  const byClass = new Map<number, Shape[]>();
  for (const s of shapes) {
    const arr = byClass.get(s.classId);
    if (arr) arr.push(s); else byClass.set(s.classId, [s]);
  }

  for (const [classId, group] of byClass) {
    let mask = rasterizeShapes(group, gw, gh, scale);
    if (op === 'merge') {
      // Union only — do NOT fill interior gaps that aren't labeled; preserve them
      // as holes so the merged shape covers exactly the labeled pixels.
      for (const p of maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })) {
        if (p.points.length >= 6) {
          results.push({ classId, points: p.points, ...(p.holes.length ? { holes: p.holes } : {}) });
        }
      }
      continue;
    }
    switch (op) {
      case 'grow':
        mask = dilate(mask, gw, gh, Math.max(1, Math.round(param / scale)));
        break;
      case 'shrink':
        mask = erode(mask, gw, gh, Math.max(1, Math.round(param / scale)));
        break;
      case 'islands':
        mask = removeSmallComponents(mask, gw, gh, Math.max(1, Math.round(param / (scale * scale))));
        break;
    }
    for (const points of maskToPolygons(mask, gw, gh, { minRegion: 4, scale })) {
      if (points.length >= 6) results.push({ classId, points });
    }
  }
  return results;
}
