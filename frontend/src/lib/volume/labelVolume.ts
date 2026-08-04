/**
 * Build a class-index voxel volume from LIVE annotations (the in-memory
 * `annotationStore`, not saved masks) for the 3D tab's overlay. The volume is
 * built on the exact grid a matching raw volume from `GET /api/image/volume`
 * uses (see `volumeDims.ts`), so the two line up voxel-for-voxel.
 */
import { rasterizeUnion } from '../rasterize';
import type { Shape } from '@/stores/annotationStore';
import type { VolumeDims } from './volumeDims';

/**
 * Build a class-index voxel volume from live annotations, on the exact grid
 * a matching raw volume from GET /api/image/volume uses (see volumeDims.ts).
 * Voxel value 0 = background; value (i+1) = classOrder[i]'s class, for the
 * i-th id in classOrder — later classes in classOrder win where shapes of
 * different classes on the same slice overlap.
 */
export function buildLabelVolume(
  slicesForSource: Record<string, Shape[]> | undefined, // annotationStore.byImage[sourceKey]
  classOrder: number[], // e.g. classes.map(c => c.classId)
  dims: VolumeDims
): Uint8Array {
  // length dims.nz * dims.ny * dims.nx
  const { nz, ny, nx, sxy, sz } = dims;
  const out = new Uint8Array(nz * ny * nx);
  if (!slicesForSource) return out;

  const planeSize = ny * nx;

  // Slice keys are decimal-string object keys ("0", "12", …); sorting as
  // strings would put "12" before "2", so sort numerically.
  const sliceKeys = Object.keys(slicesForSource)
    .map(Number)
    .sort((a, b) => a - b);

  for (const j of sliceKeys) {
    const shapes = slicesForSource[String(j)];
    if (!shapes || shapes.length === 0) continue;

    // Nearest-sample mapping from source slice index to the downsampled z
    // index. When sz > 1, multiple source slices can map to the same z;
    // since sliceKeys is processed in ascending numeric order, a LATER
    // source slice's paint wins at a shared z — consistent with "later
    // class wins" semantics elsewhere in this function.
    const z = Math.min(nz - 1, Math.max(0, Math.round(j / sz)));

    // Group this slice's shapes by classId.
    const byClass = new Map<number, Shape[]>();
    for (const shape of shapes) {
      const bucket = byClass.get(shape.classId);
      if (bucket) bucket.push(shape);
      else byClass.set(shape.classId, [shape]);
    }

    // Paint classes in classOrder order, so later classes overwrite earlier
    // ones where they overlap on this slice (intentional "later class wins",
    // matching rasterizeShapes'/rasterizeUnion's per-shape-later-wins wording).
    // A classId with no entry in classOrder (e.g. a class deleted from
    // classStore while its shapes still live in an old draft) is silently
    // skipped: it's simply never looked up here, since we iterate classOrder
    // rather than the shapes' own classIds.
    for (let i = 0; i < classOrder.length; i++) {
      const classShapes = byClass.get(classOrder[i]);
      if (!classShapes || classShapes.length === 0) continue;
      // Classes at position 254+ in classOrder all collapse to voxel value
      // 255 — an intentional clamp for an 8-bit LUT (only 255 non-background
      // values fit in a Uint8Array), not a bug.
      const value = Math.min(i + 1, 255);
      const mask = rasterizeUnion(classShapes, nx, ny, sxy);
      const base = z * planeSize;
      for (let p = 0; p < planeSize; p++) {
        if (mask[p]) out[base + p] = value;
      }
    }
  }

  return out;
}
