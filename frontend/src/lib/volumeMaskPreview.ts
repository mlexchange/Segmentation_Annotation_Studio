/**
 * Client-side, no-network rasterization of a sample's CURRENT annotation
 * shapes (every slice, live in the annotation store) into a class-id volume
 * — feeds the 3D view's "Fast (iPred)" mask layer via
 * `WebGpuViewerInstance.loadMaskFromArray(slot, data, dims)` without
 * requiring a "Sync masks to Tiled" round trip first.
 *
 * Unlike a Tiled-backed mask, this does NOT need to match the primary
 * volume's own chosen texture resolution: `sampleMask()` in the viewer's WGSL
 * maps a ray position to the mask's OWN voxel grid independently of the
 * primary's (`frame.maskCtl.yzw` carries the mask's dims) — the two are only
 * required to describe the same normalized [0,1]^3 box, not the same voxel
 * count. That's what makes a coarse, fast, purely-client-side rasterization
 * a legitimate live preview rather than something that needs to negotiate
 * resolution with the renderer.
 *
 * Deliberately approximate, not a port of the backend's exact rasterizer
 * (`coco_export.shape_to_mask`'s per-SHAPE "last one wins" order): shapes are
 * grouped by class and each class's union is painted, so overlapping shapes
 * of DIFFERENT classes resolve by ascending class id rather than draw order.
 * Good enough for "does this look roughly right in 3D," not a source of
 * truth — "Push to Tiled" is what produces the precise result.
 */
import { rasterizeUnion, gridFor } from './rasterize';
import type { Shape } from '@/stores/annotationStore';

/** Longest in-plane edge the live preview rasterizes at. Coarser than the
 * default 2-D mask grid (1600) on purpose — this runs entirely on the main
 * thread for every slice up front, and a 3-D GPU mask texture reads it at
 * whatever resolution it's given (see this module's own doc), so there is no
 * fidelity reason to go higher for a quick preview. */
const LIVE_PREVIEW_MAX_DIM = 256;

export interface LiveMaskVolume {
  data: Uint8Array;
  /** `[width, height, depth]` — the order `loadMaskFromArray` expects. */
  dims: readonly [number, number, number];
}

/**
 * Rasterize every slice of `byImageForSource` (already scoped to one sample)
 * into one `width*height*depth` class-id volume, `depth` = `nSlices` (every
 * slice in the sample gets a slot, annotated or not, so the volume's aspect
 * ratio matches the real dataset rather than just the annotated subset).
 *
 * Returns `null` if there is nothing to rasterize (no shapes anywhere) —
 * callers should treat that as "nothing to load," not an error.
 */
export function buildLiveMaskVolume(
  byImageForSource: Record<string, Shape[]>,
  imageWidth: number,
  imageHeight: number,
  nSlices: number,
  maxDim: number = LIVE_PREVIEW_MAX_DIM,
): LiveMaskVolume | null {
  const hasAnyShape = Object.values(byImageForSource).some((shapes) => shapes.length > 0);
  if (!hasAnyShape || nSlices <= 0) return null;

  const { gw, gh, scale } = gridFor(imageWidth, imageHeight, maxDim);
  const depth = Math.max(1, Math.floor(nSlices));
  const data = new Uint8Array(gw * gh * depth);

  for (let z = 0; z < depth; z++) {
    const shapes = byImageForSource[String(z)];
    if (!shapes || shapes.length === 0) continue;

    const byClass = new Map<number, Shape[]>();
    for (const shape of shapes) {
      const list = byClass.get(shape.classId);
      if (list) list.push(shape);
      else byClass.set(shape.classId, [shape]);
    }

    const slice = data.subarray(z * gw * gh, (z + 1) * gw * gh);
    // Ascending class id so a higher id visually "wins" ties — arbitrary but
    // deterministic, matching mask_pyramid.majority_downsample's own
    // tie-break convention on the backend.
    for (const classId of [...byClass.keys()].sort((a, b) => a - b)) {
      if (classId <= 0 || classId > 255) continue; // 0 is background by convention
      const mask = rasterizeUnion(byClass.get(classId)!, gw, gh, scale);
      for (let i = 0; i < mask.length; i++) {
        if (mask[i]) slice[i] = classId;
      }
    }
  }

  return { data, dims: [gw, gh, depth] };
}
