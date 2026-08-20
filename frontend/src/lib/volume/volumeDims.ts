/**
 * Shared raw/label volume grid for the 3D tab.
 *
 * The backend serves a downsampled raw-intensity volume from
 * `GET /api/image/volume` (see `backend/volumes.py`'s `volume_dims`, which
 * this mirrors byte-for-byte); the frontend rasterizes its OWN class-label
 * volume from live annotations (see `labelVolume.ts`) onto the exact same
 * grid. The two must agree on dimensions voxel-for-voxel, or the annotation
 * overlay renders offset from the raw data it describes.
 *
 * `nx`/`ny`/`sxy` are delegated to the already-shared `gridFor` (used
 * elsewhere for 2-D mask rasterization) rather than reimplemented here, so
 * this and the backend's Python port can never independently drift on the
 * in-plane stride math.
 */
import { gridFor } from '../rasterize';

export interface VolumeDims {
  nz: number;
  ny: number;
  nx: number;
  sxy: number;
  sz: number;
}

/**
 * Grid dimensions for an `nSlices x height x width` source downsampled so no
 * axis exceeds `maxDim`. Mirrors `backend/volumes.py: volume_dims` exactly.
 */
export function volumeDimsFor(nSlices: number, height: number, width: number, maxDim: number): VolumeDims {
  const { gw: nx, gh: ny, scale: sxy } = gridFor(width, height, maxDim);
  const sz = nSlices <= maxDim ? 1 : Math.max(1, Math.ceil(nSlices / maxDim));
  const nz = Math.ceil(nSlices / sz);
  return { nz, ny, nx, sxy, sz };
}

/** Quality ladder offered by the 3-D tab. Mirrors `volumes.QUALITY_LADDER`. */
export const QUALITY_LADDER = [128, 192, 256, 320, 384, 512, 640, 768, 1024] as const;

/**
 * Voxel budget for ONE volume, in uint8 bytes. Mirrors
 * `volumes.MAX_VOLUME_VOXELS`. The tab holds two (raw + the label volume
 * rasterized here), each also uploaded as a GL 3-D texture, so the real
 * footprint is about double.
 */
export const MAX_VOLUME_VOXELS = 192 * 1024 * 1024;

/**
 * Largest ladder quality <= `requested` that fits the voxel budget.
 * Mirrors `backend/volumes.py: effective_max_dim` exactly.
 *
 * Capping per-axis alone is the wrong constraint: it makes a thin stack
 * needlessly coarse (63 x 3232² at 1024 is only 41 MB) while letting a cubic
 * source reach 1024³ = 1 GB. Budgeting total voxels lets an anisotropic stack
 * reach full quality and holds a cubic one to roughly 570³.
 *
 * Both sides clamp identically so the label volume rasterized here lands on the
 * same grid as the raw volume the server returns — otherwise a server-side
 * downgrade would surface as an overlay offset from the data it describes.
 */
export function effectiveMaxDim(
  nSlices: number, height: number, width: number, requested: number,
): number {
  const fits = (quality: number) => {
    const d = volumeDimsFor(nSlices, height, width, quality);
    return d.nz * d.ny * d.nx <= MAX_VOLUME_VOXELS;
  };
  // Only ever REDUCES — a request that already fits passes through untouched,
  // so arbitrary (non-ladder) values are preserved exactly.
  if (fits(requested)) return requested;
  const below = QUALITY_LADDER.filter((q) => q < requested);
  for (let i = below.length - 1; i >= 0; i -= 1) {
    if (fits(below[i])) return below[i];
  }
  return Math.min(requested, QUALITY_LADDER[0]);
}
