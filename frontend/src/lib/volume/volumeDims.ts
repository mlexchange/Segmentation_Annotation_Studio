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
