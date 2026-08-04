import { describe, it, expect } from 'vitest';
import { volumeDimsFor } from './volumeDims';

// Hand-verified against the Python `volumes.volume_dims` reference
// implementation — if a row disagrees, the bug is in volumeDimsFor, not here.
const TABLE: Array<{
  nSlices: number; h: number; w: number; maxDim: number;
  sxy: number; ny: number; nx: number; sz: number; nz: number;
}> = [
  { nSlices: 1, h: 512, w: 512, maxDim: 256, sxy: 2, ny: 256, nx: 256, sz: 1, nz: 1 },
  { nSlices: 700, h: 1000, w: 900, maxDim: 256, sxy: 4, ny: 250, nx: 225, sz: 3, nz: 234 },
  { nSlices: 700, h: 1000, w: 900, maxDim: 384, sxy: 3, ny: 333, nx: 300, sz: 2, nz: 350 },
  { nSlices: 100, h: 515, w: 100, maxDim: 256, sxy: 3, ny: 171, nx: 33, sz: 1, nz: 100 },
  { nSlices: 5, h: 10000, w: 10, maxDim: 256, sxy: 40, ny: 250, nx: 1, sz: 1, nz: 5 },
];

describe('volumeDimsFor', () => {
  it.each(TABLE)(
    'nSlices=$nSlices h=$h w=$w maxDim=$maxDim -> sxy=$sxy ny=$ny nx=$nx sz=$sz nz=$nz',
    ({ nSlices, h, w, maxDim, sxy, ny, nx, sz, nz }) => {
      const dims = volumeDimsFor(nSlices, h, w, maxDim);
      expect(dims).toEqual({ nz, ny, nx, sxy, sz });
    }
  );
});
