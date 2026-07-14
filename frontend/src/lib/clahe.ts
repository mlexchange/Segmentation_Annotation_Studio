/**
 * CLAHE — Contrast-Limited Adaptive Histogram Equalization (display-only).
 *
 * Enhances local contrast on low-contrast scans (tomography/microscopy) so faint
 * features become visible for annotation. Operates on the LUMINANCE channel and
 * rescales RGB by Y'/Y, so hue is preserved. Tiled histogram equalization with a
 * clip limit (excess redistributed) and bilinear interpolation between tile maps
 * to avoid block artifacts.
 *
 * Mutates the RGBA Uint8ClampedArray in place (alpha untouched). Pure/DOM-free so
 * it's unit-testable; the display pipeline calls it on an ImageData buffer.
 */
export interface ClaheOptions {
  /** Tile grid columns (default 8). */
  tilesX?: number;
  /** Tile grid rows (default 8). */
  tilesY?: number;
  /** Clip limit as a multiple of the average bin count (default 2; higher = stronger). */
  clipLimit?: number;
}

/** Apply CLAHE to an RGBA pixel buffer in place. */
export function applyCLAHE(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  opts: ClaheOptions = {},
): void {
  const tilesX = Math.max(1, Math.floor(opts.tilesX ?? 8));
  const tilesY = Math.max(1, Math.floor(opts.tilesY ?? 8));
  const clipMul = opts.clipLimit ?? 2;
  const nPix = width * height;
  if (nPix === 0) return;

  // Luminance per pixel.
  const lum = new Float32Array(nPix);
  for (let i = 0, p = 0; p < nPix; i += 4, p++) {
    lum[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  const tileW = width / tilesX;
  const tileH = height / tilesY;

  // Per-tile intensity→intensity mapping (256 entries) from a clipped CDF.
  const maps: Uint8Array[] = new Array(tilesX * tilesY);
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = Math.floor(tx * tileW), x1 = Math.floor((tx + 1) * tileW);
      const y0 = Math.floor(ty * tileH), y1 = Math.floor((ty + 1) * tileH);
      const hist = new Float32Array(256);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const v = lum[y * width + x];
          hist[v < 0 ? 0 : v > 255 ? 255 : v | 0]++;
          count++;
        }
      }
      const map = new Uint8Array(256);
      if (count === 0) {
        for (let k = 0; k < 256; k++) map[k] = k; // identity for empty tile
        maps[ty * tilesX + tx] = map;
        continue;
      }
      // Clip the histogram and redistribute the excess uniformly.
      const clip = Math.max(1, (clipMul * count) / 256);
      let excess = 0;
      for (let k = 0; k < 256; k++) {
        if (hist[k] > clip) { excess += hist[k] - clip; hist[k] = clip; }
      }
      const inc = excess / 256;
      // CDF → mapping.
      let cum = 0;
      const scale = 255 / count;
      for (let k = 0; k < 256; k++) {
        cum += hist[k] + inc;
        const mv = Math.round(cum * scale);
        map[k] = mv < 0 ? 0 : mv > 255 ? 255 : mv;
      }
      maps[ty * tilesX + tx] = map;
    }
  }

  const clampTile = (t: number, n: number) => (t < 0 ? 0 : t > n - 1 ? n - 1 : t);

  // Remap each pixel via bilinear interpolation of the 4 nearest tile maps.
  for (let y = 0; y < height; y++) {
    const fy = y / tileH - 0.5;
    const ty0f = Math.floor(fy);
    const wy = fy - ty0f;
    const ty0 = clampTile(ty0f, tilesY);
    const ty1 = clampTile(ty0f + 1, tilesY);
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      const Y = lum[p];
      const yi = Y < 0 ? 0 : Y > 255 ? 255 : Y | 0;
      const fx = x / tileW - 0.5;
      const tx0f = Math.floor(fx);
      const wx = fx - tx0f;
      const tx0 = clampTile(tx0f, tilesX);
      const tx1 = clampTile(tx0f + 1, tilesX);

      const m00 = maps[ty0 * tilesX + tx0][yi];
      const m01 = maps[ty0 * tilesX + tx1][yi];
      const m10 = maps[ty1 * tilesX + tx0][yi];
      const m11 = maps[ty1 * tilesX + tx1][yi];
      const top = m00 * (1 - wx) + m01 * wx;
      const bot = m10 * (1 - wx) + m11 * wx;
      const Ynew = top * (1 - wy) + bot * wy;

      const i = p * 4;
      if (Y > 1e-4) {
        const f = Ynew / Y;
        data[i] = data[i] * f;
        data[i + 1] = data[i + 1] * f;
        data[i + 2] = data[i + 2] * f;
      } else {
        data[i] = data[i + 1] = data[i + 2] = Ynew;
      }
    }
  }
}
