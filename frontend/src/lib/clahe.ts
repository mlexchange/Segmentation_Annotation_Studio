/**
 * CLAHE — Contrast Limited Adaptive Histogram Equalization (grayscale).
 *
 * Tile the image, equalize each tile's histogram with a clip limit, then
 * bilinear-interpolate the per-tile CDFs so the result has no tile seams.
 * Used as a display preprocessor for low-contrast slices (does not change
 * exported pixel values — same posture as brightness/contrast).
 */

export interface ClaheOptions {
  /** Histogram clip limit relative to tile average bin height (OpenCV-style). Default 2. */
  clipLimit?: number;
  /** Number of tiles along width. Default 8. */
  tilesX?: number;
  /** Number of tiles along height. Default 8. */
  tilesY?: number;
}

const NUM_BINS = 256;

/**
 * Apply CLAHE to a packed grayscale buffer (row-major, one byte per pixel).
 *
 * Args:
 *   src: Length ``width * height`` luminance bytes.
 *   width / height: Image size in pixels.
 *   options: Clip limit and tile grid.
 *
 * Returns:
 *   New ``Uint8Array`` of the same length with CLAHE applied.
 */
export function applyClaheGray(
  src: Uint8Array,
  width: number,
  height: number,
  options: ClaheOptions = {},
): Uint8Array {
  const clipLimit = Math.max(0, options.clipLimit ?? 2);
  const tilesX = Math.max(1, Math.min(options.tilesX ?? 8, width));
  const tilesY = Math.max(1, Math.min(options.tilesY ?? 8, height));
  const out = new Uint8Array(src.length);

  // Tile geometry (last tiles absorb remainders).
  const tileW = Math.floor(width / tilesX);
  const tileH = Math.floor(height / tilesY);
  if (tileW < 1 || tileH < 1) {
    out.set(src);
    return out;
  }

  // Per-tile LUTs: map input gray → equalized gray via clipped CDF.
  const luts = new Array<Uint8Array>(tilesX * tilesY);
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * tileW;
      const y0 = ty * tileH;
      const x1 = tx === tilesX - 1 ? width : x0 + tileW;
      const y1 = ty === tilesY - 1 ? height : y0 + tileH;
      luts[ty * tilesX + tx] = buildTileLut(src, width, x0, y0, x1, y1, clipLimit);
    }
  }

  // Bilinear blend of the 4 surrounding tile LUTs at each pixel.
  for (let y = 0; y < height; y++) {
    const tyF = (y + 0.5) / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(tyF)));
    const ty1 = Math.max(0, Math.min(tilesY - 1, ty0 + 1));
    const fy = ty0 === ty1 ? 0 : tyF - ty0;

    for (let x = 0; x < width; x++) {
      const txF = (x + 0.5) / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(txF)));
      const tx1 = Math.max(0, Math.min(tilesX - 1, tx0 + 1));
      const fx = tx0 === tx1 ? 0 : txF - tx0;

      const v = src[y * width + x];
      const a = luts[ty0 * tilesX + tx0][v];
      const b = luts[ty0 * tilesX + tx1][v];
      const c = luts[ty1 * tilesX + tx0][v];
      const d = luts[ty1 * tilesX + tx1][v];
      const top = a + (b - a) * fx;
      const bot = c + (d - c) * fx;
      out[y * width + x] = Math.round(top + (bot - top) * fy);
    }
  }
  return out;
}

/**
 * In-place CLAHE on RGBA ImageData bytes via luminance, then rescale RGB.
 *
 * Grayscale slices keep R=G=B. Color (e.g. viridis) keeps hue by scaling
 * channels with ``Y'/Y``.
 */
export function claheRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  options: ClaheOptions = {},
): void {
  const n = width * height;
  const gray = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    gray[i] = Math.round(0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]);
  }
  const eq = applyClaheGray(gray, width, height, options);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const y0 = gray[i];
    const y1 = eq[i];
    if (y0 === 0) {
      rgba[p] = y1;
      rgba[p + 1] = y1;
      rgba[p + 2] = y1;
    } else {
      const s = y1 / y0;
      rgba[p] = Math.round(rgba[p] * s);
      rgba[p + 1] = Math.round(rgba[p + 1] * s);
      rgba[p + 2] = Math.round(rgba[p + 2] * s);
    }
    // alpha unchanged
  }
}

/** Build a 256-entry LUT for one tile using a clip-limited histogram CDF. */
function buildTileLut(
  src: Uint8Array,
  width: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  clipLimit: number,
): Uint8Array {
  const hist = new Float64Array(NUM_BINS);
  let area = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * width;
    for (let x = x0; x < x1; x++) {
      hist[src[row + x]] += 1;
      area += 1;
    }
  }
  if (area === 0) return identityLut();

  // OpenCV-style clip: max bin height = clipLimit * (area / bins).
  const limit = Math.max(1, (clipLimit * area) / NUM_BINS);
  let clipped = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    if (hist[i] > limit) {
      clipped += hist[i] - limit;
      hist[i] = limit;
    }
  }
  const redistrib = clipped / NUM_BINS;
  for (let i = 0; i < NUM_BINS; i++) hist[i] += redistrib;

  // CDF → LUT mapped to [0, 255].
  const lut = new Uint8Array(NUM_BINS);
  let cdf = 0;
  let cdfMin = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    cdf += hist[i];
    if (cdfMin === 0 && cdf > 0) cdfMin = cdf;
    const denom = area - cdfMin;
    lut[i] = denom <= 0 ? 0 : Math.round(((cdf - cdfMin) / denom) * 255);
  }
  return lut;
}

function identityLut(): Uint8Array {
  const lut = new Uint8Array(NUM_BINS);
  for (let i = 0; i < NUM_BINS; i++) lut[i] = i;
  return lut;
}
