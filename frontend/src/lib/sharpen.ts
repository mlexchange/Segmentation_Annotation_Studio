/**
 * Fast classic sharpening — 3×3 Laplacian high-boost convolution.
 *
 * Kernel (amount = 1):
 *   [  0  -1   0 ]
 *   [ -1   5  -1 ]
 *   [  0  -1   0 ]
 *
 * Equivalent to ``out = src + amount * (4·center − N − S − E − W)``.
 * Much cheaper than unsharp-mask (no blur pass). Display-only preprocessor.
 */

export interface SharpenOptions {
  /** Edge boost strength; 1 ≈ classic high-boost kernel. Default 1. */
  amount?: number;
}

/**
 * In-place 3×3 Laplacian sharpen on RGBA ImageData (alpha unchanged).
 *
 * Args:
 *   rgba: Packed RGBA bytes (mutated in place).
 *   width / height: Image size in pixels.
 *   options: Sharpen strength.
 */
export function sharpenRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  options: SharpenOptions = {},
): void {
  const amount = options.amount ?? 1;
  if (amount === 0 || width < 2 || height < 2) return;

  // Work from a copy so neighbouring reads stay original.
  const src = new Uint8ClampedArray(rgba);
  for (let y = 0; y < height; y++) {
    const yN = y > 0 ? y - 1 : 0;
    const yS = y < height - 1 ? y + 1 : height - 1;
    for (let x = 0; x < width; x++) {
      const xW = x > 0 ? x - 1 : 0;
      const xE = x < width - 1 ? x + 1 : width - 1;
      const iC = (y * width + x) * 4;
      const iN = (yN * width + x) * 4;
      const iS = (yS * width + x) * 4;
      const iW = (y * width + xW) * 4;
      const iE = (y * width + xE) * 4;
      for (let k = 0; k < 3; k++) {
        const c = src[iC + k];
        const lap = 4 * c - src[iN + k] - src[iS + k] - src[iW + k] - src[iE + k];
        rgba[iC + k] = c + amount * lap; // Uint8ClampedArray clamps
      }
    }
  }
}
