/**
 * Histogram stretch (auto-contrast) — display-only.
 *
 * Linearly remaps the luminance range between the low/high intensity percentiles
 * to [0,255] (clipping outliers), so a low-contrast scan uses the full display
 * range. This is a GLOBAL operation — unlike local/adaptive CLAHE it visibly
 * "stretches the histogram". RGB is rescaled by the luminance mapping so hue is
 * preserved (for grayscale scans R=G=B, so it's an exact per-pixel remap).
 *
 * Mutates the RGBA Uint8ClampedArray in place (alpha untouched). Pure/DOM-free so
 * it's unit-testable; the display pipeline calls it on an ImageData buffer.
 */
export interface StretchOptions {
  /** Lower percentile mapped to 0 (default 2). */
  lowPct?: number;
  /** Upper percentile mapped to 255 (default 98). */
  highPct?: number;
}

/** Apply a percentile histogram stretch to an RGBA pixel buffer in place. */
export function applyStretch(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  opts: StretchOptions = {},
): void {
  const n = width * height;
  if (n === 0) return;
  const lowPct = opts.lowPct ?? 2;
  const highPct = opts.highPct ?? 98;

  // Luminance + its 256-bin histogram.
  const lum = new Float32Array(n);
  const hist = new Float32Array(256);
  for (let i = 0, p = 0; p < n; i += 4, p++) {
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    lum[p] = y;
    hist[y < 0 ? 0 : y > 255 ? 255 : y | 0]++;
  }

  // Percentile bounds.
  const loCount = (n * lowPct) / 100;
  const hiCount = (n * highPct) / 100;
  let cum = 0;
  let lo = 0;
  let hi = 255;
  let foundLo = false;
  for (let k = 0; k < 256; k++) {
    cum += hist[k];
    if (!foundLo && cum >= loCount) { lo = k; foundLo = true; }
    if (cum >= hiCount) { hi = k; break; }
  }
  // Degenerate/flat range → nothing to stretch.
  if (hi <= lo) return;

  const range = hi - lo;
  for (let p = 0; p < n; p++) {
    const Y = lum[p];
    const Ynew = Y <= lo ? 0 : Y >= hi ? 255 : ((Y - lo) * 255) / range;
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
