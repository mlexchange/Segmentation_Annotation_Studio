/**
 * displayTransform — the display chain the canvas applies on the GPU
 * (brightness/contrast/levels as one affine, then gamma), expressed as plain math
 * so tools can reason about it without re-rendering pixels.
 *
 * The canvas bakes nonlinear preprocessors (blur/CLAHE/sharpen) into an offscreen
 * base and then applies THIS transform as an SVG filter. Anything that needs to
 * know "which pixels look like X on screen" therefore has two options: re-render
 * the whole image with the transform baked in (expensive, and it invalidates every
 * cached field whenever a slider moves), or keep a field in un-adjusted base space
 * and map the QUESTION through the inverse transform instead (O(1) per change).
 *
 * The Threshold Brush takes the second route: its intensity band is authored in
 * displayed space, `displayToBase` maps the band's endpoints back to base space
 * once per change, and the comparison happens against the cached base field. The
 * selected pixel set is identical either way, because the transform is monotonic.
 */

export interface DisplayAffine {
  /** Per-channel slope on NORMALIZED [0,1] values (SVG feFuncX type="linear"). */
  slope: number;
  /** Per-channel intercept on normalized values. */
  intercept: number;
}

/**
 * Fold brightness, contrast, and the levels window into one normalized affine —
 * the exact transform `renderAdjusted` bakes and the canvas filter applies:
 * Brighten adds `brightness*255`; Contrast scales around mid-grey by
 * `((contrast+100)/100)^2`; Levels remaps `[lo,hi] → [0,255]`.
 */
export function displayAffineFor(
  brightness: number,
  contrast: number,
  levelsLo: number,
  levelsHi: number,
): DisplayAffine {
  const b255 = brightness * 255;
  const adjust = Math.pow((contrast + 100) / 100, 2);
  const range = Math.max(1, levelsHi - levelsLo);
  const slope = adjust * (255 / range);
  const intercept =
    ((255 / range) * (adjust * b255 + 127.5 * (1 - adjust)) - (255 * levelsLo) / range) / 255;
  return { slope, intercept };
}

/** Base (un-adjusted) 0–255 value → what the screen shows, 0–255. */
export function baseToDisplay(base: number, affine: DisplayAffine, gamma = 1): number {
  const a = affine.slope * (base / 255) + affine.intercept;
  const clamped = a < 0 ? 0 : a > 1 ? 1 : a;
  const g = gamma === 1 ? clamped : Math.pow(clamped, gamma);
  return g * 255;
}

/**
 * Displayed 0–255 value → the base value that produces it. The inverse of
 * `baseToDisplay`, ignoring its output clamp: a displayed 0 or 255 corresponds to
 * an unbounded range of base values, which callers handle by treating a band edge
 * at the extreme as open (see `displayBandToBase`).
 *
 * Returns null when the transform collapses (contrast −100 flattens every input to
 * a single grey, so no inverse exists).
 */
export function displayToBase(display: number, affine: DisplayAffine, gamma = 1): number | null {
  if (Math.abs(affine.slope) < 1e-9) return null;
  const g = Math.max(0, Math.min(1, display / 255));
  const a = gamma === 1 ? g : Math.pow(g, 1 / gamma);
  return ((a - affine.intercept) / affine.slope) * 255;
}

/**
 * Map an intensity band authored in DISPLAYED space to the equivalent band in base
 * space, so a threshold test against a cached base field selects exactly the pixels
 * that look in-band on screen.
 *
 * Edges at 0 / 255 become open (∓Infinity): everything the display clamps to black
 * or white is genuinely in-band. When the transform collapses, the band either
 * covers everything or nothing depending on where the single output grey lands.
 */
export function displayBandToBase(
  lo: number,
  hi: number,
  affine: DisplayAffine,
  gamma = 1,
): { lo: number; hi: number } {
  if (Math.abs(affine.slope) < 1e-9) {
    // Every base value maps to the same displayed grey; the band is all-or-nothing.
    const flat = baseToDisplay(128, affine, gamma);
    return flat >= lo && flat <= hi
      ? { lo: -Infinity, hi: Infinity }
      : { lo: Infinity, hi: -Infinity };
  }
  const bLo = lo <= 0 ? -Infinity : displayToBase(lo, affine, gamma) ?? -Infinity;
  const bHi = hi >= 255 ? Infinity : displayToBase(hi, affine, gamma) ?? Infinity;
  // A negative slope would invert the ordering; keep the band well-formed.
  return affine.slope > 0 ? { lo: bLo, hi: bHi } : { lo: bHi, hi: bLo };
}

/**
 * Re-bin a base-space luminance histogram into displayed space, so a picker drawn
 * over it lines up with what the user sees (and with where the threshold band
 * actually cuts). Pure 256-bin remap — no pixels touched.
 */
export function remapHistogramToDisplay(
  bins: number[],
  affine: DisplayAffine,
  gamma = 1,
): number[] {
  const out = new Array(256).fill(0);
  for (let b = 0; b < bins.length; b++) {
    const count = bins[b];
    if (!count) continue;
    let d = Math.round(baseToDisplay(b, affine, gamma));
    d = d < 0 ? 0 : d > 255 ? 255 : d;
    out[d] += count;
  }
  return out;
}
