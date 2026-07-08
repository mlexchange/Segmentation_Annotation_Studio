/**
 * Konva image filter implementing a min/max "levels" window on the DISPLAYED
 * 8-bit image (client-side, no refetch): pixels ≤ lo → 0, ≥ hi → 255, linearly
 * remapped in between. Stacks after Brighten/Contrast in the filter chain.
 *
 * Reads `levelsLo` / `levelsHi` (0–255) set on the image node via `setAttr`, so
 * changing them + `cache()`/`batchDraw()` re-applies without a re-fetch. Konva
 * invokes filters with the node as `this`.
 */
export function LevelsFilter(this: any, imageData: ImageData): void {
  const lo = (this?.getAttr?.('levelsLo') as number) ?? 0;
  const hi = (this?.getAttr?.('levelsHi') as number) ?? 255;
  if (lo <= 0 && hi >= 255) return; // full range → no-op
  const range = Math.max(1, hi - lo);
  const lut = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = i <= lo ? 0 : i >= hi ? 255 : Math.round(((i - lo) * 255) / range);
  }
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]];
    d[i + 1] = lut[d[i + 1]];
    d[i + 2] = lut[d[i + 2]];
  }
}
