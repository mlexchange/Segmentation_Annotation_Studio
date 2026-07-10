/**
 * Bake the same display adjustments the canvas shows (Konva's Brighten +
 * Contrast filters, then the min/max Levels remap) into an offscreen canvas, so
 * the tools (SAM encode, classic wand, Fill) operate on the image the user
 * actually sees. Windowing a low-contrast tomography slice before encoding is
 * one of the biggest levers on mask quality.
 *
 * Mirrors the canvas filter chain exactly (Brighten → Contrast → Levels):
 * Brighten adds `brightness*255`; Contrast scales around mid-grey by
 * `((contrast+100)/100)^2`; Levels remaps `[lo,hi] → [0,255]` (0–255 units).
 * Returns a canvas at the image's native resolution.
 */
export function renderAdjusted(
  image: CanvasImageSource,
  width: number,
  height: number,
  brightness: number,
  contrast: number,
  levelsLo = 0,
  levelsHi = 255,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(image, 0, 0, width, height);

  const bcNoop = brightness === 0 && contrast === 0;
  const levelsNoop = levelsLo <= 0 && levelsHi >= 255;
  if (bcNoop && levelsNoop) return canvas;

  const imageData = ctx.getImageData(0, 0, width, height);
  const d = imageData.data; // Uint8ClampedArray → assignments auto-clamp to [0,255]
  const b = brightness * 255;
  const adjust = Math.pow((contrast + 100) / 100, 2);
  // Levels lookup table (matches lib/levelsFilter.ts): ≤lo→0, ≥hi→255, linear between.
  const range = Math.max(1, levelsHi - levelsLo);
  for (let i = 0; i < d.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      let v = d[i + k];
      if (!bcNoop) {
        v = v + b;                                  // brighten
        v = ((v / 255 - 0.5) * adjust + 0.5) * 255; // contrast
      }
      if (!levelsNoop) {
        v = v <= levelsLo ? 0 : v >= levelsHi ? 255 : Math.round(((v - levelsLo) * 255) / range);
      }
      d[i + k] = v;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}
