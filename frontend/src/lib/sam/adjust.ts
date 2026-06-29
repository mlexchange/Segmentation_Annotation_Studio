/**
 * Bake the same brightness/contrast the canvas shows (Konva's Brighten +
 * Contrast filters) into an offscreen canvas, so SAM encodes the image the user
 * actually sees. Windowing a low-contrast tomography slice before encoding is
 * one of the biggest levers on mask quality.
 *
 * Mirrors Konva's filter math exactly (filters applied in order Brighten →
 * Contrast): Brighten adds `brightness*255`; Contrast scales around mid-grey by
 * `((contrast+100)/100)^2`. Returns a canvas at the image's native resolution.
 */
export function renderAdjusted(
  image: CanvasImageSource,
  width: number,
  height: number,
  brightness: number,
  contrast: number,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(image, 0, 0, width, height);
  if (brightness === 0 && contrast === 0) return canvas;

  const imageData = ctx.getImageData(0, 0, width, height);
  const d = imageData.data; // Uint8ClampedArray → assignments auto-clamp to [0,255]
  const b = brightness * 255;
  const adjust = Math.pow((contrast + 100) / 100, 2);
  for (let i = 0; i < d.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      const v = d[i + k] + b;                       // brighten
      d[i + k] = ((v / 255 - 0.5) * adjust + 0.5) * 255; // contrast
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}
