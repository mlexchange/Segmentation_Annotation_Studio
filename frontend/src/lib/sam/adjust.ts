import { applyStretch } from '@/lib/stretch';
import { applySharpen } from '@/lib/sharpen';

/** Nonlinear display preprocessors baked before the linear brightness/contrast/levels. */
export interface PreprocessOpts {
  /** Global percentile histogram stretch (auto-contrast). */
  stretch?: boolean;
  sharpen?: boolean;
}

/**
 * Bake the display adjustments the canvas shows into an offscreen canvas, so the
 * tools (SAM encode, classic wand, Fill, magnetic cost map) operate on the image
 * the user actually sees. Windowing a low-contrast tomography slice before
 * encoding is one of the biggest levers on mask quality.
 *
 * Order: nonlinear preprocessors first (Stretch → Sharpen), then the linear chain
 * mirroring the canvas filter exactly (Brighten → Contrast → Levels): Brighten
 * adds `brightness*255`; Contrast scales around mid-grey by `((contrast+100)/100)^2`;
 * Levels remaps `[lo,hi] → [0,255]`. Returns a canvas at native resolution.
 */
export function renderAdjusted(
  image: CanvasImageSource,
  width: number,
  height: number,
  brightness: number,
  contrast: number,
  levelsLo = 0,
  levelsHi = 255,
  preprocess?: PreprocessOpts,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(image, 0, 0, width, height);

  const doPre = !!(preprocess && (preprocess.stretch || preprocess.sharpen));
  const bcNoop = brightness === 0 && contrast === 0;
  const levelsNoop = levelsLo <= 0 && levelsHi >= 255;
  if (!doPre && bcNoop && levelsNoop) return canvas;

  const imageData = ctx.getImageData(0, 0, width, height);
  const d = imageData.data; // Uint8ClampedArray → assignments auto-clamp to [0,255]

  // Nonlinear preprocessors first (baked once), in a fixed order.
  if (doPre) {
    if (preprocess!.stretch) applyStretch(d, width, height);
    if (preprocess!.sharpen) applySharpen(d, width, height);
  }

  if (!(bcNoop && levelsNoop)) {
    const b = brightness * 255;
    const adjust = Math.pow((contrast + 100) / 100, 2);
    // Levels lookup (matches the canvas SVG filter): ≤lo→0, ≥hi→255, linear between.
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
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * Bake ONLY the nonlinear preprocessors (Stretch / Sharpen) into a canvas — used as
 * the Konva display base, since brightness/contrast/levels/gamma/colormap stay on
 * the GPU SVG filter applied over it. Returns the source untouched when no
 * preprocessor is active.
 */
export function renderPreprocessOnly(
  image: CanvasImageSource,
  width: number,
  height: number,
  preprocess: PreprocessOpts,
): HTMLCanvasElement {
  return renderAdjusted(image, width, height, 0, 0, 0, 255, preprocess);
}
