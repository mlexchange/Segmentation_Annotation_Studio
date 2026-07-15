/**
 * Bake the same display adjustments the canvas shows into an offscreen canvas
 * so tools (SAM encode, classic wand, Fill) operate on what the user sees.
 *
 * Pipeline: optional CLAHE → optional 3×3 sharpen → Brighten → Contrast → Levels.
 */
import { claheRgba } from '../clahe';
import { sharpenRgba } from '../sharpen';

export interface DisplayPreprocess {
  clahe?: boolean;
  sharpen?: boolean;
}

export function renderAdjusted(
  image: CanvasImageSource,
  width: number,
  height: number,
  brightness: number,
  contrast: number,
  levelsLo = 0,
  levelsHi = 255,
  preprocess: boolean | DisplayPreprocess = false,
): HTMLCanvasElement {
  const opts: DisplayPreprocess =
    typeof preprocess === 'boolean' ? { clahe: preprocess } : preprocess;
  const doClahe = !!opts.clahe;
  const doSharpen = !!opts.sharpen;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(image, 0, 0, width, height);

  const bcNoop = brightness === 0 && contrast === 0;
  const levelsNoop = levelsLo <= 0 && levelsHi >= 255;
  if (bcNoop && levelsNoop && !doClahe && !doSharpen) return canvas;

  const imageData = ctx.getImageData(0, 0, width, height);
  if (doClahe) claheRgba(imageData.data, width, height);
  if (doSharpen) sharpenRgba(imageData.data, width, height);
  if (bcNoop && levelsNoop) {
    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  const d = imageData.data;
  const b = brightness * 255;
  const adjust = Math.pow((contrast + 100) / 100, 2);
  const range = Math.max(1, levelsHi - levelsLo);
  for (let i = 0; i < d.length; i += 4) {
    for (let k = 0; k < 3; k++) {
      let v = d[i + k];
      if (!bcNoop) {
        v = v + b;
        v = ((v / 255 - 0.5) * adjust + 0.5) * 255;
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

/**
 * Offscreen canvas with CLAHE and/or sharpen only (no brightness/contrast/levels).
 * Used as the Konva image source when nonlinear preprocessors are on.
 */
export function renderPreprocessOnly(
  image: CanvasImageSource,
  width: number,
  height: number,
  preprocess: DisplayPreprocess,
): HTMLCanvasElement {
  return renderAdjusted(image, width, height, 0, 0, 0, 255, preprocess);
}

/** @deprecated Use renderPreprocessOnly({ clahe: true }). */
export function renderClaheOnly(
  image: CanvasImageSource,
  width: number,
  height: number,
): HTMLCanvasElement {
  return renderPreprocessOnly(image, width, height, { clahe: true });
}
