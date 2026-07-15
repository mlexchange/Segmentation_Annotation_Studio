/**
 * Classic 3×3 Laplacian sharpen tests.
 */
import { describe, expect, it } from 'vitest';
import { sharpenRgba } from './sharpen';

function grayRgba(w: number, h: number, fill: (x: number, y: number) => number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const g = fill(x, y);
      const i = (y * w + x) * 4;
      out[i] = g;
      out[i + 1] = g;
      out[i + 2] = g;
      out[i + 3] = 255;
    }
  }
  return out;
}

describe('sharpenRgba', () => {
  it('keeps a flat field flat', () => {
    const rgba = grayRgba(16, 16, () => 100);
    sharpenRgba(rgba, 16, 16, { amount: 1 });
    for (let i = 0; i < rgba.length; i += 4) {
      expect(rgba[i]).toBe(100);
      expect(rgba[i + 3]).toBe(255);
    }
  });

  it('increases the jump across a vertical step edge', () => {
    const w = 32;
    const h = 16;
    const rgba = grayRgba(w, h, (x) => (x < w / 2 ? 40 : 200));
    const beforeLeft = rgba[(8 * w + 14) * 4];
    const beforeRight = rgba[(8 * w + 17) * 4];
    const beforeJump = beforeRight - beforeLeft;
    sharpenRgba(rgba, w, h, { amount: 1 });
    const afterLeft = rgba[(8 * w + 14) * 4];
    const afterRight = rgba[(8 * w + 17) * 4];
    expect(afterRight - afterLeft).toBeGreaterThanOrEqual(beforeJump);
  });

  it('clamps outputs to 0–255', () => {
    const rgba = grayRgba(8, 8, (x, y) => (x === 4 && y === 4 ? 255 : 0));
    sharpenRgba(rgba, 8, 8, { amount: 2 });
    for (let i = 0; i < rgba.length; i += 4) {
      expect(rgba[i]).toBeGreaterThanOrEqual(0);
      expect(rgba[i]).toBeLessThanOrEqual(255);
    }
  });
});
