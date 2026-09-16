import { describe, it, expect } from 'vitest';
import { applyGaussianBlurRgba } from './blur';

function grayBuffer(w: number, h: number, fn: (x: number, y: number) => number): Uint8ClampedArray {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = fn(x, y);
      const i = (y * w + x) * 4;
      d[i] = d[i + 1] = d[i + 2] = v;
      d[i + 3] = 255;
    }
  }
  return d;
}

describe('applyGaussianBlurRgba', () => {
  it('leaves a flat image unchanged (edges clamp, so no darkening at the border)', () => {
    const d = grayBuffer(16, 16, () => 120);
    applyGaussianBlurRgba(d, 16, 16, 2);
    for (let i = 0; i < d.length; i += 4) expect(d[i]).toBeCloseTo(120, -0.5);
  });

  it('spreads an impulse into its neighbourhood', () => {
    const d = grayBuffer(21, 21, (x, y) => (x === 10 && y === 10 ? 255 : 0));
    applyGaussianBlurRgba(d, 21, 21, 2);
    const at = (x: number, y: number) => d[(y * 21 + x) * 4];
    expect(at(10, 10)).toBeLessThan(255);      // peak flattened
    expect(at(10, 10)).toBeGreaterThan(0);
    expect(at(11, 10)).toBeGreaterThan(0);     // energy moved outward
    expect(at(10, 11)).toBeGreaterThan(0);
    // Monotonically decreasing away from the impulse.
    expect(at(10, 10)).toBeGreaterThan(at(12, 10));
    expect(at(12, 10)).toBeGreaterThan(at(15, 10));
  });

  it('reduces the contrast of a step edge', () => {
    const w = 32;
    const d = grayBuffer(w, 4, (x) => (x < w / 2 ? 0 : 255));
    applyGaussianBlurRgba(d, w, 4, 3);
    const at = (x: number) => d[(1 * w + x) * 4];
    // Straddling the edge, values are now intermediate rather than 0/255.
    expect(at(w / 2 - 1)).toBeGreaterThan(0);
    expect(at(w / 2)).toBeLessThan(255);
    // Far from the edge the plateaus survive.
    expect(at(0)).toBeLessThan(10);
    expect(at(w - 1)).toBeGreaterThan(245);
  });

  it('is a no-op for sigma <= 0', () => {
    const d = grayBuffer(8, 8, (x, y) => (x + y) * 8);
    const before = new Uint8ClampedArray(d);
    applyGaussianBlurRgba(d, 8, 8, 0);
    expect(Array.from(d)).toEqual(Array.from(before));
    applyGaussianBlurRgba(d, 8, 8, -1);
    expect(Array.from(d)).toEqual(Array.from(before));
  });

  it('preserves alpha', () => {
    const d = grayBuffer(12, 12, (x) => 20 * x);
    applyGaussianBlurRgba(d, 12, 12, 1.5);
    for (let i = 3; i < d.length; i += 4) expect(d[i]).toBe(255);
  });

  it('does not throw on empty input', () => {
    expect(() => applyGaussianBlurRgba(new Uint8ClampedArray(0), 0, 0, 2)).not.toThrow();
  });
});
