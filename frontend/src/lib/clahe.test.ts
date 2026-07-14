import { describe, it, expect } from 'vitest';
import { applyCLAHE } from './clahe';

/** Build a W×H RGBA buffer from a per-pixel gray value function. */
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

function stdOfR(d: Uint8ClampedArray): number {
  const vals: number[] = [];
  for (let i = 0; i < d.length; i += 4) vals.push(d[i]);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  return Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
}

describe('applyCLAHE', () => {
  it('keeps values in [0,255] and preserves alpha', () => {
    const d = grayBuffer(16, 16, (x) => 100 + x); // low-contrast ramp
    applyCLAHE(d, 16, 16, { tilesX: 2, tilesY: 2 });
    for (let i = 0; i < d.length; i++) {
      expect(d[i]).toBeGreaterThanOrEqual(0);
      expect(d[i]).toBeLessThanOrEqual(255);
    }
    for (let i = 3; i < d.length; i += 4) expect(d[i]).toBe(255); // alpha untouched
  });

  it('increases contrast of a low-contrast image', () => {
    // Values confined to a narrow band [100,131] → CLAHE should spread them out.
    const d = grayBuffer(32, 32, (x, y) => 100 + ((x + y) % 32));
    const before = stdOfR(d);
    applyCLAHE(d, 32, 32, { tilesX: 4, tilesY: 4, clipLimit: 4 });
    const after = stdOfR(d);
    expect(after).toBeGreaterThan(before);
  });

  it('leaves a flat image uniform', () => {
    const d = grayBuffer(16, 16, () => 128);
    applyCLAHE(d, 16, 16);
    const first = d[0];
    for (let i = 0; i < d.length; i += 4) expect(d[i]).toBe(first);
  });

  it('does not throw on empty input', () => {
    expect(() => applyCLAHE(new Uint8ClampedArray(0), 0, 0)).not.toThrow();
  });
});
