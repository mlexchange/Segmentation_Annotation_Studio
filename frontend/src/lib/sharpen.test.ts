import { describe, it, expect } from 'vitest';
import { applySharpen } from './sharpen';

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

describe('applySharpen', () => {
  it('leaves a flat image unchanged (5c - 4c = c)', () => {
    const d = grayBuffer(8, 8, () => 120);
    applySharpen(d, 8, 8);
    for (let i = 0; i < d.length; i += 4) expect(d[i]).toBe(120);
  });

  it('amplifies a bright pixel and darkens its neighbours', () => {
    // Dark field with one bright center pixel.
    const d = grayBuffer(5, 5, (x, y) => (x === 2 && y === 2 ? 200 : 40));
    applySharpen(d, 5, 5);
    const at = (x: number, y: number) => d[(y * 5 + x) * 4];
    expect(at(2, 2)).toBe(255);          // 5*200 - 4*40 = 840 → clamped to 255
    expect(at(2, 1)).toBeLessThan(40);   // 5*40 - 200 - 3*40 = -80 → clamped to 0
    expect(at(1, 2)).toBeLessThan(40);
  });

  it('preserves alpha', () => {
    const d = grayBuffer(6, 6, (x) => 30 * x);
    applySharpen(d, 6, 6);
    for (let i = 3; i < d.length; i += 4) expect(d[i]).toBe(255);
  });

  it('does not throw on empty input', () => {
    expect(() => applySharpen(new Uint8ClampedArray(0), 0, 0)).not.toThrow();
  });
});
