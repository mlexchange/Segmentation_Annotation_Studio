import { describe, it, expect } from 'vitest';
import { applyStretch } from './stretch';

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

function range(d: Uint8ClampedArray): [number, number] {
  let mn = 255, mx = 0;
  for (let i = 0; i < d.length; i += 4) { mn = Math.min(mn, d[i]); mx = Math.max(mx, d[i]); }
  return [mn, mx];
}

describe('applyStretch', () => {
  it('stretches a low-contrast range toward the full [0,255]', () => {
    // Values confined to [110,136] with a uniform spread across the tile.
    const d = grayBuffer(64, 64, (x, y) => 110 + ((x + y) % 27));
    const [mn0, mx0] = range(d);
    expect(mx0 - mn0).toBeLessThan(30);
    applyStretch(d, 64, 64, { lowPct: 2, highPct: 98 });
    const [mn1, mx1] = range(d);
    expect(mn1).toBeLessThan(20);   // low end pushed toward 0
    expect(mx1).toBeGreaterThan(235); // high end pushed toward 255
    expect(mx1 - mn1).toBeGreaterThan(mx0 - mn0);
  });

  it('leaves a flat image unchanged (nothing to stretch)', () => {
    const d = grayBuffer(16, 16, () => 128);
    applyStretch(d, 16, 16);
    for (let i = 0; i < d.length; i += 4) expect(d[i]).toBe(128);
  });

  it('keeps values in [0,255] and preserves alpha', () => {
    const d = grayBuffer(32, 32, (x) => 90 + x);
    applyStretch(d, 32, 32);
    for (let i = 0; i < d.length; i++) {
      expect(d[i]).toBeGreaterThanOrEqual(0);
      expect(d[i]).toBeLessThanOrEqual(255);
    }
    for (let i = 3; i < d.length; i += 4) expect(d[i]).toBe(255);
  });

  it('does not throw on empty input', () => {
    expect(() => applyStretch(new Uint8ClampedArray(0), 0, 0)).not.toThrow();
  });
});
