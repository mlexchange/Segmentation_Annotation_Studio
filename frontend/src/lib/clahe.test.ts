/**
 * CLAHE unit tests — contrast-limited adaptive histogram equalization.
 */
import { describe, expect, it } from 'vitest';
import { applyClaheGray, claheRgba } from './clahe';

/** Build a gray gradient (left dark → right bright) as Uint8Array luminance. */
function gradient(w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = Math.round((x / Math.max(1, w - 1)) * 255);
    }
  }
  return out;
}

/** Soft-edged dark blob on mid-gray — local contrast should rise under CLAHE. */
function blobOnGray(w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  out.fill(128);
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) * 0.2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = Math.hypot(x - cx, y - cy);
      if (d < r) out[y * w + x] = Math.round(40 + (d / r) * 40);
    }
  }
  return out;
}

describe('applyClaheGray', () => {
  it('keeps values in 0–255', () => {
    const src = blobOnGray(64, 64);
    const out = applyClaheGray(src, 64, 64, { clipLimit: 2, tilesX: 8, tilesY: 8 });
    expect(out.length).toBe(src.length);
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(0);
      expect(out[i]).toBeLessThanOrEqual(255);
    }
  });

  it('is a no-op on a flat field', () => {
    const src = new Uint8Array(32 * 32).fill(90);
    const out = applyClaheGray(src, 32, 32, { clipLimit: 2, tilesX: 4, tilesY: 4 });
    // Flat images stay flat (all equal, near the original level).
    const first = out[0];
    expect(out.every((v) => v === first)).toBe(true);
  });

  it('increases local contrast on a soft blob', () => {
    const src = blobOnGray(128, 128);
    const out = applyClaheGray(src, 128, 128, { clipLimit: 3, tilesX: 8, tilesY: 8 });
    const srcRange = Math.max(...src) - Math.min(...src);
    const outRange = Math.max(...out) - Math.min(...out);
    expect(outRange).toBeGreaterThanOrEqual(srcRange);
  });

  it('leaves a full-range gradient roughly ordered', () => {
    const src = gradient(64, 16);
    const out = applyClaheGray(src, 64, 16, { clipLimit: 2, tilesX: 8, tilesY: 2 });
    // Left third still darker than right third after CLAHE.
    const left = out.slice(0, 8).reduce((a, b) => a + b, 0);
    const right = out.slice(56, 64).reduce((a, b) => a + b, 0);
    expect(left).toBeLessThan(right);
  });
});

describe('claheRgba', () => {
  it('writes equal RGB for a grayscale source', () => {
    const w = 32;
    const h = 32;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const g = (i * 7) % 256;
      rgba[i * 4] = g;
      rgba[i * 4 + 1] = g;
      rgba[i * 4 + 2] = g;
      rgba[i * 4 + 3] = 255;
    }
    claheRgba(rgba, w, h, { clipLimit: 2, tilesX: 4, tilesY: 4 });
    for (let i = 0; i < w * h; i++) {
      expect(rgba[i * 4]).toBe(rgba[i * 4 + 1]);
      expect(rgba[i * 4 + 1]).toBe(rgba[i * 4 + 2]);
      expect(rgba[i * 4 + 3]).toBe(255);
    }
  });
});
