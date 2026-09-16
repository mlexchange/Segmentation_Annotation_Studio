/**
 * A per-pixel intensity gate speckles, and every speck becomes its own polygon —
 * which is what made a single Threshold Brush stroke commit hundreds of shapes
 * and dominate clip cost. These pin the opening + small-component pass that the
 * commit path applies before vectorizing.
 */
import { describe, it, expect } from 'vitest';
import { dilate, erode, removeSmallComponents } from './morphology';
import { maskToPolygonsWithHoles } from './magicwand';

const W = 256;
const H = 256;

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The commit path's regularization, in one place. */
function regularize(mask: Uint8Array, minRegion = 12): Uint8Array {
  return removeSmallComponents(dilate(erode(mask, W, H, 1), W, H, 1), W, H, minRegion);
}

const area = (m: Uint8Array) => m.reduce((n, v) => n + (v ? 1 : 0), 0);
const polys = (m: Uint8Array) =>
  maskToPolygonsWithHoles(m, W, H, { minRegion: 4, scale: 1 }).filter((p) => p.points.length >= 6);

/** A solid square feature plus scattered thresholding noise. */
function speckledFeature(noise: number) {
  const mask = new Uint8Array(W * H);
  const rnd = mulberry32(42);
  for (let y = 80; y < 176; y++) for (let x = 80; x < 176; x++) mask[y * W + x] = 1;
  for (let i = 0; i < mask.length; i++) if (rnd() > 1 - noise) mask[i] = 1;
  return mask;
}

describe('threshold stroke regularization', () => {
  it('collapses speckle into the real feature', () => {
    const raw = speckledFeature(0.25);
    const cleaned = regularize(raw);
    // 96x96 = 9216 true pixels; the raw gate is far larger because of speckle.
    expect(area(raw)).toBeGreaterThan(9216 * 1.5);
    expect(area(cleaned)).toBeGreaterThan(9216 * 0.95);
    expect(area(cleaned)).toBeLessThan(9216 * 1.05);
  });

  it('turns hundreds of speck polygons into one', () => {
    const raw = speckledFeature(0.25);
    expect(polys(raw).length).toBeGreaterThan(50);
    expect(polys(regularize(raw)).length).toBe(1);
  });

  it('leaves a clean region essentially unchanged', () => {
    const clean = new Uint8Array(W * H);
    for (let y = 60; y < 200; y++) for (let x = 60; x < 200; x++) clean[y * W + x] = 1;
    const out = regularize(clean);
    expect(area(out)).toBeGreaterThan(area(clean) * 0.97);
    expect(polys(out).length).toBe(1);
  });

  it('erases a hairline stroke entirely — which is why the commit keeps a fallback', () => {
    // A 1px line cannot survive an erosion. The commit path detects this and
    // falls back to the raw mask rather than discarding the user's stroke.
    const hairline = new Uint8Array(W * H);
    for (let x = 40; x < 200; x++) hairline[128 * W + x] = 1;
    expect(area(regularize(hairline))).toBe(0);
    expect(area(hairline)).toBeGreaterThan(0);
  });

  it('preserves a genuine hole rather than filling it', () => {
    const donut = new Uint8Array(W * H);
    for (let y = 60; y < 200; y++) for (let x = 60; x < 200; x++) donut[y * W + x] = 1;
    for (let y = 110; y < 150; y++) for (let x = 110; x < 150; x++) donut[y * W + x] = 0;
    const out = regularize(donut);
    expect(polys(out)[0].holes.length).toBe(1);
  });
});
