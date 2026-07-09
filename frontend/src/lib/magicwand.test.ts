import { describe, it, expect } from 'vitest';
import { magicSelect, maskToPolygons, maskToPolygonsWithHoles, gradientField, type GrayField } from './magicwand';

/** 40x40 grid (scale 1): background 0 with two value-200 blocks. */
function twoBlocks(): GrayField {
  const gw = 40, gh = 40;
  const gray = new Float32Array(gw * gh);
  const fill = (x0: number, y0: number, s: number) => {
    for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) gray[y * gw + x] = 200;
  };
  fill(4, 4, 10);    // block A (top-left)
  fill(26, 26, 10);  // block B (bottom-right)
  return { gw, gh, scale: 1, gray };
}

describe('magicwand', () => {
  it('global mode selects all similar regions at once', () => {
    const field = twoBlocks();
    const polys = magicSelect(field, 9, 9, { toleranceFrac: 0.2, mode: 'global', smooth: 0, minRegion: 8 });
    expect(polys.length).toBe(2);
  });

  it('contiguous mode selects only the clicked region', () => {
    const field = twoBlocks();
    const polys = magicSelect(field, 9, 9, { toleranceFrac: 0.2, mode: 'contiguous', smooth: 0, minRegion: 8 });
    expect(polys.length).toBe(1);
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    const ys = polys[0].filter((_, i) => i % 2 === 1);
    expect(Math.max(...xs)).toBeLessThan(20); // stayed in the top-left block
    expect(Math.max(...ys)).toBeLessThan(20);
  });

  it('minRegion drops specks', () => {
    const gw = 40, gh = 40;
    const gray = new Float32Array(gw * gh);
    for (let y = 4; y < 16; y++) for (let x = 4; x < 16; x++) gray[y * gw + x] = 200; // real region
    gray[35 * gw + 35] = 200; // 1px speck
    const polys = magicSelect({ gw, gh, scale: 1, gray }, 8, 8, {
      toleranceFrac: 0.3, mode: 'global', smooth: 0, minRegion: 12,
    });
    expect(polys.length).toBe(1);
  });

  it('edgeStop keeps the flood from crossing a gradient wall', () => {
    const gw = 40, gh = 40;
    const gray = new Float32Array(gw * gh).fill(50); // uniform intensity
    const grad = new Float32Array(gw * gh);
    for (let y = 0; y < gh; y++) grad[y * gw + 20] = 1; // strong vertical edge at x=20
    const field: GrayField = { gw, gh, scale: 1, gray, grad };

    // Without an edge barrier the whole uniform field floods (crosses x=20).
    const open = magicSelect(field, 5, 5, { toleranceFrac: 0.5, mode: 'contiguous', smooth: 0, edgeStop: 0, minRegion: 8 });
    expect(open.length).toBe(1);
    expect(Math.max(...open[0].filter((_, i) => i % 2 === 0))).toBeGreaterThan(25);

    // With the barrier the flood stops at the wall (stays left of x=20).
    const walled = magicSelect(field, 5, 5, { toleranceFrac: 0.5, mode: 'contiguous', smooth: 0, edgeStop: 0.5, minRegion: 8 });
    expect(walled.length).toBe(1);
    expect(Math.max(...walled[0].filter((_, i) => i % 2 === 0))).toBeLessThan(21);
  });

  it('fills a uniform region with edgeStop on (noise must not wall the flood)', () => {
    // Left half uniform (100, with faint noise), right half 200, sharp edge at x=30.
    const gw = 60, gh = 40;
    const gray = new Float32Array(gw * gh);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) gray[y * gw + x] = x < 30 ? 100 : 200;
    }
    gray[20 * gw + 10] = 104; // a faint interior "noise" pixel
    gray[15 * gw + 18] = 97;
    const field: GrayField = { gw, gh, scale: 1, gray, grad: gradientField(gray, gw, gh) };

    // Seed in the uniform left region with the edge barrier active.
    const polys = magicSelect(field, 10, 20, {
      toleranceFrac: 0.2, mode: 'contiguous', smooth: 0, edgeStop: 0.3, minRegion: 8,
    });
    expect(polys.length).toBe(1);
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    // Should fill most of the left region (not get trapped at the seed) and not cross x=30.
    expect(Math.max(...xs)).toBeGreaterThan(20);
    expect(Math.max(...xs)).toBeLessThan(31);
  });

  it('scales polygon coordinates back to image space', () => {
    const field = twoBlocks();
    field.scale = 4; // pretend the field was downsampled 4x
    const polys = magicSelect(field, 36, 36, { toleranceFrac: 0.2, mode: 'contiguous', smooth: 0, minRegion: 8 });
    expect(polys.length).toBe(1);
    // block A spans grid x∈[4,14) → image x∈[~16,~58); coords should exceed grid range.
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    expect(Math.max(...xs)).toBeGreaterThan(20);
  });
});

/** Build a w×h binary mask with `fill` setting rectangular blocks to 1. */
function maskWith(w: number, h: number, blocks: Array<[number, number, number, number]>): Uint8Array {
  const m = new Uint8Array(w * h);
  for (const [x0, y0, bw, bh] of blocks) {
    for (let y = y0; y < y0 + bh; y++) for (let x = x0; x < x0 + bw; x++) m[y * w + x] = 1;
  }
  return m;
}

describe('maskToPolygons (shared by SAM + classic)', () => {
  it('returns one polygon for one blob', () => {
    const mask = maskWith(40, 40, [[5, 5, 12, 12]]);
    const polys = maskToPolygons(mask, 40, 40, { minRegion: 8 });
    expect(polys.length).toBe(1);
    expect(polys[0].length).toBeGreaterThanOrEqual(6);
  });

  it('returns two polygons for two disjoint blobs', () => {
    const mask = maskWith(40, 40, [[4, 4, 10, 10], [26, 26, 10, 10]]);
    const polys = maskToPolygons(mask, 40, 40, { minRegion: 8 });
    expect(polys.length).toBe(2);
  });

  it('drops components smaller than minRegion', () => {
    const mask = maskWith(40, 40, [[4, 4, 12, 12]]);
    mask[38 * 40 + 38] = 1; // 1px speck
    const polys = maskToPolygons(mask, 40, 40, { minRegion: 12 });
    expect(polys.length).toBe(1);
  });

  it('maps coordinates to image space via scale', () => {
    const mask = maskWith(40, 40, [[4, 4, 10, 10]]);
    const polys = maskToPolygons(mask, 40, 40, { minRegion: 8, scale: 4 });
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    // grid x∈[4,14) → image x∈[~18,~58): max should clear the raw grid range.
    expect(Math.max(...xs)).toBeGreaterThan(40);
  });
});

describe('maskToPolygonsWithHoles', () => {
  const W = 60, H = 60;
  /** 60x60 mask: filled outer square minus a centered square = a donut. */
  function donut(): Uint8Array {
    const m = new Uint8Array(W * H);
    for (let y = 10; y < 50; y++) for (let x = 10; x < 50; x++) m[y * W + x] = 1; // outer
    for (let y = 22; y < 38; y++) for (let x = 22; x < 38; x++) m[y * W + x] = 0; // hole
    return m;
  }

  it('returns one outer ring and one hole for a donut', () => {
    const out = maskToPolygonsWithHoles(donut(), W, H, { minRegion: 8 });
    expect(out.length).toBe(1);
    expect(out[0].holes.length).toBe(1);
    expect(out[0].points.length).toBeGreaterThanOrEqual(6);
  });

  it('returns no holes for a solid region', () => {
    const m = new Uint8Array(W * H);
    for (let y = 10; y < 50; y++) for (let x = 10; x < 50; x++) m[y * W + x] = 1;
    const out = maskToPolygonsWithHoles(m, W, H, { minRegion: 8 });
    expect(out.length).toBe(1);
    expect(out[0].holes.length).toBe(0);
  });
});
