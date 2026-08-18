import { describe, it, expect } from 'vitest';
import { magicSelect, maskToPolygons, maskToPolygonsWithHoles, gradientField, otsuThreshold, type GrayField } from './magicwand';

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

  it('attaches the hole even when the outer region is concave (magic-like)', () => {
    // A "C"/notched block (concave outer) with an enclosed square hole. The hole's
    // centroid can fall in the notch (outside the outer), so this exercises the
    // single-region direct-assign path rather than the centroid test.
    const m = new Uint8Array(W * H);
    for (let y = 8; y < 52; y++) for (let x = 8; x < 52; x++) m[y * W + x] = 1; // block
    for (let y = 8; y < 30; y++) for (let x = 40; x < 52; x++) m[y * W + x] = 0; // notch (concavity)
    for (let y = 34; y < 46; y++) for (let x = 16; x < 28; x++) m[y * W + x] = 0; // enclosed hole
    const out = maskToPolygonsWithHoles(m, W, H, { minRegion: 8 });
    expect(out.length).toBe(1);
    expect(out[0].holes.length).toBe(1);
  });

  it('returns no holes for a solid region', () => {
    const m = new Uint8Array(W * H);
    for (let y = 10; y < 50; y++) for (let x = 10; x < 50; x++) m[y * W + x] = 1;
    const out = maskToPolygonsWithHoles(m, W, H, { minRegion: 8 });
    expect(out.length).toBe(1);
    expect(out[0].holes.length).toBe(0);
  });
});

describe('fractional scale (upscaled working resolution)', () => {
  /** 40x40 grid at scale 0.5 — i.e. a 20x20 NATIVE image sampled at 2x. */
  function upscaledBlock(): GrayField {
    const gw = 40, gh = 40;
    const gray = new Float32Array(gw * gh);
    // Grid cells 8..24 → native image coords 4..12.
    for (let y = 8; y < 24; y++) for (let x = 8; x < 24; x++) gray[y * gw + x] = 200;
    return { gw, gh, scale: 0.5, gray };
  }

  it('magicSelect seeds and returns polygons in NATIVE image coords', () => {
    const field = upscaledBlock();
    // Seed at native (8,8) → grid (16,16), inside the block.
    const polys = magicSelect(field, 8, 8, { toleranceFrac: 0.2, mode: 'contiguous', smooth: 0, minRegion: 8 });
    expect(polys.length).toBe(1);
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    const ys = polys[0].filter((_, i) => i % 2 === 1);
    // Native extent of the block is 4..12, not the 8..24 grid extent.
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...xs)).toBeLessThanOrEqual(13);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...ys)).toBeLessThanOrEqual(13);
  });

  it('yields sub-pixel (half-integer) vertices a native-resolution grid could not', () => {
    const field = upscaledBlock();
    const polys = magicSelect(field, 8, 8, { toleranceFrac: 0.2, mode: 'contiguous', smooth: 0, minRegion: 8 });
    const coords = polys[0];
    expect(coords.some((v) => !Number.isInteger(v))).toBe(true);
  });
});

describe('otsuThreshold', () => {
  it('splits a clean bimodal histogram between the modes', () => {
    const bins = new Array(256).fill(0);
    bins[40] = 1000;   // dark mode
    bins[200] = 1000;  // bright mode
    const t = otsuThreshold(bins);
    expect(t).toBeGreaterThan(40);
    expect(t).toBeLessThan(200);
  });

  it('handles broad overlapping modes', () => {
    const bins = new Array(256).fill(0);
    for (let i = 30; i < 70; i++) bins[i] = 100;
    for (let i = 150; i < 220; i++) bins[i] = 100;
    const t = otsuThreshold(bins);
    expect(t).toBeGreaterThanOrEqual(69);
    expect(t).toBeLessThanOrEqual(150);
  });

  it('returns a safe default for degenerate histograms', () => {
    expect(otsuThreshold([])).toBe(128);
    expect(otsuThreshold(new Array(256).fill(0))).toBe(128);
    // Single-valued: no valid two-class split, so the default stands.
    const one = new Array(256).fill(0);
    one[77] = 500;
    expect(otsuThreshold(one)).toBe(128);
  });
});
