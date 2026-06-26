import { describe, it, expect } from 'vitest';
import { magicSelect, type GrayField } from './magicwand';

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
