import { describe, it, expect } from 'vitest';
import { keepComponentsAtPoints } from './morphology';

/** 10×10 grid with two separate blocks: A at (1..3,1..3), B (speckle) at (7..8,7..8). */
function twoBlocks(): { mask: Uint8Array; gw: number; gh: number } {
  const gw = 10, gh = 10;
  const mask = new Uint8Array(gw * gh);
  for (let y = 1; y <= 3; y++) for (let x = 1; x <= 3; x++) mask[y * gw + x] = 1;
  for (let y = 7; y <= 8; y++) for (let x = 7; x <= 8; x++) mask[y * gw + x] = 1;
  return { mask, gw, gh };
}

describe('keepComponentsAtPoints', () => {
  it('keeps only the component containing the prompt point', () => {
    const { mask, gw, gh } = twoBlocks();
    const out = keepComponentsAtPoints(mask, gw, gh, [{ x: 2, y: 2 }]);
    expect(out[2 * gw + 2]).toBe(1);   // block A kept
    expect(out[7 * gw + 7]).toBe(0);   // speckle B dropped
    let n = 0; for (const v of out) n += v;
    expect(n).toBe(9);                 // only the 3×3 block
  });

  it('falls back to the largest component when no point hits foreground', () => {
    const { mask, gw, gh } = twoBlocks();
    const out = keepComponentsAtPoints(mask, gw, gh, [{ x: 5, y: 5 }]); // background
    expect(out[2 * gw + 2]).toBe(1);   // larger block A kept
    expect(out[7 * gw + 7]).toBe(0);   // smaller block B dropped
  });

  it('keeps multiple components when multiple points hit', () => {
    const { mask, gw, gh } = twoBlocks();
    const out = keepComponentsAtPoints(mask, gw, gh, [{ x: 2, y: 2 }, { x: 7, y: 7 }]);
    expect(out[2 * gw + 2]).toBe(1);
    expect(out[7 * gw + 7]).toBe(1);
  });
});
