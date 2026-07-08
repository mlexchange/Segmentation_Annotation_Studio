import { describe, it, expect } from 'vitest';
import { fillHoles, removeSmallComponents, dilate, erode, smooth } from './morphology';

const gw = 20, gh = 20;

/** Solid filled square [x0,x0+s) × [y0,y0+s). */
function square(x0: number, y0: number, s: number): Uint8Array {
  const m = new Uint8Array(gw * gh);
  for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) m[y * gw + x] = 1;
  return m;
}

function area(m: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) n++;
  return n;
}

describe('morphology', () => {
  it('fillHoles fills an interior hole but leaves the border background alone', () => {
    const m = square(5, 5, 10);
    m[10 * gw + 10] = 0; // punch a hole
    const filled = fillHoles(m, gw, gh);
    expect(filled[10 * gw + 10]).toBe(1);
    expect(filled[0]).toBe(0); // outside untouched
    expect(area(filled)).toBe(100);
  });

  it('removeSmallComponents drops islands below the threshold', () => {
    const m = square(2, 2, 8); // area 64
    m[18 * gw + 18] = 1; // a 1px island
    const cleaned = removeSmallComponents(m, gw, gh, 10);
    expect(cleaned[18 * gw + 18]).toBe(0);
    expect(area(cleaned)).toBe(64);
  });

  it('dilate grows and erode shrinks (roughly inverse)', () => {
    const m = square(8, 8, 4); // area 16
    const grown = dilate(m, gw, gh, 1);
    expect(area(grown)).toBeGreaterThan(area(m));
    const shrunk = erode(grown, gw, gh, 1);
    expect(area(shrunk)).toBe(area(m)); // dilate then erode restores a solid square
  });

  it('smooth removes an isolated speck', () => {
    const m = new Uint8Array(gw * gh);
    m[10 * gw + 10] = 1; // lone pixel
    const s = smooth(m, gw, gh, 1);
    expect(area(s)).toBe(0);
  });
});
