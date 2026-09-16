import { describe, it, expect } from 'vitest';
import { rasterizeShapes, gridFor, fullResGridFor, stampStroke } from './rasterize';
import { maskToPolygons } from './magicwand';
import type { Shape } from '@/stores/annotationStore';

/** Count set pixels in a binary mask. */
function area(mask: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
  return n;
}

describe('rasterizeShapes', () => {
  it('fills a rectangle with the expected area', () => {
    const rect: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 10, y: 20, w: 30, h: 40 };
    const mask = rasterizeShapes([rect], 100, 100, 1);
    // ~30x40 = 1200 px, allow small boundary rounding.
    expect(area(mask)).toBeGreaterThan(1100);
    expect(area(mask)).toBeLessThan(1400);
    expect(mask[25 * 100 + 25]).toBe(1); // interior set
    expect(mask[5 * 100 + 5]).toBe(0); // outside clear
  });

  it('fills a polygon (triangle ~ half the bounding box)', () => {
    const tri: Shape = { id: 't', classId: 0, kind: 'polygon', points: [0, 0, 40, 0, 0, 40] };
    const mask = rasterizeShapes([tri], 40, 40, 1);
    const a = area(mask);
    expect(a).toBeGreaterThan(600); // ~800 (half of 1600), leave slack
    expect(a).toBeLessThan(900);
  });

  it('carves an erase stroke back out of a rectangle', () => {
    const rect: Shape = {
      id: 'r', classId: 0, kind: 'rectangle', x: 0, y: 0, w: 40, h: 40,
      erased: [{ points: [20, 20, 20, 20], radius: 5 }],
    };
    const mask = rasterizeShapes([rect], 40, 40, 1);
    expect(mask[20 * 40 + 20]).toBe(0); // erased center
    expect(mask[2 * 40 + 2]).toBe(1); // corner still set
  });

  it('leaves a polygon hole empty (invert-shape complement)', () => {
    // Outer = full 40x40 frame; hole = centered 20x20 square.
    const inverted: Shape = {
      id: 'inv', classId: 0, kind: 'polygon',
      points: [0, 0, 40, 0, 40, 40, 0, 40],
      holes: [[10, 10, 30, 10, 30, 30, 10, 30]],
    };
    const mask = rasterizeShapes([inverted], 40, 40, 1);
    expect(mask[20 * 40 + 20]).toBe(0); // inside the hole → empty
    expect(mask[2 * 40 + 2]).toBe(1);   // frame corner → filled
    // Area ≈ 40² − 20² = 1200, allow boundary slack.
    expect(area(mask)).toBeGreaterThan(1050);
    expect(area(mask)).toBeLessThan(1350);
  });

  it('round-trips shape → mask → polygons preserving rough area', () => {
    const rect: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 20, y: 20, w: 60, h: 60 };
    const { gw, gh, scale } = gridFor(200, 200);
    const mask = rasterizeShapes([rect], gw, gh, scale);
    const polys = maskToPolygons(mask, gw, gh, { minRegion: 4, scale });
    expect(polys.length).toBe(1);
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    const ys = polys[0].filter((_, i) => i % 2 === 1);
    // Vectorized polygon should sit within a few px of the original rect.
    expect(Math.min(...xs)).toBeLessThan(26);
    expect(Math.max(...xs)).toBeGreaterThan(74);
    expect(Math.min(...ys)).toBeLessThan(26);
    expect(Math.max(...ys)).toBeGreaterThan(74);
  });
});

describe('fullResGridFor upscale', () => {
  it('is unchanged at 1x', () => {
    expect(fullResGridFor(200, 100)).toEqual({ gw: 200, gh: 100, scale: 1 });
  });

  it('doubles the grid and halves the scale at 2x', () => {
    expect(fullResGridFor(200, 100, 2)).toEqual({ gw: 400, gh: 200, scale: 0.5 });
  });

  it('keeps the downsample guard for very large images', () => {
    // >4096 falls back to the 1600 cap (scale 4 natively), then 2x halves it.
    const g = fullResGridFor(6400, 6400, 2);
    expect(g.scale).toBe(2);
    expect(g.gw).toBe(3200);
  });

  it('round-trips a rect through an upscaled grid', () => {
    const rect: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 20, y: 20, w: 60, h: 60 };
    const { gw, gh, scale } = fullResGridFor(200, 200, 2);
    const mask = rasterizeShapes([rect], gw, gh, scale);
    // 60x60 image px at 2x = 120x120 grid cells.
    expect(area(mask)).toBeGreaterThan(14000);
    expect(area(mask)).toBeLessThan(14700);
    const polys = maskToPolygons(mask, gw, gh, { minRegion: 4, scale });
    const xs = polys[0].filter((_, i) => i % 2 === 0);
    expect(Math.min(...xs)).toBeGreaterThan(18); // still native coords, not doubled
    expect(Math.max(...xs)).toBeLessThan(82);
  });
});

describe('stampStroke gate', () => {
  it('restricts the stamp to gated cells (Threshold Brush band)', () => {
    const gw = 20, gh = 20;
    // Gate lets only the left half through.
    const gate = new Uint8Array(gw * gh);
    for (let y = 0; y < gh; y++) for (let x = 0; x < 10; x++) gate[y * gw + x] = 1;

    const ungated = new Uint8Array(gw * gh);
    stampStroke(ungated, gw, gh, [5, 10, 15, 10], 3, 1, 1);
    const gated = new Uint8Array(gw * gh);
    stampStroke(gated, gw, gh, [5, 10, 15, 10], 3, 1, 1, gate);

    expect(area(gated)).toBeGreaterThan(0);
    expect(area(gated)).toBeLessThan(area(ungated));
    // Nothing landed outside the gate.
    for (let i = 0; i < gated.length; i++) if (gated[i]) expect(gate[i]).toBe(1);
  });
});
