import { describe, it, expect } from 'vitest';
import { rasterizeShapes, gridFor } from './rasterize';
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
