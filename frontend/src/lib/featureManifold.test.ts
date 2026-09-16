import { describe, expect, it } from 'vitest';
import { colorizeManifoldHeatmap, manifoldMarkerRect } from './featureManifold';

describe('colorizeManifoldHeatmap', () => {
  it('returns a canvas matching the grayscale map size', () => {
    const w = 4;
    const h = 3;
    const gray = new Uint8Array(w * h);
    gray[0] = 255;
    gray[1] = 0;
    const canvas = colorizeManifoldHeatmap(gray, w, h, 0.5);
    expect(canvas.width).toBe(w);
    expect(canvas.height).toBe(h);
  });
});

describe('manifoldMarkerRect', () => {
  it('uses server box coordinates when present (no edge shift)', () => {
    const r = manifoldMarkerRect(
      { x: 10, y: 10, box_size: 64, box: { x0: 0, y0: 0, x1: 32, y1: 32 } },
      { side: 64, width: 400, height: 300 },
    );
    expect(r).toEqual({ x: 0, y: 0, width: 32, height: 32 });
  });

  it('clips reconstructed squares instead of shifting them', () => {
    const r = manifoldMarkerRect(
      { x: 10, y: 50 },
      { side: 64, width: 200, height: 200 },
    );
    expect(r.x).toBe(0);
    expect(r.width).toBe(42); // 10 + 32, clipped — not shifted to 64 wide
    expect(r.y).toBe(18);
    expect(r.height).toBe(64);
  });
});
