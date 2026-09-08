import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildCostMap, dijkstra, tracePath, imageToGrid, simplifyPath, type CostMap } from './livewire';

/** 5x5 grid, scale 1, with a zero-cost "edge" corridor along row y=2. */
function corridorMap(): CostMap {
  const gw = 5, gh = 5;
  const cost = new Float32Array(gw * gh).fill(1);
  for (let x = 0; x < gw; x++) cost[2 * gw + x] = 0; // strong edge on row 2
  return { gw, gh, scale: 1, cost };
}

describe('livewire', () => {
  it('imageToGrid clamps and maps to the right cell', () => {
    const cm = corridorMap();
    expect(imageToGrid(cm, 0, 0)).toBe(0);
    expect(imageToGrid(cm, 4, 2)).toBe(2 * 5 + 4);
    expect(imageToGrid(cm, 999, 999)).toBe(5 * 5 - 1); // clamped
  });

  it('traces the least-cost path along the edge corridor', () => {
    const cm = corridorMap();
    const seed = imageToGrid(cm, 0, 2);
    const prev = dijkstra(cm, seed);
    const path = tracePath(cm, prev, imageToGrid(cm, 4, 2));
    // Path is flat [x,y,...]; every vertex should sit on the corridor (y center = 2.5).
    expect(path.length).toBeGreaterThanOrEqual(4);
    for (let i = 1; i < path.length; i += 2) {
      expect(path[i]).toBeCloseTo(2.5, 5);
    }
    // Starts at the seed, ends at the target.
    expect(path[0]).toBeCloseTo(0.5, 5);
    expect(path[path.length - 2]).toBeCloseTo(4.5, 5);
  });

  it('maps and traces in native coords at a fractional (upscaled) scale', () => {
    // 6x6 grid at scale 0.5 = a 3x3 NATIVE image sampled at 2x.
    const gw = 6, gh = 6;
    const cost = new Float32Array(gw * gh).fill(1);
    for (let x = 0; x < gw; x++) cost[2 * gw + x] = 0;
    const cm: CostMap = { gw, gh, scale: 0.5, cost };
    // Native x=1 → grid cell 2 (1 / 0.5).
    expect(imageToGrid(cm, 1, 1)).toBe(2 * gw + 2);
    const prev = dijkstra(cm, imageToGrid(cm, 0, 1));
    const path = tracePath(cm, prev, imageToGrid(cm, 2.5, 1));
    // Vertices are cell centers in NATIVE coords: gy=2 → 2*0.5 + 0.25 = 1.25.
    for (let i = 1; i < path.length; i += 2) expect(path[i]).toBeCloseTo(1.25, 5);
    // And they stay inside the 3-px-wide native image.
    for (let i = 0; i < path.length; i += 2) expect(path[i]).toBeLessThanOrEqual(3);
  });

  it('simplifyPath keeps endpoints and thins the middle', () => {
    const dense = [0, 0, 1, 0, 2, 0, 3, 0, 4, 0]; // 5 points
    const simplified = simplifyPath(dense, 2);
    expect(simplified.slice(0, 2)).toEqual([0, 0]);
    expect(simplified.slice(-2)).toEqual([4, 0]);
    expect(simplified.length).toBeLessThan(dense.length);
  });

  describe('buildCostMap', () => {
    const stores = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();

    function seedSource(width: number, height: number, fill: (i: number) => number): HTMLCanvasElement {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const data = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < data.length; i++) data[i] = fill(i);
      stores.set(canvas, data);
      return canvas;
    }

    beforeEach(() => {
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
        this: HTMLCanvasElement,
      ): any {
        const canvas = this;
        return {
          drawImage: (source: HTMLCanvasElement, _sx: number, _sy: number, w: number, h: number) => {
            const src = stores.get(source);
            const dst = new Uint8ClampedArray(w * h * 4);
            if (src) dst.set(src.subarray(0, dst.length));
            stores.set(canvas, dst);
          },
          getImageData: (_x: number, _y: number, w: number, h: number) => ({
            data: stores.get(canvas) ?? new Uint8ClampedArray(w * h * 4),
            width: w,
            height: h,
          }),
        };
      });
    });

    afterEach(() => vi.restoreAllMocks());

    it('downsamples so the long side stays within maxDim', () => {
      const source = seedSource(1000, 500, () => 128);
      const cm = buildCostMap(source, 1000, 500, 100);
      expect(cm).not.toBeNull();
      expect(Math.max(cm!.gw, cm!.gh)).toBeLessThanOrEqual(100);
    });

    it('produces near-zero cost on a strong edge and near-one cost on a flat region', () => {
      // A 10x10 image split vertically: black left half, white right half —
      // a strong vertical edge at x=5.
      const source = seedSource(10, 10, (i) => {
        if (i % 4 === 3) return 255; // alpha
        const px = Math.floor(i / 4);
        const x = px % 10;
        return x < 5 ? 0 : 255;
      });
      const cm = buildCostMap(source, 10, 10, 512)!;
      expect(cm).not.toBeNull();
      // A flat region far from the edge should have cost close to 1 (no gradient).
      const flatIdx = imageToGrid(cm, 1, 5);
      // The edge column should have a lower cost than the flat region.
      const edgeIdx = imageToGrid(cm, 5, 5);
      expect(cm.cost[edgeIdx]).toBeLessThan(cm.cost[flatIdx]);
    });

    it('scales the grid by the upscale factor', () => {
      const source = seedSource(4, 4, () => 100);
      const cm1 = buildCostMap(source, 4, 4, 512, 1)!;
      const cm2 = buildCostMap(source, 4, 4, 512, 2)!;
      expect(cm2.gw).toBeGreaterThan(cm1.gw);
    });

    it('returns null when the canvas cannot produce a 2D context', () => {
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as any);
      const source = seedSource(4, 4, () => 0);
      expect(buildCostMap(source, 4, 4)).toBeNull();
    });

    it('returns null when getImageData throws (tainted canvas)', () => {
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (): any {
        return {
          drawImage: () => {},
          getImageData: () => { throw new Error('tainted'); },
        };
      });
      const source = seedSource(4, 4, () => 0);
      expect(buildCostMap(source, 4, 4)).toBeNull();
    });
  });
});
