import { describe, it, expect } from 'vitest';
import { dijkstra, tracePath, imageToGrid, simplifyPath, type CostMap } from './livewire';

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

  it('simplifyPath keeps endpoints and thins the middle', () => {
    const dense = [0, 0, 1, 0, 2, 0, 3, 0, 4, 0]; // 5 points
    const simplified = simplifyPath(dense, 2);
    expect(simplified.slice(0, 2)).toEqual([0, 0]);
    expect(simplified.slice(-2)).toEqual([4, 0]);
    expect(simplified.length).toBeLessThan(dense.length);
  });
});
