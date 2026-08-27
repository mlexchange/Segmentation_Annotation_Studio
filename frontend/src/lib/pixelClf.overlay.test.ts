import { describe, expect, it } from 'vitest';
import { colorizeConformalOverlay } from './pixelClf';

describe('colorizeConformalOverlay filters', () => {
  it('builds a canvas with class / multi / abstain filters without throwing', () => {
    const w = 2;
    const h = 2;
    const commit = new Uint8Array([1, 2, 0, 0]);
    const status = new Uint8Array([1, 1, 2, 0]);
    const colors = new Map([
      [1, '#ff0000'],
      [2, '#00ff00'],
    ]);

    const canvas = colorizeConformalOverlay(commit, status, w, h, colors, {
      classVisible: (cid) => cid === 1,
      showMulti: false,
      showAbstain: false,
    });
    expect(canvas.width).toBe(2);
    expect(canvas.height).toBe(2);
  });
});
