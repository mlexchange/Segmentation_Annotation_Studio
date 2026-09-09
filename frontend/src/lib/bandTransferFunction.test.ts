import { describe, expect, it } from 'vitest';
import { buildBandOpacityCurve } from './bandTransferFunction';

/** Every curve must have strictly increasing x across its points. */
function expectMonotonic(points: readonly (readonly [number, number])[]) {
  for (let i = 1; i < points.length; i++) {
    expect(points[i][0]).toBeGreaterThan(points[i - 1][0]);
  }
}

/** Linear interpolation, mirroring how a GPU sampler would read the curve. */
function opacityAt(points: readonly (readonly [number, number])[], x: number): number {
  if (x <= points[0][0]) return points[0][1];
  for (let i = 1; i < points.length; i++) {
    if (x <= points[i][0]) {
      const [x0, o0] = points[i - 1];
      const [x1, o1] = points[i];
      const t = (x - x0) / (x1 - x0);
      return o0 + (o1 - o0) * t;
    }
  }
  return points[points.length - 1][1];
}

describe('buildBandOpacityCurve', () => {
  it('is opaque in the middle of the band and transparent well outside it', () => {
    const points = buildBandOpacityCurve(80, 180);
    expectMonotonic(points);
    expect(opacityAt(points, 130 / 255)).toBeCloseTo(1, 5);
    expect(opacityAt(points, 0)).toBeCloseTo(0, 5);
    expect(opacityAt(points, 1)).toBeCloseTo(0, 5);
  });

  it('normalizes 0-255 input into the 0-1 domain the viewer expects', () => {
    const points = buildBandOpacityCurve(80, 180);
    for (const [x] of points) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
  });

  it('swaps a reversed lo/hi so the band is always well-formed', () => {
    const swapped = buildBandOpacityCurve(180, 80);
    const normal = buildBandOpacityCurve(80, 180);
    expect(swapped).toEqual(normal);
  });

  it('handles a band starting at 0 without producing non-monotonic points', () => {
    const points = buildBandOpacityCurve(0, 100);
    expectMonotonic(points);
    expect(opacityAt(points, 50 / 255)).toBeCloseTo(1, 5);
  });

  it('handles a band ending at 255 without producing non-monotonic points', () => {
    const points = buildBandOpacityCurve(150, 255);
    expectMonotonic(points);
    expect(opacityAt(points, 1)).toBeCloseTo(1, 5);
  });

  it('handles the full-range band (0-255) as fully opaque throughout', () => {
    const points = buildBandOpacityCurve(0, 255);
    expectMonotonic(points);
    expect(opacityAt(points, 0.5)).toBeCloseTo(1, 5);
  });
});
