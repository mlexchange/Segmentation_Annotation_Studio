import { describe, expect, it } from 'vitest';
import { buildBandOpacityCurve, mapByteBandToViewerDomain } from './bandTransferFunction';

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
    const points = buildBandOpacityCurve(80 / 255, 180 / 255);
    expectMonotonic(points);
    expect(opacityAt(points, 130 / 255)).toBeCloseTo(1, 5);
    expect(opacityAt(points, 0)).toBeCloseTo(0, 5);
    expect(opacityAt(points, 1)).toBeCloseTo(0, 5);
  });

  it('stays within the 0-1 domain the viewer expects', () => {
    const points = buildBandOpacityCurve(80 / 255, 180 / 255);
    for (const [x] of points) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
  });

  it('swaps a reversed lo/hi so the band is always well-formed', () => {
    const swapped = buildBandOpacityCurve(180 / 255, 80 / 255);
    const normal = buildBandOpacityCurve(80 / 255, 180 / 255);
    expect(swapped).toEqual(normal);
  });

  it('handles a band starting at 0 without producing non-monotonic points', () => {
    const points = buildBandOpacityCurve(0, 100 / 255);
    expectMonotonic(points);
    expect(opacityAt(points, 50 / 255)).toBeCloseTo(1, 5);
  });

  it('handles a band ending at 1 without producing non-monotonic points', () => {
    const points = buildBandOpacityCurve(150 / 255, 1);
    expectMonotonic(points);
    expect(opacityAt(points, 1)).toBeCloseTo(1, 5);
  });

  it('handles the full-range band (0-1) as fully opaque throughout', () => {
    const points = buildBandOpacityCurve(0, 1);
    expectMonotonic(points);
    expect(opacityAt(points, 0.5)).toBeCloseTo(1, 5);
  });
});

describe('mapByteBandToViewerDomain', () => {
  it('converts a byte band through the global range into the viewer range', () => {
    // Global range matches the real petiole sample from live testing.
    const globalRange: [number, number] = [-73.0, 71.3];
    const viewerRange: [number, number] = [-40, 40];
    const [lo, hi] = mapByteBandToViewerDomain(194, 215, globalRange, viewerRange);
    // raw ≈ -73 + (194/255)*144.3 ≈ 36.8 ; -73 + (215/255)*144.3 ≈ 48.6
    // domain ≈ (36.8+40)/80 ≈ 0.960 ; (48.6+40)/80 clamped to 1
    expect(lo).toBeCloseTo(0.96, 1);
    expect(hi).toBe(1);
  });

  it('falls back to treating the byte scale as already-physical when globalRange is missing', () => {
    const viewerRange: [number, number] = [0, 255];
    const [lo, hi] = mapByteBandToViewerDomain(0, 255, null, viewerRange);
    expect(lo).toBeCloseTo(0, 5);
    expect(hi).toBeCloseTo(1, 5);
  });

  it('clamps to [0,1] when the converted value falls outside the viewer range', () => {
    const globalRange: [number, number] = [0, 255];
    const viewerRange: [number, number] = [100, 200];
    const [lo, hi] = mapByteBandToViewerDomain(0, 255, globalRange, viewerRange);
    expect(lo).toBe(0);
    expect(hi).toBe(1);
  });

  it('ignores a degenerate (zero-span or inverted) global range and falls back to [0,255]', () => {
    const viewerRange: [number, number] = [0, 255];
    const [lo, hi] = mapByteBandToViewerDomain(0, 255, [10, 10], viewerRange);
    expect(lo).toBeCloseTo(0, 5);
    expect(hi).toBeCloseTo(1, 5);
  });
});
