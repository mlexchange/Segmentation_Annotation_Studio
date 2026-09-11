import { describe, it, expect } from 'vitest';
import {
  displayAffineFor,
  baseToDisplay,
  displayToBase,
  displayBandToBase,
  remapHistogramToDisplay,
} from './displayTransform';

const IDENTITY = displayAffineFor(0, 0, 0, 255);

describe('displayAffineFor', () => {
  it('is the identity with neutral settings', () => {
    expect(IDENTITY.slope).toBeCloseTo(1, 6);
    expect(IDENTITY.intercept).toBeCloseTo(0, 6);
    for (const v of [0, 37, 128, 200, 255]) {
      expect(baseToDisplay(v, IDENTITY)).toBeCloseTo(v, 4);
    }
  });

  it('matches renderAdjusted for brightness + contrast + levels', () => {
    // Mirror of the pipeline: brighten, contrast about mid-grey, then levels.
    const brightness = 0.2, contrast = 40, lo = 20, hi = 200;
    const affine = displayAffineFor(brightness, contrast, lo, hi);
    const manual = (v: number) => {
      const adjust = Math.pow((contrast + 100) / 100, 2);
      let x = v + brightness * 255;
      x = ((x / 255 - 0.5) * adjust + 0.5) * 255;
      x = x <= lo ? 0 : x >= hi ? 255 : ((x - lo) * 255) / (hi - lo);
      return x;
    };
    for (const v of [0, 30, 64, 128, 190, 255]) {
      expect(baseToDisplay(v, affine)).toBeCloseTo(Math.max(0, Math.min(255, manual(v))), 3);
    }
  });
});

describe('displayToBase', () => {
  it('round-trips baseToDisplay in the unclamped interior', () => {
    const affine = displayAffineFor(0.1, 25, 10, 240);
    for (const v of [60, 100, 128, 160]) {
      const shown = baseToDisplay(v, affine);
      expect(displayToBase(shown, affine)).toBeCloseTo(v, 3);
    }
  });

  it('round-trips through gamma too', () => {
    const affine = displayAffineFor(0, 0, 0, 255);
    for (const gamma of [0.5, 1, 2.2]) {
      for (const v of [40, 128, 210]) {
        const shown = baseToDisplay(v, affine, gamma);
        expect(displayToBase(shown, affine, gamma)).toBeCloseTo(v, 3);
      }
    }
  });

  it('returns null when the transform collapses (contrast -100)', () => {
    const affine = displayAffineFor(0, -100, 0, 255);
    expect(displayToBase(128, affine)).toBeNull();
  });
});

describe('displayBandToBase', () => {
  it('selects the same pixels as thresholding the adjusted image', () => {
    const affine = displayAffineFor(0.15, 30, 15, 230);
    const gamma = 1.4;
    const lo = 90, hi = 180;
    const band = displayBandToBase(lo, hi, affine, gamma);
    // For every base value, gating in base space must agree with gating on the
    // value the screen actually shows — that equivalence is the whole point.
    for (let v = 0; v <= 255; v++) {
      const shown = baseToDisplay(v, affine, gamma);
      const viaDisplay = shown >= lo && shown <= hi;
      const viaBase = v >= band.lo && v <= band.hi;
      expect(viaBase).toBe(viaDisplay);
    }
  });

  it('treats band edges at 0 / 255 as open', () => {
    const affine = displayAffineFor(0.3, 0, 0, 255);
    const band = displayBandToBase(0, 255, affine);
    expect(band.lo).toBe(-Infinity);
    expect(band.hi).toBe(Infinity);
  });

  it('is all-or-nothing when the transform collapses', () => {
    const affine = displayAffineFor(0, -100, 0, 255);
    // Everything renders mid-grey, so a band containing it selects everything.
    const covering = displayBandToBase(100, 200, affine);
    expect(covering.lo).toBe(-Infinity);
    expect(covering.hi).toBe(Infinity);
    const missing = displayBandToBase(0.5, 3, affine);
    expect(missing.lo).toBeGreaterThan(missing.hi); // empty band
  });
});

describe('remapHistogramToDisplay', () => {
  it('preserves total count', () => {
    const bins = new Array(256).fill(0).map((_, i) => i);
    const out = remapHistogramToDisplay(bins, displayAffineFor(0.1, 20, 0, 255));
    const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
    expect(sum(out)).toBe(sum(bins));
  });

  it('is a no-op for the identity transform', () => {
    const bins = new Array(256).fill(0);
    bins[10] = 5; bins[200] = 7;
    expect(remapHistogramToDisplay(bins, IDENTITY)).toEqual(bins);
  });

  it('shifts mass brighter when brightness increases', () => {
    const bins = new Array(256).fill(0);
    bins[100] = 1000;
    const out = remapHistogramToDisplay(bins, displayAffineFor(0.2, 0, 0, 255));
    expect(out[100]).toBe(0);
    expect(out.findIndex((c) => c > 0)).toBeGreaterThan(100);
  });
});
