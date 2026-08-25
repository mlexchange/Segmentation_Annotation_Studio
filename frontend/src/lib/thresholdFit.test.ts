import { describe, it, expect } from 'vitest';
import {
  fitBand,
  describeFit,
  sampleHistograms,
  backgroundRing,
  fitWithBlurSweep,
  combineSigma,
  ringWidthFor,
  MAX_RING_WIDTH,
  type BandFit,
} from './thresholdFit';
import { dilate } from './morphology';
import { applyGaussianBlurGray } from './blur';
import { displayAffineFor, baseToDisplay, displayBandToBase, displayToBase } from './displayTransform';

/** Histogram with `count` samples placed at each listed bin. */
function hist(entries: Array<[number, number]>): Float64Array {
  const h = new Float64Array(256);
  for (const [bin, count] of entries) h[bin] += count;
  return h;
}

/** Spread `count` samples uniformly across [lo,hi]. */
function band(lo: number, hi: number, count: number): Float64Array {
  const h = new Float64Array(256);
  const per = count / (hi - lo + 1);
  for (let i = lo; i <= hi; i++) h[i] += per;
  return h;
}

/** Reference implementation: the same score, computed the slow obvious way. */
function bruteForce(pos: ArrayLike<number>, neg: ArrayLike<number>): { lo: number; hi: number; dice: number } {
  let best = { lo: 0, hi: 0, dice: -1 };
  let totalPos = 0;
  for (let i = 0; i < 256; i++) totalPos += pos[i];
  for (let lo = 0; lo < 256; lo++) {
    for (let hi = lo; hi < 256; hi++) {
      let tp = 0;
      let fp = 0;
      for (let i = lo; i <= hi; i++) { tp += pos[i]; fp += neg[i]; }
      if (tp === 0) continue;
      const dice = (2 * tp) / (2 * tp + fp + (totalPos - tp));
      if (dice > best.dice) best = { lo, hi, dice };
    }
  }
  return best;
}

describe('fitBand — cleanly separated material', () => {
  it('recovers the planted band almost exactly', () => {
    // Feature at 100–140, surroundings well away at 10–40.
    const fit = fitBand(band(100, 140, 1000), band(10, 40, 4000));
    expect(fit.ok).toBe(true);
    expect(fit.lo).toBeLessThanOrEqual(100);
    expect(fit.hi).toBeGreaterThanOrEqual(140);
    expect(fit.dice).toBeGreaterThan(0.99);
    expect(fit.coverage).toBeGreaterThan(0.99);
    expect(fit.leakage).toBeLessThan(0.01);
  });

  it('excludes a nearby but distinct background band', () => {
    const fit = fitBand(band(120, 160, 1000), band(60, 110, 1000));
    // The gap is at 111–119; the fitted low edge must sit inside it.
    expect(fit.lo).toBeGreaterThan(110);
    expect(fit.dice).toBeGreaterThan(0.95);
  });
});

describe('fitBand — overlapping material', () => {
  it('reports weak separation instead of a falsely confident band', () => {
    // Same range for both: no band can separate them.
    const fit = fitBand(band(80, 160, 1000), band(80, 160, 1000));
    expect(fit.ok).toBe(true);
    // Dice looks deceptively respectable here (0.67 is the FLOOR against an
    // equal-sized ring, not a decent result) — skill is what exposes it.
    expect(fit.dice).toBeCloseTo(fit.baselineDice, 6);
    expect(fit.skill).toBeCloseTo(0, 6);
    expect(describeFit(fit).quality).toBe('weak');
  });

  it('still returns the best achievable band on partial overlap', () => {
    // Positives 100–160, negatives 130–200: the useful part is 100–129.
    const fit = fitBand(band(100, 160, 1000), band(130, 200, 1000));
    expect(fit.dice).toBeGreaterThan(0.5);
    expect(fit.dice).toBeLessThan(0.95);
    expect(fit.lo).toBeLessThanOrEqual(100);
    // It should stop before swallowing the whole negative range.
    expect(fit.hi).toBeLessThan(200);
  });

  it('grades fits by skill, not by raw Dice', () => {
    const mk = (skill: number): BandFit => ({
      lo: 0, hi: 1, dice: 0.9, baselineDice: 0.67, skill, coverage: 1, leakage: 0, ok: true,
    });
    expect(describeFit(mk(0.95)).quality).toBe('good');
    expect(describeFit(mk(0.5)).quality).toBe('fair');
    // High Dice but no skill => still weak, which is the whole point.
    expect(describeFit(mk(0.05)).quality).toBe('weak');
  });

  it('reports skill relative to the ring size, so grading is size-independent', () => {
    // Same inseparable overlap, but a 4x larger ring: Dice collapses to 0.33
    // while skill stays at 0 — the grade must not move.
    const small = fitBand(band(80, 160, 1000), band(80, 160, 1000));
    const large = fitBand(band(80, 160, 1000), band(80, 160, 4000));
    expect(small.dice).toBeGreaterThan(large.dice + 0.2);
    expect(small.skill).toBeCloseTo(0, 6);
    expect(large.skill).toBeCloseTo(0, 6);
    expect(describeFit(small).quality).toBe(describeFit(large).quality);
  });
});

describe('fitBand — agrees with a brute-force reference', () => {
  function mulberry32(seed: number) {
    return () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  for (const seed of [1, 17, 99, 2024]) {
    it(`matches on random histograms (seed ${seed})`, () => {
      const rnd = mulberry32(seed);
      const pos = new Float64Array(256);
      const neg = new Float64Array(256);
      for (let i = 0; i < 256; i++) {
        pos[i] = Math.floor(rnd() * 30);
        neg[i] = Math.floor(rnd() * 30);
      }
      const got = fitBand(pos, neg);
      const want = bruteForce(pos, neg);
      // The prefix-sum search must find the same optimum as the naive one.
      expect(got.dice).toBeCloseTo(want.dice, 10);
    });
  }
});

describe('fitBand — degenerate inputs', () => {
  it('reports not-ok when nothing was sampled', () => {
    const fit = fitBand(new Float64Array(256), band(10, 20, 100));
    expect(fit.ok).toBe(false);
    expect(fit.dice).toBe(0);
  });

  it('handles an empty negative ring (selects the whole sample)', () => {
    const fit = fitBand(band(50, 90, 500), new Float64Array(256));
    expect(fit.ok).toBe(true);
    expect(fit.dice).toBeCloseTo(1, 6);
    expect(fit.leakage).toBe(0);
  });

  it('handles a single-valued sample', () => {
    const fit = fitBand(hist([[123, 400]]), hist([[7, 400]]));
    expect(fit.lo).toBeLessThanOrEqual(123);
    expect(fit.hi).toBeGreaterThanOrEqual(123);
    expect(fit.dice).toBeCloseTo(1, 6);
  });

  it('never returns NaN when both inputs are empty', () => {
    const fit = fitBand(new Float64Array(256), new Float64Array(256));
    expect(fit.ok).toBe(false);
    expect(Number.isNaN(fit.dice)).toBe(false);
  });

  it('handles identical positive and negative sets without dividing by zero', () => {
    const same = band(100, 100, 50);
    const fit = fitBand(same, same);
    expect(Number.isFinite(fit.dice)).toBe(true);
    expect(fit.dice).toBeCloseTo(2 / 3, 6); // 2TP/(2TP+FP+FN) with FP=TP, FN=0
  });
});

describe('sampleHistograms', () => {
  it('bins each mask separately and ignores unmasked pixels', () => {
    const field = [10, 10, 200, 200, 50, 50];
    const pos = [1, 1, 0, 0, 0, 0];
    const neg = [0, 0, 1, 1, 0, 0];
    const { posHist, negHist, posCount, negCount } = sampleHistograms(field, pos, neg);
    expect(posCount).toBe(2);
    expect(negCount).toBe(2);
    expect(posHist[10]).toBe(2);
    expect(negHist[200]).toBe(2);
    expect(posHist[50]).toBe(0); // outside both masks
  });

  it('clamps out-of-range and skips non-finite values', () => {
    const field = [-40, 999, NaN, 128];
    const pos = [1, 1, 1, 1];
    const neg = [0, 0, 0, 0];
    const { posHist, posCount } = sampleHistograms(field, pos, neg);
    expect(posHist[0]).toBe(1);   // -40 clamped up
    expect(posHist[255]).toBe(1); // 999 clamped down
    expect(posHist[128]).toBe(1);
    // A non-finite pixel is not a usable sample, so it is skipped entirely
    // rather than being counted toward the sample size.
    expect(posCount).toBe(3);
  });

  it('treats a pixel in both masks as positive', () => {
    const { posHist, negHist } = sampleHistograms([77], [1], [1]);
    expect(posHist[77]).toBe(1);
    expect(negHist[77]).toBe(0);
  });
});

describe('backgroundRing', () => {
  it('surrounds the mask without overlapping it', () => {
    const gw = 40, gh = 40;
    const mask = new Uint8Array(gw * gh);
    for (let y = 15; y < 25; y++) for (let x = 15; x < 25; x++) mask[y * gw + x] = 1;

    const ring = backgroundRing(mask, gw, gh, dilate, 3);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) expect(ring[i]).toBe(0); // disjoint from the sample
    }
    let ringCount = 0;
    for (let i = 0; i < ring.length; i++) if (ring[i]) ringCount++;
    expect(ringCount).toBeGreaterThan(0);
  });

  it('scales its width with the region when not given one', () => {
    const gw = 80, gh = 80;
    const small = new Uint8Array(gw * gh);
    for (let y = 40; y < 44; y++) for (let x = 40; x < 44; x++) small[y * gw + x] = 1;
    const big = new Uint8Array(gw * gh);
    for (let y = 20; y < 60; y++) for (let x = 20; x < 60; x++) big[y * gw + x] = 1;

    const area = (m: Uint8Array) => m.reduce((n, v) => n + (v ? 1 : 0), 0);
    const ringSmall = area(backgroundRing(small, gw, gh, dilate));
    const ringBig = area(backgroundRing(big, gw, gh, dilate));
    expect(ringBig).toBeGreaterThan(ringSmall);
  });

  it('returns an empty ring for an empty mask', () => {
    const empty = new Uint8Array(100);
    const ring = backgroundRing(empty, 10, 10, dilate);
    expect(ring.some((v) => v === 1)).toBe(false);
  });
});

describe('fitWithBlurSweep', () => {
  /** A speckled target: alternating pixels stray into the background's range. */
  function speckled(w: number, h: number, targetVal: number, bgVal: number, noise: number) {
    const field = new Float32Array(w * h);
    const pos = new Uint8Array(w * h);
    const neg = new Uint8Array(w * h);
    let seed = 7;
    const rnd = () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const inside = x >= w / 4 && x < (3 * w) / 4 && y >= h / 4 && y < (3 * h) / 4;
        field[i] = (inside ? targetVal : bgVal) + (rnd() - 0.5) * 2 * noise;
        if (inside) pos[i] = 1; else neg[i] = 1;
      }
    }
    return { field, pos, neg };
  }

  it('prefers blur when the target is speckled into the background range', () => {
    // Heavy noise: per-pixel the two classes overlap badly, but their MEANS differ,
    // so smoothing should separate them.
    const { field, pos, neg } = speckled(64, 64, 150, 110, 55);
    const swept = fitWithBlurSweep(field, 64, 64, pos, neg, applyGaussianBlurGray);
    const unblurred = fitBand(...(() => {
      const { posHist, negHist } = sampleHistograms(field, pos, neg);
      return [posHist, negHist] as const;
    })());
    expect(swept.extraSigma).toBeGreaterThan(0);
    expect(swept.skill).toBeGreaterThan(unblurred.skill);
  });

  it('leaves a cleanly separated target unblurred', () => {
    const { field, pos, neg } = speckled(64, 64, 200, 60, 2);
    const swept = fitWithBlurSweep(field, 64, 64, pos, neg, applyGaussianBlurGray);
    expect(swept.extraSigma).toBe(0);
    expect(swept.skill).toBeGreaterThan(0.9);
  });

  it('does not mutate the caller’s field', () => {
    const { field, pos, neg } = speckled(32, 32, 150, 110, 40);
    const before = Float32Array.from(field);
    fitWithBlurSweep(field, 32, 32, pos, neg, applyGaussianBlurGray);
    expect(Array.from(field)).toEqual(Array.from(before));
  });

  it('returns not-ok when there is nothing to sample', () => {
    const field = new Float32Array(64);
    const empty = new Uint8Array(64);
    const swept = fitWithBlurSweep(field, 8, 8, empty, empty, applyGaussianBlurGray);
    expect(swept.ok).toBe(false);
  });
});

describe('combineSigma', () => {
  it('adds blurs in quadrature', () => {
    expect(combineSigma(0, 2)).toBeCloseTo(2, 6);
    expect(combineSigma(3, 4)).toBeCloseTo(5, 6);
  });
});

describe('base <-> display round trip (the sampler’s load-bearing assumption)', () => {
  // The fit runs on the field (BASE space) but the band is stored in DISPLAYED
  // space. If that conversion drifted, the applied band would quietly select
  // different pixels than the ones that were fitted.
  it('recovers the fitted base band through any display transform', () => {
    for (const [b, c, lo, hi, gamma] of [
      [0, 0, 0, 255, 1],
      [0.2, 30, 10, 240, 1],
      [-0.1, -20, 0, 255, 1.8],
      [0.05, 15, 40, 200, 0.6],
    ] as const) {
      const affine = displayAffineFor(b, c, lo, hi);
      for (const [baseLo, baseHi] of [[60, 90], [100, 180], [20, 240]] as const) {
        const dLo = baseToDisplay(baseLo, affine, gamma);
        const dHi = baseToDisplay(baseHi, affine, gamma);
        // Skip settings that clamp this band — the sampler detects and reports
        // that case rather than applying an unrepresentable band.
        if (Math.round(dHi) - Math.round(dLo) < 1) continue;
        // Skip saturating endpoints too: displayBandToBase deliberately reopens
        // an edge at 0/255 to ∓Infinity, which the sampler detects and refuses
        // rather than applying a silently wider band.
        if (Math.round(dLo) <= 0 || Math.round(dHi) >= 255) continue;
        const back = displayBandToBase(Math.round(dLo), Math.round(dHi), affine, gamma);
        // Integer storage costs up to half a displayed level. Its width in base
        // units varies along the range once gamma is involved, so measure it at
        // the endpoint rather than assuming the affine slope.
        const width = (d: number) =>
          Math.abs((displayToBase(d + 0.5, affine, gamma) ?? 0) - (displayToBase(d - 0.5, affine, gamma) ?? 0));
        const tol = Math.max(0.51, width(Math.round(dLo)), width(Math.round(dHi))) + 0.01;
        expect(Math.abs(back.lo - baseLo)).toBeLessThanOrEqual(tol);
        expect(Math.abs(back.hi - baseHi)).toBeLessThanOrEqual(tol);
      }
    }
  });

  it('collapses a narrow band under heavy negative contrast', () => {
    // Low contrast compresses the whole range, so neighbouring base values land
    // on the same displayed value and no band can distinguish them.
    const affine = displayAffineFor(0, -60, 0, 255);
    const dLo = baseToDisplay(128, affine, 1);
    const dHi = baseToDisplay(130, affine, 1);
    expect(Math.round(dHi) - Math.round(dLo)).toBeLessThan(1);
  });

  it('reopens a saturated endpoint, which is why the sampler must check', () => {
    // A bright fitted edge pushed past white comes back as +Infinity — applying
    // that band would select everything above the fitted range.
    const affine = displayAffineFor(0.5, 40, 0, 255);
    const dHi = Math.round(baseToDisplay(200, affine, 1));
    expect(dHi).toBeGreaterThanOrEqual(255);
    const back = displayBandToBase(100, dHi, affine, 1);
    expect(Number.isFinite(back.hi)).toBe(false);
  });
});

describe('ringWidthFor', () => {
  it('caps the ring so a large sample cannot explode the dilation cost', () => {
    // sqrt(area)/2 for a 800x800 region is 400 — 400 full passes over the grid,
    // which is what made a big lasso hang.
    const big = new Uint8Array(1000 * 1000).fill(1);
    expect(ringWidthFor(big)).toBe(MAX_RING_WIDTH);
  });

  it('still scales down for small samples', () => {
    const small = new Uint8Array(100 * 100);
    for (let y = 40; y < 60; y++) for (let x = 40; x < 60; x++) small[y * 100 + x] = 1;
    const w = ringWidthFor(small);
    expect(w).toBeGreaterThanOrEqual(4);
    expect(w).toBeLessThan(MAX_RING_WIDTH);
  });

  it('has a floor so a tiny sample still gets a usable ring', () => {
    const tiny = new Uint8Array(100);
    tiny[55] = 1;
    expect(ringWidthFor(tiny)).toBe(4);
  });
});
