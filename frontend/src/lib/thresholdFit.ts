/**
 * thresholdFit — derive a Threshold Brush intensity band from a lassoed example.
 *
 * The user lassos one representative feature; everything inside is a positive
 * example and a ring just outside it supplies negatives. The question "which band
 * best fills that region and not its neighbours?" is then a supervised threshold
 * fit with an *exact* answer, not something needing a search heuristic:
 *
 *   - histogram both sets into 256 bins,
 *   - prefix-sum them, so any band's counts are two subtractions,
 *   - evaluate every (lo, hi) pair and keep the best.
 *
 * That is 256² ≈ 33k ordered pairs of integer work — well under a millisecond —
 * so the returned band is the true optimum over all bands, with no tuning
 * constants and no iteration to converge.
 *
 * Deliberately pure and canvas-free: the correctness of this file is what makes
 * the feature trustworthy, and it can be tested directly.
 */

/** How well a band matched the example, and what it selected. */
export interface BandFit {
  /** Inclusive bin bounds of the best band, in the field's own 0–255 space. */
  lo: number;
  hi: number;
  /** Dice overlap with the positive sample, 0–1. */
  dice: number;
  /**
   * Dice of the trivial "select everything" band — the score achievable with no
   * separation at all. It depends on how big the ring is relative to the sample
   * (equal sizes give 0.67; a 4x larger ring gives 0.33), which is exactly why
   * raw Dice cannot be graded on its own.
   */
  baselineDice: number;
  /**
   * How far the fit closed the gap between that baseline and a perfect match:
   * `(dice - baseline) / (1 - baseline)`, clamped to 0–1. This is the number to
   * judge quality by — 0 means the band did no better than selecting everything,
   * however flattering its Dice looks.
   */
  skill: number;
  /** Fraction of the positive sample the band captures (recall). */
  coverage: number;
  /** Fraction of selected pixels that came from the negative ring. */
  leakage: number;
  /** False when there was nothing to fit (empty sample). */
  ok: boolean;
}

const BINS = 256;

/** An empty/degenerate result — a band that selects nothing. */
const NO_FIT: BandFit = {
  lo: 0, hi: 0, dice: 0, baselineDice: 0, skill: 0, coverage: 0, leakage: 0, ok: false,
};

/** Running totals of `hist`, where `cum[i]` counts bins 0..i inclusive. */
function prefixSum(hist: ArrayLike<number>): Float64Array {
  const cum = new Float64Array(BINS);
  let running = 0;
  for (let i = 0; i < BINS; i++) {
    running += hist[i] ?? 0;
    cum[i] = running;
  }
  return cum;
}

/** Count within `[lo,hi]` inclusive, from a prefix sum. */
function rangeCount(cum: Float64Array, lo: number, hi: number): number {
  return cum[hi] - (lo > 0 ? cum[lo - 1] : 0);
}

/**
 * Best intensity band separating `posHist` (the lassoed feature) from `negHist`
 * (its surroundings), maximising Dice overlap with the positives.
 *
 * Dice — 2TP / (2TP + FP + FN) — is used rather than raw accuracy because the
 * negative ring usually outnumbers the positives, and accuracy would then be
 * maximised by selecting almost nothing.
 *
 * @param posHist 256-bin histogram of values inside the lasso.
 * @param negHist 256-bin histogram of values in the surrounding ring.
 */
export function fitBand(posHist: ArrayLike<number>, negHist: ArrayLike<number>): BandFit {
  const cumPos = prefixSum(posHist);
  const cumNeg = prefixSum(negHist);
  const totalPos = cumPos[BINS - 1];
  const totalNeg = cumNeg[BINS - 1];
  if (totalPos <= 0) return { ...NO_FIT };

  // The score a band gets for free by selecting the entire range. Quality has to
  // be measured against this, not against zero.
  const baselineDice = (2 * totalPos) / (2 * totalPos + totalNeg);

  let best: BandFit = { ...NO_FIT, ok: true };
  let bestDice = -1;

  for (let lo = 0; lo < BINS; lo++) {
    // Widening `hi` only ever adds counts, so both terms grow monotonically —
    // but Dice itself is not monotonic, so every hi must still be evaluated.
    for (let hi = lo; hi < BINS; hi++) {
      const tp = rangeCount(cumPos, lo, hi);
      if (tp === 0) continue; // selects none of the sample — cannot be best
      const fp = rangeCount(cumNeg, lo, hi);
      const fn = totalPos - tp;
      const dice = (2 * tp) / (2 * tp + fp + fn);
      if (dice > bestDice) {
        bestDice = dice;
        best = {
          lo,
          hi,
          dice,
          baselineDice,
          skill: baselineDice < 1 ? Math.max(0, (dice - baselineDice) / (1 - baselineDice)) : 0,
          coverage: tp / totalPos,
          leakage: tp + fp > 0 ? fp / (tp + fp) : 0,
          ok: true,
        };
      }
    }
  }

  return bestDice < 0 ? { ...NO_FIT } : best;
}

/**
 * Plain-language reading of a fit, so a weak result is visible as weak rather
 * than being silently applied and discovered later while painting.
 */
export function describeFit(fit: BandFit): { label: string; quality: 'good' | 'fair' | 'weak' } {
  if (!fit.ok) return { label: 'Nothing sampled', quality: 'weak' };
  // Graded on `skill`, not `dice`: a feature that cannot be separated at all still
  // scores 0.67 Dice against an equal-sized ring, and calling that "fair" would be
  // exactly the false confidence this readout exists to prevent.
  if (fit.skill >= 0.7) {
    return { label: 'Strong — this band matches your sample closely', quality: 'good' };
  }
  if (fit.skill >= 0.4) {
    return { label: 'Fair — expect some over- or under-fill', quality: 'fair' };
  }
  return {
    label: 'Weak — intensity alone barely separates this from its surroundings',
    quality: 'weak',
  };
}

/**
 * Histogram a field over two masks at once.
 *
 * `field` values are the 0–255 greys the Threshold Brush gates on; `pos` and
 * `neg` are same-length 0/1 masks over the same grid.
 */
export function sampleHistograms(
  field: ArrayLike<number>,
  pos: ArrayLike<number>,
  neg: ArrayLike<number>,
): { posHist: Float64Array; negHist: Float64Array; posCount: number; negCount: number } {
  const posHist = new Float64Array(BINS);
  const negHist = new Float64Array(BINS);
  let posCount = 0;
  let negCount = 0;
  const n = Math.min(field.length, pos.length, neg.length);
  for (let i = 0; i < n; i++) {
    const inPos = pos[i];
    const inNeg = neg[i];
    if (!inPos && !inNeg) continue;
    let v = Math.round(field[i]);
    if (!Number.isFinite(v)) continue;
    v = v < 0 ? 0 : v > 255 ? 255 : v;
    if (inPos) { posHist[v]++; posCount++; }
    else { negHist[v]++; negCount++; }
  }
  return { posHist, negHist, posCount, negCount };
}

/** Sigmas the sampler tries, in addition to whatever blur is already applied. */
export const BLUR_CANDIDATES: readonly number[] = [0, 0.5, 1, 1.5, 2, 3];

/** Result of a sweep: the winning band and the extra blur it needed. */
export interface SweepResult extends BandFit {
  /** Additional sigma applied on top of the current setting (0 = none). */
  extraSigma: number;
}

/**
 * Fit a band across several candidate blur levels and keep the best.
 *
 * Blur is swept because — unlike the display sliders — it genuinely changes what
 * is separable: it is a spatial operation, so it can pull a speckled feature's
 * values together and away from its surroundings. Sigmas are *additional* blur on
 * top of whatever the user has already set, since the field handed in is the one
 * the brush currently uses.
 *
 * @param field  Crop of the threshold field (grays), row-major `w × h`.
 * @param pos    Positive mask over the same crop.
 * @param neg    Negative (ring) mask over the same crop.
 * @param blurFn Injected gray blur (`blur.applyGaussianBlurGray`) so this module
 *   stays dependency-free and directly testable.
 */
export function fitWithBlurSweep(
  field: Float32Array,
  w: number,
  h: number,
  pos: Uint8Array,
  neg: Uint8Array,
  blurFn: (data: Float32Array, w: number, h: number, sigma: number) => void,
  sigmas: readonly number[] = BLUR_CANDIDATES,
): SweepResult {
  let best: SweepResult = { ...fitBand(new Float64Array(BINS), new Float64Array(BINS)), extraSigma: 0 };
  let bestSkill = -1;

  for (const sigma of sigmas) {
    const candidate = sigma > 0 ? Float32Array.from(field) : field;
    if (sigma > 0) blurFn(candidate, w, h, sigma);
    const { posHist, negHist } = sampleHistograms(candidate, pos, neg);
    const fit = fitBand(posHist, negHist);
    if (!fit.ok) continue;
    // Compare on skill, not Dice: Dice shifts with class balance, and blurring
    // does not change the balance — but skill is the number the UI grades on, so
    // optimising anything else could pick a sigma the readout then calls worse.
    if (fit.skill > bestSkill) {
      bestSkill = fit.skill;
      best = { ...fit, extraSigma: sigma };
    }
  }
  return best;
}

/** Combine two Gaussian blurs: sigmas add in quadrature. */
export function combineSigma(current: number, extra: number): number {
  return Math.sqrt(current * current + extra * extra);
}

/** Widest background ring, in grid cells. See `ringWidthFor`. */
export const MAX_RING_WIDTH = 24;

/**
 * How wide a background ring to grow around a sample.
 *
 * Scales with the region so a small lasso gets a proportionate neighbourhood,
 * but is **capped**: the ring is meant to be the immediate surroundings, and
 * dilation costs one pass per cell of width. Uncapped, lassoing an 800x800
 * region asked for a 400-cell ring — 400 passes over the grid, which is why a
 * big sample used to hang. A capped ring is both faster and more correct: what
 * a feature touches is a local question.
 */
export function ringWidthFor(mask: ArrayLike<number>): number {
  let area = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) area++;
  return Math.min(MAX_RING_WIDTH, Math.max(4, Math.round(Math.sqrt(area) / 2)));
}

/**
 * Ring of background around `mask`: `dilate(mask, iters) \ mask`.
 *
 * Fitting against the immediate neighbourhood rather than the whole slice is
 * what makes the result mean "fill this and not what it touches" — using the
 * entire image as negatives would penalise the very lookalikes elsewhere that
 * the user wants highlighted.
 *
 * @param dilateFn Injected (`morphology.dilate`) to keep this module canvas- and
 *   dependency-free for testing.
 */
export function backgroundRing(
  mask: Uint8Array,
  gw: number,
  gh: number,
  dilateFn: (m: Uint8Array, gw: number, gh: number, iters: number) => Uint8Array,
  iters?: number,
): Uint8Array {
  const width = iters ?? ringWidthFor(mask);
  const grown = dilateFn(mask, gw, gh, width);
  const ring = new Uint8Array(mask.length);
  for (let i = 0; i < ring.length; i++) ring[i] = grown[i] && !mask[i] ? 1 : 0;
  return ring;
}
