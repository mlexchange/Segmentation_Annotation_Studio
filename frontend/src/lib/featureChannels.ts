/**
 * featureChannels — extra per-pixel descriptors for the Threshold Brush's fit.
 *
 * Phase 1 gates on intensity alone, which fails in two specific ways that show up
 * constantly on tomography:
 *
 *   1. *Overlapping intensity.* Two materials share a grey range but differ in
 *      texture or grain scale. No band on intensity can separate them.
 *   2. *Drifting intensity.* The same material is darker on one side of the slice
 *      (beam hardening, illumination), so one global band fits the middle and
 *      misses both ends.
 *
 * Each channel below answers one of those. They are all built from Gaussian
 * blurs, which are separable and O(n) — the same job an FFT band-pass would do
 * for this purpose, without the transform.
 *
 * Everything is pure and canvas-free so the maths can be tested directly.
 */
import { applyGaussianBlurGray } from '@/lib/blur';

/** Channels evaluated for every sampled pixel, in a fixed order. */
export const CHANNEL_NAMES = ['intensity', 'dogFine', 'dogCoarse', 'localStd', 'meanRatio'] as const;
export type ChannelName = (typeof CHANNEL_NAMES)[number];
export const CHANNEL_COUNT = CHANNEL_NAMES.length;

/** Scales (sigma, px) the texture channels are built at. */
const SIGMA_FINE = 1;
const SIGMA_MID = 2.5;
const SIGMA_COARSE = 6;

function blurred(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const out = Float32Array.from(src);
  applyGaussianBlurGray(out, w, h, sigma);
  return out;
}

/**
 * Build every channel for a field crop.
 *
 * Returns one Float32Array per channel, all `w*h` long and index-aligned with
 * `field`, so a pixel's feature vector is `channels.map(c => c[i])`.
 */
export function buildChannels(field: Float32Array, w: number, h: number): Float32Array[] {
  const fine = blurred(field, w, h, SIGMA_FINE);
  const mid = blurred(field, w, h, SIGMA_MID);
  const coarse = blurred(field, w, h, SIGMA_COARSE);

  const n = field.length;
  const dogFine = new Float32Array(n);
  const dogCoarse = new Float32Array(n);
  const localStd = new Float32Array(n);
  const meanRatio = new Float32Array(n);

  // Local variance via E[x²] − E[x]², both from the same Gaussian window.
  const sq = new Float32Array(n);
  for (let i = 0; i < n; i++) sq[i] = field[i] * field[i];
  const sqMean = blurred(sq, w, h, SIGMA_MID);

  for (let i = 0; i < n; i++) {
    // Difference-of-Gaussians: a band-pass. Responds to structure at a scale,
    // which is how two materials of equal brightness but different grain are
    // told apart (failure mode 1).
    dogFine[i] = fine[i] - mid[i];
    dogCoarse[i] = mid[i] - coarse[i];

    const variance = Math.max(0, sqMean[i] - mid[i] * mid[i]);
    localStd[i] = Math.sqrt(variance);

    // Brightness relative to the neighbourhood rather than absolute. Constant for
    // a material even where the illumination drifts (failure mode 2).
    const denom = Math.abs(coarse[i]) + 1e-3;
    meanRatio[i] = (field[i] - coarse[i]) / denom;
  }

  return [field, dogFine, dogCoarse, localStd, meanRatio];
}

/** Mean and standard deviation of `values` at the indices where `mask` is set. */
function moments(values: Float32Array, mask: Uint8Array): { mean: number; count: number } {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    sum += values[i];
    count++;
  }
  return { mean: count ? sum / count : 0, count };
}

/**
 * Weights projecting the channels onto the single axis that best separates the
 * two samples — a diagonal-covariance LDA (a.k.a. naive Fisher discriminant).
 *
 * Full LDA would invert the channel covariance matrix; with five channels and
 * possibly few sampled pixels that inverse is easily ill-conditioned, and a
 * silently unstable projection is worse than a slightly suboptimal one. Using
 * per-channel variance only is stable for any sample size and, once each channel
 * is standardised, loses little.
 *
 * Returns weights aligned with `channels`, plus the per-channel standardisation
 * needed to apply them to new pixels.
 */
export function fitProjection(
  channels: Float32Array[],
  pos: Uint8Array,
  neg: Uint8Array,
): { weights: number[]; centers: number[]; scales: number[] } | null {
  const weights: number[] = [];
  const centers: number[] = [];
  const scales: number[] = [];

  for (const channel of channels) {
    const p = moments(channel, pos);
    const n = moments(channel, neg);
    if (p.count === 0 || n.count === 0) return null;

    // Standardise on the pooled spread so channels with different units (grey
    // levels vs a ratio) contribute comparably.
    let ss = 0;
    let total = 0;
    for (const [mask, m] of [[pos, p.mean], [neg, n.mean]] as const) {
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        const d = channel[i] - m;
        ss += d * d;
        total++;
      }
    }
    const variance = total > 1 ? ss / (total - 1) : 0;
    const sd = Math.sqrt(variance);
    const center = (p.mean + n.mean) / 2;

    if (!(sd > 1e-6)) {
      // Constant channel — carries no information; drop it rather than dividing
      // by ~0 and letting numerical noise dominate the projection.
      weights.push(0);
      centers.push(center);
      scales.push(1);
      continue;
    }
    weights.push((p.mean - n.mean) / sd);
    centers.push(center);
    scales.push(sd);
  }

  const magnitude = Math.hypot(...weights);
  if (!(magnitude > 1e-9)) return null; // no channel separates anything
  return { weights: weights.map((w) => w / magnitude), centers, scales };
}

/**
 * Project every pixel onto the fitted axis and rescale to the 0–255 range the
 * band fitter and the overlay already speak — so Phase 2 changes what the number
 * *means* without changing any of the machinery that consumes it.
 */
export function projectToScore(
  channels: Float32Array[],
  projection: { weights: number[]; centers: number[]; scales: number[] },
): Float32Array {
  const n = channels[0].length;
  const raw = new Float32Array(n);
  const { weights, centers, scales } = projection;

  for (let c = 0; c < channels.length; c++) {
    const w = weights[c];
    if (w === 0) continue;
    const channel = channels[c];
    const center = centers[c];
    const scale = scales[c] || 1;
    for (let i = 0; i < n; i++) raw[i] += w * ((channel[i] - center) / scale);
  }

  // Robust rescale: 1st–99th percentile onto 0–255, so a few extreme pixels can't
  // squash everything else into one bin.
  const sorted = Float32Array.from(raw).sort();
  const lo = sorted[Math.floor(0.01 * (sorted.length - 1))];
  const hi = sorted[Math.floor(0.99 * (sorted.length - 1))];
  const span = hi - lo;
  const out = new Float32Array(n);
  if (!(span > 1e-9)) return out;
  for (let i = 0; i < n; i++) {
    const v = ((raw[i] - lo) / span) * 255;
    out[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return out;
}
