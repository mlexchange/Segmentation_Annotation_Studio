import { describe, it, expect } from 'vitest';
import {
  buildChannels,
  fitProjection,
  projectToScore,
  CHANNEL_NAMES,
  CHANNEL_COUNT,
} from './featureChannels';
import { fitBand, sampleHistograms } from './thresholdFit';

const W = 64;
const H = 64;

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Inside the central square is the target; outside is background. */
function masks(): { pos: Uint8Array; neg: Uint8Array; inside: (i: number) => boolean } {
  const pos = new Uint8Array(W * H);
  const neg = new Uint8Array(W * H);
  const inside = (i: number) => {
    const x = i % W;
    const y = (i / W) | 0;
    return x >= 16 && x < 48 && y >= 16 && y < 48;
  };
  for (let i = 0; i < W * H; i++) (inside(i) ? pos : neg)[i] = 1;
  return { pos, neg, inside };
}

/** Best Dice achievable by thresholding a single channel directly. */
function bandSkill(values: Float32Array, pos: Uint8Array, neg: Uint8Array): number {
  const { posHist, negHist } = sampleHistograms(values, pos, neg);
  return fitBand(posHist, negHist).skill;
}

describe('buildChannels', () => {
  it('returns one aligned array per named channel', () => {
    const field = new Float32Array(W * H).fill(100);
    const channels = buildChannels(field, W, H);
    expect(channels).toHaveLength(CHANNEL_COUNT);
    expect(CHANNEL_NAMES).toHaveLength(CHANNEL_COUNT);
    for (const c of channels) expect(c).toHaveLength(W * H);
  });

  it('keeps intensity as the first channel, unmodified', () => {
    const field = new Float32Array(W * H);
    for (let i = 0; i < field.length; i++) field[i] = i % 200;
    const [intensity] = buildChannels(field, W, H);
    expect(Array.from(intensity)).toEqual(Array.from(field));
  });

  it('gives a flat field no texture response', () => {
    const field = new Float32Array(W * H).fill(77);
    const [, dogFine, dogCoarse, localStd] = buildChannels(field, W, H);
    for (let i = 0; i < field.length; i++) {
      expect(Math.abs(dogFine[i])).toBeLessThan(1e-3);
      expect(Math.abs(dogCoarse[i])).toBeLessThan(1e-3);
      expect(localStd[i]).toBeLessThan(1e-2);
    }
  });

  it('responds to fine texture where intensity is identical', () => {
    // Both regions average 128; only the target is speckled.
    const field = new Float32Array(W * H);
    const { inside } = masks();
    const rnd = mulberry32(3);
    for (let i = 0; i < field.length; i++) {
      field[i] = inside(i) ? (rnd() > 0.5 ? 168 : 88) : 128;
    }
    const [, , , localStd] = buildChannels(field, W, H);
    let insideStd = 0, insideN = 0, outsideStd = 0, outsideN = 0;
    for (let i = 0; i < field.length; i++) {
      if (inside(i)) { insideStd += localStd[i]; insideN++; }
      else { outsideStd += localStd[i]; outsideN++; }
    }
    expect(insideStd / insideN).toBeGreaterThan((outsideStd / outsideN) * 5);
  });
});

describe('fitProjection + projectToScore', () => {
  it('separates equal-brightness regions that differ only in texture', () => {
    // The case Phase 1 cannot solve: identical means, different grain.
    const field = new Float32Array(W * H);
    const { pos, neg, inside } = masks();
    const rnd = mulberry32(11);
    for (let i = 0; i < field.length; i++) {
      field[i] = inside(i) ? (rnd() > 0.5 ? 170 : 86) : 128 + (rnd() - 0.5) * 2;
    }

    const intensitySkill = bandSkill(field, pos, neg);
    const channels = buildChannels(field, W, H);
    const projection = fitProjection(channels, pos, neg);
    expect(projection).not.toBeNull();
    const score = projectToScore(channels, projection!);
    const projectedSkill = bandSkill(score, pos, neg);

    // A bimodal target against a mid-grey background IS partly reachable by an
    // intensity band (it can grab one lobe), so the claim is not "intensity is
    // useless" — it is that the projection is decisively better.
    expect(intensitySkill).toBeLessThan(0.6);
    expect(projectedSkill).toBeGreaterThan(0.9);
    expect(projectedSkill).toBeGreaterThan(intensitySkill + 0.4);
  });

  it('leans on the texture channel when that is what distinguishes the regions', () => {
    // Same fixture as above: equal means, different grain. The fitted weights
    // should concentrate on localStd — evidence the projection is separating for
    // the right reason rather than getting lucky.
    const field = new Float32Array(W * H);
    const { pos, neg, inside } = masks();
    const rnd = mulberry32(11);
    for (let i = 0; i < field.length; i++) {
      field[i] = inside(i) ? (rnd() > 0.5 ? 170 : 86) : 128 + (rnd() - 0.5) * 2;
    }
    const projection = fitProjection(buildChannels(field, W, H), pos, neg)!;
    const stdIndex = CHANNEL_NAMES.indexOf('localStd');
    const dominant = projection.weights
      .map((w, i) => ({ w: Math.abs(w), i }))
      .sort((a, b) => b.w - a.w)[0].i;
    expect(dominant).toBe(stdIndex);
  });

  it('separates a material whose brightness drifts across the image', () => {
    // Target is always +25 above its surroundings, but absolute level ramps
    // across x — so no single intensity band covers it.
    const field = new Float32Array(W * H);
    const { pos, neg, inside } = masks();
    for (let i = 0; i < field.length; i++) {
      const x = i % W;
      const ramp = 40 + (x / W) * 150;
      field[i] = inside(i) ? ramp + 25 : ramp;
    }

    const intensitySkill = bandSkill(field, pos, neg);
    const channels = buildChannels(field, W, H);
    const projection = fitProjection(channels, pos, neg)!;
    const projectedSkill = bandSkill(projectToScore(channels, projection), pos, neg);

    expect(projectedSkill).toBeGreaterThan(intensitySkill);
    expect(projectedSkill).toBeGreaterThan(0.5);
  });

  it('does not do worse than intensity on a cleanly separable target', () => {
    const field = new Float32Array(W * H);
    const { pos, neg, inside } = masks();
    for (let i = 0; i < field.length; i++) field[i] = inside(i) ? 200 : 60;

    const channels = buildChannels(field, W, H);
    const projection = fitProjection(channels, pos, neg)!;
    const projectedSkill = bandSkill(projectToScore(channels, projection), pos, neg);
    expect(projectedSkill).toBeGreaterThan(0.9);
  });

  it('returns normalised weights', () => {
    const field = new Float32Array(W * H);
    const { pos, neg, inside } = masks();
    for (let i = 0; i < field.length; i++) field[i] = inside(i) ? 180 : 70;
    const projection = fitProjection(buildChannels(field, W, H), pos, neg)!;
    expect(Math.hypot(...projection.weights)).toBeCloseTo(1, 6);
  });

  it('returns null when a sample is empty', () => {
    const field = new Float32Array(W * H).fill(120);
    const channels = buildChannels(field, W, H);
    const empty = new Uint8Array(W * H);
    const full = new Uint8Array(W * H).fill(1);
    expect(fitProjection(channels, empty, full)).toBeNull();
    expect(fitProjection(channels, full, empty)).toBeNull();
  });

  it('returns null when nothing distinguishes the samples', () => {
    // Identical constant field: every channel is constant, so no axis separates.
    const field = new Float32Array(W * H).fill(120);
    const { pos, neg } = masks();
    expect(fitProjection(buildChannels(field, W, H), pos, neg)).toBeNull();
  });

  it('produces scores inside the 0–255 range the band fitter expects', () => {
    const field = new Float32Array(W * H);
    const { pos, neg, inside } = masks();
    const rnd = mulberry32(5);
    for (let i = 0; i < field.length; i++) field[i] = (inside(i) ? 150 : 90) + (rnd() - 0.5) * 40;
    const channels = buildChannels(field, W, H);
    const score = projectToScore(channels, fitProjection(channels, pos, neg)!);
    for (let i = 0; i < score.length; i++) {
      expect(score[i]).toBeGreaterThanOrEqual(0);
      expect(score[i]).toBeLessThanOrEqual(255);
    }
  });
});
