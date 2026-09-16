import { describe, expect, it } from 'vitest';
import {
  denoiseMethodLabel,
  trainDenoiseBlockedReason,
  trainDenoisePayload,
  trainDenoiseSummary,
} from './trainDenoiseOption';

/** Shape of `capability.denoise.methods` entries this module cares about. */
const METHODS = [
  { method: 'none', label: 'None' },
  { method: 'gaussian', label: 'Gaussian' },
  { method: 'tv', label: 'Total variation' },
  { method: 'nlm', label: 'Non-local means' },
];

describe('trainDenoiseBlockedReason', () => {
  it('blocks "none" — there is no filter to apply', () => {
    expect(trainDenoiseBlockedReason('none')).toMatch(/pick a denoise filter/i);
  });

  it('blocks "model" — a learned denoiser is not a preprocessor', () => {
    expect(trainDenoiseBlockedReason('model')).toMatch(/learned denoiser/i);
  });

  it('allows any classical filter', () => {
    expect(trainDenoiseBlockedReason('tv')).toBeNull();
    expect(trainDenoiseBlockedReason('gaussian')).toBeNull();
    expect(trainDenoiseBlockedReason('median3d')).toBeNull();
  });
});

describe('trainDenoisePayload', () => {
  it('adds no key at all when the option is off', () => {
    // Not `{denoise: null}`: an un-denoised request must stay byte-identical
    // to what the app sent before this option existed.
    expect(trainDenoisePayload(false, { method: 'tv', strength: 0.6 })).toEqual({});
    expect(Object.keys(trainDenoisePayload(false, { method: 'tv', strength: 0.6 }))).toEqual([]);
  });

  it('adds no key when off even with denoising set to none', () => {
    expect(trainDenoisePayload(false, { method: 'none', strength: 0.5 })).toEqual({});
  });

  it('sends {method, strength} when on with a classical filter', () => {
    expect(trainDenoisePayload(true, { method: 'tv', strength: 0.6 }))
      .toEqual({ denoise: { method: 'tv', strength: 0.6 } });
  });

  it('sends nothing when on but the method is "none"', () => {
    expect(trainDenoisePayload(true, { method: 'none', strength: 0.5 })).toEqual({});
  });

  it('sends nothing when on but the method is "model"', () => {
    // The UI disables the checkbox in this case; the payload builder refuses
    // independently so a stale checked box can't reach the server.
    expect(trainDenoisePayload(true, { method: 'model', strength: 0.5 })).toEqual({});
  });

  it('strips preview-only fields the strict backend schema would reject', () => {
    const payload = trainDenoisePayload(true, {
      method: 'nlm', strength: 0.4, crop: 768, runId: 'run-123',
    } as { method: string; strength: number });
    expect(payload).toEqual({ denoise: { method: 'nlm', strength: 0.4 } });
    expect(Object.keys((payload as { denoise: object }).denoise).sort()).toEqual(['method', 'strength']);
  });

  it('passes the strength through unrounded at the extremes', () => {
    expect(trainDenoisePayload(true, { method: 'tv', strength: 0 }))
      .toEqual({ denoise: { method: 'tv', strength: 0 } });
    expect(trainDenoisePayload(true, { method: 'tv', strength: 1 }))
      .toEqual({ denoise: { method: 'tv', strength: 1 } });
  });
});

describe('trainDenoiseSummary', () => {
  it('names the filter and strength as a percentage', () => {
    expect(trainDenoiseSummary({ method: 'tv', strength: 0.6 }, METHODS)).toBe('Total variation, 60%');
  });

  it('rounds the percentage to a whole number', () => {
    expect(trainDenoiseSummary({ method: 'gaussian', strength: 0.333 }, METHODS)).toBe('Gaussian, 33%');
  });

  it('returns null for a method that cannot be trained on', () => {
    expect(trainDenoiseSummary({ method: 'none', strength: 0.5 }, METHODS)).toBeNull();
    expect(trainDenoiseSummary({ method: 'model', strength: 0.5 }, METHODS)).toBeNull();
  });

  it('falls back to the raw method id when the server did not describe it', () => {
    expect(trainDenoiseSummary({ method: 'wavelet', strength: 0.5 }, METHODS)).toBe('wavelet, 50%');
    expect(trainDenoiseSummary({ method: 'tv', strength: 0.5 }, [])).toBe('tv, 50%');
  });
});

describe('denoiseMethodLabel', () => {
  it('maps a known method id to its label', () => {
    expect(denoiseMethodLabel('nlm', METHODS)).toBe('Non-local means');
  });

  it('falls back to the id itself rather than rendering blank', () => {
    expect(denoiseMethodLabel('median3d', METHODS)).toBe('median3d');
    expect(denoiseMethodLabel('tv', [])).toBe('tv');
  });
});
