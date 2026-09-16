import { describe, expect, it } from 'vitest';
import { denoiserScopeBlockedReason, resolveDenoiserScopeIndices } from './denoiserTrainScope';

describe('resolveDenoiserScopeIndices', () => {
  it('returns just the current slice for scope "current"', () => {
    expect(resolveDenoiserScopeIndices('current', 5, 0, 0, 10)).toEqual([5]);
  });

  it('returns [] for "current" when the current slice is out of range', () => {
    expect(resolveDenoiserScopeIndices('current', 20, 0, 0, 10)).toEqual([]);
    expect(resolveDenoiserScopeIndices('current', -1, 0, 0, 10)).toEqual([]);
  });

  it('returns every index for scope "all"', () => {
    expect(resolveDenoiserScopeIndices('all', 0, 0, 0, 4)).toEqual([0, 1, 2, 3]);
  });

  it('returns [] for an empty volume regardless of scope', () => {
    expect(resolveDenoiserScopeIndices('all', 0, 0, 0, 0)).toEqual([]);
    expect(resolveDenoiserScopeIndices('current', 0, 0, 0, 0)).toEqual([]);
    expect(resolveDenoiserScopeIndices('range', 0, 0, 0, 0)).toEqual([]);
  });

  it('returns the inclusive range for scope "range"', () => {
    expect(resolveDenoiserScopeIndices('range', 0, 2, 5, 10)).toEqual([2, 3, 4, 5]);
  });

  it('normalizes a reversed range (end typed before start)', () => {
    expect(resolveDenoiserScopeIndices('range', 0, 5, 2, 10)).toEqual([2, 3, 4, 5]);
  });

  it('clamps a range to [0, nSlices)', () => {
    expect(resolveDenoiserScopeIndices('range', 0, -3, 2, 5)).toEqual([0, 1, 2]);
    expect(resolveDenoiserScopeIndices('range', 0, 3, 99, 5)).toEqual([3, 4]);
  });

  it('handles a single-slice range', () => {
    expect(resolveDenoiserScopeIndices('range', 0, 3, 3, 10)).toEqual([3]);
  });
});

describe('denoiserScopeBlockedReason', () => {
  it('blocks an empty scope regardless of scheme', () => {
    // Enumerates EVERY scheme on purpose: this used to cover only two of them,
    // so a newly-added scheme could silently skip the check.
    expect(denoiserScopeBlockedReason([], 'n2v')).toMatch(/no slices/i);
    expect(denoiserScopeBlockedReason([], 'n2n')).toMatch(/no slices/i);
    expect(denoiserScopeBlockedReason([], 'ae')).toMatch(/no slices/i);
  });

  it('blocks Noise2Noise on a single slice', () => {
    expect(denoiserScopeBlockedReason([4], 'n2n')).toMatch(/noise2noise/i);
  });

  it('allows Noise2Void on a single slice', () => {
    expect(denoiserScopeBlockedReason([4], 'n2v')).toBeNull();
  });

  it('allows the autoencoder on a single slice', () => {
    // It reconstructs one slice through a bottleneck — unlike Noise2Noise it
    // needs no second slice to pair with.
    expect(denoiserScopeBlockedReason([4], 'ae')).toBeNull();
    expect(denoiserScopeBlockedReason([0, 1, 2], 'ae')).toBeNull();
  });

  it('allows Noise2Noise once there are at least 2 slices', () => {
    expect(denoiserScopeBlockedReason([4, 5], 'n2n')).toBeNull();
    expect(denoiserScopeBlockedReason([1, 2, 3], 'n2n')).toBeNull();
  });
});
