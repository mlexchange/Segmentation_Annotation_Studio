/**
 * denoiserTrainScope — resolves the Learned Denoiser panel's current/range/all
 * scope selector into the concrete slice indices to train on, and validates
 * the result before a training job is submitted.
 *
 * Mirrors ApplyModelPanel's fine-tune-scope selector (see its `trainScope`/
 * `trainSliceIndices`) as closely as possible, kept here as a standalone pure
 * function (rather than inline component state, as ApplyModelPanel and
 * InferencePanel each do) so it's independently testable and its validation
 * message can't drift between call sites.
 *
 * Denoiser training needs no annotations at all — Noise2Noise and Noise2Void
 * are both self-supervised on the raw slice pixels themselves — so the
 * validation here is about the SCOPE being usable at all, not about
 * annotation coverage. Noise2Noise specifically needs to pair up two
 * independent-noise realizations of (approximately) the same structure, so a
 * scope of a single slice can't train it; Noise2Void and the autoencoder train
 * on single slices fine (the first masks pixels within one slice, the second
 * reconstructs the slice through a bottleneck).
 */

export type DenoiserTrainScope = 'current' | 'range' | 'all';
export type DenoiserScheme = 'n2n' | 'n2v' | 'ae';

/**
 * Concrete, ascending, deduplicated slice indices for `scope`, clamped to
 * `[0, nSlices)`. Returns `[]` when the scope resolves to nothing usable
 * (e.g. `currentSlice` outside `[0, nSlices)`, or an empty volume).
 */
export function resolveDenoiserScopeIndices(
  scope: DenoiserTrainScope,
  currentSlice: number,
  rangeStart: number,
  rangeEnd: number,
  nSlices: number,
): number[] {
  if (nSlices <= 0) return [];
  if (scope === 'current') {
    return currentSlice >= 0 && currentSlice < nSlices ? [currentSlice] : [];
  }
  if (scope === 'range') {
    const lo = Math.max(0, Math.min(rangeStart, rangeEnd));
    const hi = Math.min(nSlices - 1, Math.max(rangeStart, rangeEnd));
    if (hi < lo) return [];
    return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
  }
  return Array.from({ length: nSlices }, (_, i) => i);
}

/**
 * Why `indices` can't be used to train `scheme`, phrased for display next to
 * the Train button — or null when the scope is ready to submit.
 */
export function denoiserScopeBlockedReason(
  indices: number[],
  scheme: DenoiserScheme,
): string | null {
  if (indices.length === 0) {
    return 'No slices selected — choose a different scope.';
  }
  if (scheme === 'n2n' && indices.length < 2) {
    return 'Noise2Noise needs at least 2 slices to pair up — choose a range or all slices, or switch to Noise2Void.';
  }
  return null;
}
