/**
 * `defaultTarget` pre-fills the bake modal's editable target-path field. It
 * mirrors `denoise_bake.default_target_path` on the backend so the suggested
 * name matches what the server would pick if the user leaves the field alone —
 * this only pins the common cases, since the field stays user-editable either way.
 */
import { describe, expect, it } from 'vitest';
import { defaultTarget } from './DenoiseBakeModal';

describe('defaultTarget', () => {
  it('appends _denoised to a single-segment sample path', () => {
    expect(defaultTarget('browse/dataset')).toBe('browse/dataset_denoised');
  });

  it('appends _denoised only to the last segment of a nested path', () => {
    expect(defaultTarget('browse/expt/scan1')).toBe('browse/expt/scan1_denoised');
  });

  it('ignores leading and trailing slashes', () => {
    expect(defaultTarget('/browse/dataset/')).toBe('browse/dataset_denoised');
  });

  it('returns an empty string for a blank source rather than "_denoised"', () => {
    expect(defaultTarget('   ')).toBe('');
    expect(defaultTarget('')).toBe('');
  });
});
