import { describe, expect, it } from 'vitest';
import { DEFAULT_FEATURE_PARAMS } from './useFeatureChannels';

describe('DEFAULT_FEATURE_PARAMS', () => {
  it('matches locked plan defaults including SlimSAM', () => {
    expect(DEFAULT_FEATURE_PARAMS).toEqual({
      sigmaMin: 1,
      sigmaMax: 8,
      intensity: true,
      edges: true,
      texture: true,
      clahe: true,
      includeSam: true,
    });
  });
});
