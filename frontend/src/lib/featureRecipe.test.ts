import { describe, expect, it } from 'vitest';
import { featureParamsFromRecipe, formatFeatureRecipe } from './featureRecipe';

describe('formatFeatureRecipe', () => {
  it('summarizes sigma and toggles', () => {
    expect(
      formatFeatureRecipe({
        sigma_min: 1,
        sigma_max: 8,
        intensity: true,
        edges: true,
        texture: false,
        clahe: true,
        include_sam: true,
      }),
    ).toBe('σ 1–8 · intensity · edges · CLAHE · SlimSAM');
  });
});

describe('featureParamsFromRecipe', () => {
  it('maps snake_case recipe into FeatureParams', () => {
    const p = featureParamsFromRecipe({
      sigma_min: 2,
      sigma_max: 16,
      intensity: true,
      edges: false,
      texture: true,
      clahe: false,
      include_sam: true,
    });
    expect(p.sigmaMin).toBe(2);
    expect(p.sigmaMax).toBe(16);
    expect(p.edges).toBe(false);
    expect(p.includeSam).toBe(true);
  });
});
