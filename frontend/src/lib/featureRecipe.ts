/**
 * Feature recipe helpers — describe shelf-model required features.
 */
import type { FeatureParams } from '@/hooks/useFeatureChannels';
import { DEFAULT_FEATURE_PARAMS } from '@/hooks/useFeatureChannels';

/** Convert a shelf feature_recipe dict into FeatureParams. */
export function featureParamsFromRecipe(
  recipe: Record<string, unknown> | null | undefined,
): FeatureParams {
  if (!recipe) return { ...DEFAULT_FEATURE_PARAMS };
  return {
    sigmaMin: Number(recipe.sigma_min ?? DEFAULT_FEATURE_PARAMS.sigmaMin),
    sigmaMax: Number(recipe.sigma_max ?? DEFAULT_FEATURE_PARAMS.sigmaMax),
    intensity: recipe.intensity !== false,
    edges: recipe.edges !== false,
    texture: recipe.texture !== false,
    clahe: recipe.clahe !== false,
    includeSam: Boolean(recipe.include_sam ?? recipe.includeSam),
  };
}

/** Short human-readable feature list for Connect/Browse/Train. */
export function formatFeatureRecipe(
  recipe: Record<string, unknown> | null | undefined,
): string {
  if (!recipe) return '—';
  const lo = recipe.sigma_min ?? '?';
  const hi = recipe.sigma_max ?? '?';
  const bits: string[] = [`σ ${lo}–${hi}`];
  if (recipe.intensity !== false) bits.push('intensity');
  if (recipe.edges !== false) bits.push('edges');
  if (recipe.texture !== false) bits.push('texture');
  if (recipe.clahe) bits.push('CLAHE');
  if (recipe.include_sam || recipe.includeSam) bits.push('SlimSAM');
  return bits.join(' · ');
}
