import { describe, expect, it } from 'vitest';
import { pathForStage, stageFromPath } from './annotateStage';

describe('annotateStage', () => {
  it('maps hub paths to stages', () => {
    expect(stageFromPath('/preprocess')).toBe('preprocess');
    expect(stageFromPath('/draw')).toBe('draw');
    expect(stageFromPath('/train')).toBe('train');
    expect(stageFromPath('/cleanup')).toBe('train');
    expect(stageFromPath('/annotate')).toBe('preprocess');
  });

  it('returns hub paths for stages', () => {
    expect(pathForStage('preprocess')).toBe('/preprocess');
    expect(pathForStage('train')).toBe('/train');
  });
});
