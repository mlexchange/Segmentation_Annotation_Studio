import { beforeEach, describe, expect, it } from 'vitest';
import { useDenoiseStore } from './denoiseStore';
import { NO_DENOISE } from '@/hooks/useImageSlice';
import { trainDenoisePayload } from '@/lib/trainDenoiseOption';

describe('denoiseStore', () => {
  beforeEach(() => {
    useDenoiseStore.getState().resetDenoise();
  });

  it('starts with denoising off', () => {
    expect(useDenoiseStore.getState().denoise).toEqual(NO_DENOISE);
    expect(useDenoiseStore.getState().denoise.method).toBe('none');
  });

  it('replaces the whole setting on set', () => {
    useDenoiseStore.getState().setDenoise({ method: 'tv', strength: 0.6, crop: 768 });
    expect(useDenoiseStore.getState().denoise).toEqual({ method: 'tv', strength: 0.6, crop: 768 });
  });

  // AnnotatePage's "reset display options" button delegates here, so this is
  // the behaviour that button now depends on.
  it('resets back to off, dropping any crop preview and run selection', () => {
    useDenoiseStore.getState().setDenoise({ method: 'model', strength: 0.6, crop: 768, runId: 'run-1' });
    useDenoiseStore.getState().resetDenoise();
    expect(useDenoiseStore.getState().denoise).toEqual(NO_DENOISE);
  });

  // The point of lifting this out of AnnotatePage: the Train tab reads the very
  // setting Annotate is displaying, so the two cannot describe different pixels.
  it('feeds the train payload builder whatever Annotate last set', () => {
    useDenoiseStore.getState().setDenoise({ method: 'nlm', strength: 0.4, crop: 768 });
    expect(trainDenoisePayload(true, useDenoiseStore.getState().denoise))
      .toEqual({ denoise: { method: 'nlm', strength: 0.4 } });

    useDenoiseStore.getState().resetDenoise();
    expect(trainDenoisePayload(true, useDenoiseStore.getState().denoise)).toEqual({});
  });
});
