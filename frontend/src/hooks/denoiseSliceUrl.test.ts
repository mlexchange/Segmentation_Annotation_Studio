/**
 * The denoise half of `buildSliceUrl`'s contract.
 *
 * Two properties matter beyond "the params appear": an un-denoised request must
 * stay byte-identical to what this app sent before denoising existed (so it
 * keeps hitting the same browser-cache entries and the same server path), and
 * every denoise parameter must reach the URL — the URL *is* the cache key, both
 * for TanStack Query and for the canvas's tool caches, which key off the
 * resulting requestKey.
 */
import { describe, expect, it } from 'vitest';
import { buildSliceUrl, NO_DENOISE } from './useImageSlice';
import type { RenderOpts } from '@/stores/datasetStore';

const RENDER_OPTS: RenderOpts = {
  norm: 'global', scale: 'linear', vminPct: 1, vmaxPct: 99, cmap: 'gray',
};

const url = (denoise?: Parameters<typeof buildSliceUrl>[5]) =>
  buildSliceUrl('browse/dataset', 'tiled', 3, RENDER_OPTS, 'http://127.0.0.1:8010', denoise);

const params = (u: string) => new URLSearchParams(u.split('?')[1]);

describe('buildSliceUrl denoise params', () => {
  it('omits every denoise param when denoising is off', () => {
    const off = params(url(NO_DENOISE));

    expect(off.has('denoise_method')).toBe(false);
    expect(off.has('denoise_strength')).toBe(false);
    expect(off.has('denoise_crop')).toBe(false);
  });

  it('produces the exact same URL with denoise off as with no denoise argument at all', () => {
    // Guards the "strictly opt-in" property: existing cached slices stay valid.
    expect(url(NO_DENOISE)).toBe(url());
  });

  it('encodes method and strength when denoising is on', () => {
    const on = params(url({ method: 'tv', strength: 0.75 }));

    expect(on.get('denoise_method')).toBe('tv');
    expect(on.get('denoise_strength')).toBe('0.75');
  });

  it('omits the crop param unless a positive crop is requested', () => {
    expect(params(url({ method: 'tv', strength: 0.5 })).has('denoise_crop')).toBe(false);
    expect(params(url({ method: 'tv', strength: 0.5, crop: 0 })).has('denoise_crop')).toBe(false);
    expect(params(url({ method: 'tv', strength: 0.5, crop: 768 })).get('denoise_crop')).toBe('768');
  });

  it('still carries the render options and server uri alongside denoise params', () => {
    const both = params(url({ method: 'nlm', strength: 0.4 }));

    expect(both.get('source')).toBe('browse/dataset');
    expect(both.get('slice_index')).toBe('3');
    expect(both.get('norm')).toBe('global');
    expect(both.get('cmap')).toBe('gray');
    expect(both.get('server_uri')).toBe('http://127.0.0.1:8010');
    expect(both.get('denoise_method')).toBe('nlm');
  });

  it('gives a different URL for every distinct denoise setting', () => {
    // If any of these collided, a slider tweak would silently show a stale image
    // (and the wand would keep a stale field, since its cache keys on this URL).
    const variants = [
      url(NO_DENOISE),
      url({ method: 'tv', strength: 0.5 }),
      url({ method: 'tv', strength: 0.6 }),
      url({ method: 'nlm', strength: 0.5 }),
      url({ method: 'tv', strength: 0.5, crop: 768 }),
    ];

    expect(new Set(variants).size).toBe(variants.length);
  });

  describe('method: "model" (learned denoiser preview)', () => {
    it('encodes the run id instead of a strength', () => {
      const on = params(url({ method: 'model', strength: 0.5, runId: 'run-123' }));

      expect(on.get('denoise_method')).toBe('model');
      expect(on.get('denoise_run_id')).toBe('run-123');
      expect(on.has('denoise_strength')).toBe(false);
    });

    it('omits denoise_run_id when no run is selected yet', () => {
      expect(params(url({ method: 'model', strength: 0.5 })).has('denoise_run_id')).toBe(false);
    });

    it('still honors crop for a 1:1 preview of a slow model forward pass', () => {
      const on = params(url({ method: 'model', strength: 0.5, runId: 'run-123', crop: 768 }));
      expect(on.get('denoise_crop')).toBe('768');
    });

    it('is a distinct URL per run id', () => {
      const a = url({ method: 'model', strength: 0.5, runId: 'run-a' });
      const b = url({ method: 'model', strength: 0.5, runId: 'run-b' });
      expect(a).not.toBe(b);
    });
  });
});
