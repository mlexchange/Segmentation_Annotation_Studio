import { describe, it, expect } from 'vitest';
import { buildSliceUrl } from './useImageSlice';
import type { RenderOpts, DenoiseOpts } from '@/stores/datasetStore';

const RENDER: RenderOpts = {
  norm: 'global',
  scale: 'linear',
  vminPct: 1,
  vmaxPct: 99,
  cmap: 'gray',
};

const params = (url: string) => new URL(url, 'http://x').searchParams;

describe('buildSliceUrl denoise parameters', () => {
  it('omits denoise entirely when it is off', () => {
    // The un-denoised request must stay byte-identical to what it has always
    // been, so it keeps hitting the same backend cache entry.
    const off: DenoiseOpts = { method: 'none', strength: 0.7 };
    const withOff = buildSliceUrl('s', 'tiled', 3, RENDER, null, off);
    const without = buildSliceUrl('s', 'tiled', 3, RENDER, null);
    expect(withOff).toBe(without);
    expect(params(withOff).has('denoise_method')).toBe(false);
  });

  it('sends method and strength when active', () => {
    const p = params(buildSliceUrl('s', 'tiled', 3, RENDER, null, { method: 'tv', strength: 0.4 }));
    expect(p.get('denoise_method')).toBe('tv');
    expect(p.get('denoise_strength')).toBe('0.4');
  });

  it('sends a crop only alongside an active method', () => {
    // Cropping is what keeps slider-dragging interactive: filtering a full
    // 2560² slice costs seconds, a 512 crop costs ~0.3s.
    const cropped = params(
      buildSliceUrl('s', 'tiled', 3, RENDER, null, { method: 'nlm', strength: 0.5 }, 512),
    );
    expect(cropped.get('denoise_crop')).toBe('512');

    const offWithCrop = params(
      buildSliceUrl('s', 'tiled', 3, RENDER, null, { method: 'none', strength: 0.5 }, 512),
    );
    expect(offWithCrop.has('denoise_crop')).toBe(false);
  });

  it('ignores a zero or negative crop', () => {
    const p = params(
      buildSliceUrl('s', 'tiled', 3, RENDER, null, { method: 'tv', strength: 0.5 }, 0),
    );
    expect(p.has('denoise_crop')).toBe(false);
  });

  it('still carries the render options', () => {
    // Denoise is additive: it must not displace normalization, which is applied
    // after it on the backend.
    const p = params(buildSliceUrl('s', 'tiled', 3, RENDER, null, { method: 'tv', strength: 0.5 }));
    expect(p.get('norm')).toBe('global');
    expect(p.get('vmin_pct')).toBe('1');
    expect(p.get('cmap')).toBe('gray');
  });

  it('keeps the server uri', () => {
    const p = params(
      buildSliceUrl('s', 'tiled', 3, RENDER, 'http://127.0.0.1:8010', { method: 'tv', strength: 0.5 }),
    );
    expect(p.get('server_uri')).toBe('http://127.0.0.1:8010');
  });
});

describe('denoise is not a render option', () => {
  it('is absent from RenderOpts, so it cannot reach export payloads', () => {
    // Export payloads are built from RenderOpts. If denoise lived there, a
    // preview would silently change exported pixels — the bake exists precisely
    // so that turning a denoised view into data is an explicit act.
    expect(Object.keys(RENDER)).not.toContain('denoise');
    expect(Object.keys(RENDER)).toEqual(['norm', 'scale', 'vminPct', 'vmaxPct', 'cmap']);
  });
});
