/**
 * Regression guard for a whole-app white screen.
 *
 * Consumers reach into nested capability keys (`capability.denoise.methods`,
 * `capability.dinov3.checkpoints`). When the running backend is older than this
 * frontend build — a dev server whose process started before a capability key
 * was added, or a stale deploy — that key is simply absent from the response.
 * `data ?? FALLBACK` doesn't help there (the response exists, it's just missing
 * a key), so the nested read threw a TypeError during render, React unmounted
 * the tree, and the entire page went blank. A missing capability must degrade to
 * "feature unavailable", never take down the app.
 *
 * These tests exercise the merge through the real hook, mocking only fetch.
 */
import type { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useTrainCapability } from './useTrainCapability';

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

const respondWith = (body: unknown) => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => body })));
};

describe('useTrainCapability response merging', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('fills in denoise when an older backend omits it entirely', async () => {
    // Exactly the shape a backend predating the denoise feature returns.
    respondWith({
      torch_available: true, torch_version: '2.4.0', device: 'mps',
      dinov3: { available: true, checkpoints: [] },
      dlsia: { available: true },
      models_dir: '/m', runs_dir: '/r', busy: false,
    });

    const { result } = renderHook(() => useTrainCapability(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.capability.torch_available).toBe(true));

    // The read that used to throw and blank the page.
    expect(result.current.capability.denoise.methods).toEqual([]);
    expect(result.current.capability.denoise.available).toBe(false);
  });

  it('fills in nested keys when a partially-failed probe omits them', async () => {
    respondWith({ busy: false, error: 'probe failed' });

    const { result } = renderHook(() => useTrainCapability(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.capability.error).toBe('probe failed'));

    expect(result.current.capability.denoise.methods).toEqual([]);
    expect(result.current.capability.dinov3.checkpoints).toEqual([]);
    expect(result.current.capability.dlsia.available).toBe(false);
  });

  it('keeps real server values rather than clobbering them with defaults', async () => {
    respondWith({
      torch_available: true, torch_version: '2.4.0', device: 'mps',
      dinov3: { available: true, checkpoints: [] },
      dlsia: { available: true },
      denoise: {
        available: true,
        methods: [{ method: 'tv', label: 'Total variation', cost: 'moderate', description: 'd', available: true, z_radius: 0 }],
      },
      models_dir: '/m', runs_dir: '/r', busy: true,
    });

    const { result } = renderHook(() => useTrainCapability(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.capability.denoise.available).toBe(true));

    expect(result.current.capability.denoise.methods).toHaveLength(1);
    expect(result.current.capability.denoise.methods[0].method).toBe('tv');
    expect(result.current.capability.busy).toBe(true);
    expect(result.current.capability.dlsia.available).toBe(true);
  });

  it('falls back completely when the request fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 500 })));

    const { result } = renderHook(() => useTrainCapability(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.capability.denoise.methods).toEqual([]);
    expect(result.current.capability.torch_available).toBe(false);
  });
});
