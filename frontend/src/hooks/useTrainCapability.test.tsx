import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useTrainCapability } from './useTrainCapability';

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useTrainCapability', () => {
  it('returns the fallback shape while loading', () => {
    (fetch as any).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useTrainCapability(), { wrapper });
    expect(result.current.isLoading).toBe(true);
    expect(result.current.capability.torch_available).toBe(false);
    expect(result.current.capability.dlsia).toEqual({ available: false });
  });

  it('returns the parsed capability once loaded', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        torch_available: true,
        torch_version: '2.1.0',
        device: 'cpu',
        dlsia: { available: true },
        denoise: { available: true, methods: [] },
        runs_dir: '/data/runs',
        busy: false,
      }),
    });
    const { result } = renderHook(() => useTrainCapability(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.capability.torch_available).toBe(true);
    expect(result.current.capability.device).toBe('cpu');
  });

  it('fills in a missing nested key (e.g. denoise) from the fallback instead of throwing', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ torch_available: true, dlsia: { available: true } }),
    });
    const { result } = renderHook(() => useTrainCapability(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.capability.denoise).toEqual({ available: false, methods: [] });
    expect(result.current.capability.torch_available).toBe(true);
  });

  it('falls back to defaults entirely on a fetch failure', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500 });
    const { result } = renderHook(() => useTrainCapability(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.capability.torch_available).toBe(false);
  });
});
