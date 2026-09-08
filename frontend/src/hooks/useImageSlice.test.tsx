import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useImageSlice } from './useImageSlice';
import type { RenderOpts } from '@/stores/datasetStore';

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const RENDER_OPTS: RenderOpts = { norm: 'slice', scale: 'linear', vminPct: 1, vmaxPct: 99, cmap: 'gray' };

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL = vi.fn(() => 'blob:fake-url');
    static revokeObjectURL = vi.fn();
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('useImageSlice', () => {
  it('is disabled (no fetch) when source is null', () => {
    const { result } = renderHook(
      () => useImageSlice(null, 'local', 0, RENDER_OPTS, null),
      { wrapper },
    );
    expect(result.current.fetchStatus).toBe('idle');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('is disabled when kind is null', () => {
    renderHook(() => useImageSlice('x.tif', null, 0, RENDER_OPTS, null), { wrapper });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('fetches the slice and resolves to a blob object URL', async () => {
    const fakeBlob = new Blob(['data']);
    (fetch as any).mockResolvedValue({ ok: true, blob: async () => fakeBlob });
    const { result } = renderHook(
      () => useImageSlice('x.tif', 'local', 0, RENDER_OPTS, null),
      { wrapper },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBe('blob:fake-url');
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/image/slice?'));
  });

  it('surfaces a fetch failure as an error state', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500 });
    const { result } = renderHook(
      () => useImageSlice('x.tif', 'local', 0, RENDER_OPTS, null),
      { wrapper },
    );
    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('includes the denoise method/strength in the request URL when active', async () => {
    (fetch as any).mockResolvedValue({ ok: true, blob: async () => new Blob() });
    renderHook(
      () => useImageSlice('x.tif', 'local', 0, RENDER_OPTS, null, { method: 'median', strength: 0.4 }),
      { wrapper },
    );
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    const url = (fetch as any).mock.calls[0][0];
    expect(url).toContain('denoise_method=median');
    expect(url).toContain('denoise_strength=0.4');
  });
});
