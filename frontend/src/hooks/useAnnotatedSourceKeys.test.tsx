import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useAnnotatedSourceKeys } from './useAnnotatedSourceKeys';
import { useAnnotationStore } from '@/stores/annotationStore';

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  useAnnotationStore.getState().reset();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('useAnnotatedSourceKeys', () => {
  it('includes in-session sources with at least one shape', () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice('local:a.tif', 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    const { result } = renderHook(() => useAnnotatedSourceKeys(), { wrapper });
    expect(result.current.has('local:a.tif')).toBe(true);
  });

  it('excludes a source whose slices are all empty', () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice('local:b.tif', 0, 1, []);
    const { result } = renderHook(() => useAnnotatedSourceKeys(), { wrapper });
    expect(result.current.has('local:b.tif')).toBe(false);
  });

  it('merges in annotated sources from the backend drafts endpoint', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => [
        { source_key: 'local:c.tif', has_annotations: true },
        { source_key: 'local:d.tif', has_annotations: false },
      ],
    });
    const { result } = renderHook(() => useAnnotatedSourceKeys(), { wrapper });
    await waitFor(() => expect(result.current.has('local:c.tif')).toBe(true));
    expect(result.current.has('local:d.tif')).toBe(false);
  });

  it('returns an empty set when the drafts fetch fails', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500 });
    const { result } = renderHook(() => useAnnotatedSourceKeys(), { wrapper });
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    expect(result.current.size).toBe(0);
  });
});
