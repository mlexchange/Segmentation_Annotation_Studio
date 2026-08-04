import type { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useImageSlice, useLoadedSliceImage, type SliceResult } from './useImageSlice';
import type { RenderOpts } from '@/stores/datasetStore';

const RENDER_OPTS: RenderOpts = {
  norm: 'global', scale: 'linear', vminPct: 1, vmaxPct: 99, cmap: 'gray',
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function queryWrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe('useImageSlice', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn((blob: Blob) => `blob:${blob.size}`),
      revokeObjectURL: vi.fn(),
    });
  });

  afterEach(() => vi.unstubAllGlobals());

  it('passes the query abort signal and never publishes a superseded response', async () => {
    const first = deferred<{ ok: boolean; status: number; blob: () => Promise<Blob> }>();
    const second = deferred<{ ok: boolean; status: number; blob: () => Promise<Blob> }>();
    vi.mocked(fetch)
      .mockReturnValueOnce(first.promise as Promise<Response>)
      .mockReturnValueOnce(second.promise as Promise<Response>);

    const { result, rerender, unmount } = renderHook(
      ({ source }) => useImageSlice(source, 'local', 0, RENDER_OPTS, null),
      { initialProps: { source: 'first.tif' }, wrapper: queryWrapper() },
    );

    expect(vi.mocked(fetch).mock.calls[0][1]).toEqual(
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    const firstSignal = (vi.mocked(fetch).mock.calls[0][1] as RequestInit).signal as AbortSignal;

    rerender({ source: 'second.tif' });
    expect(firstSignal.aborted).toBe(true);
    expect(result.current.data).toBeUndefined();

    await act(async () => {
      first.resolve({ ok: true, status: 200, blob: async () => new Blob(['old']) });
      await first.promise;
    });
    expect(result.current.data).toBeUndefined();

    await act(async () => {
      second.resolve({ ok: true, status: 200, blob: async () => new Blob(['new-image']) });
      await second.promise;
    });
    await waitFor(() => expect(result.current.data?.url).toBe('blob:9'));
    expect(result.current.data?.requestKey).toContain('second.tif');
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('revokes the previous object URL as soon as the requested identity changes', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true, status: 200, blob: async () => new Blob(['image']),
    } as Response);
    const { result, rerender, unmount } = renderHook(
      ({ slice }) => useImageSlice('sample.tif', 'local', slice, RENDER_OPTS, null),
      { initialProps: { slice: 0 }, wrapper: queryWrapper() },
    );
    await waitFor(() => expect(result.current.data).toBeDefined());
    const oldUrl = result.current.data!.url;

    rerender({ slice: 1 });
    expect(result.current.data).toBeUndefined();
    await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith(oldUrl));

    await waitFor(() => expect(result.current.data).toBeDefined());
    const currentUrl = result.current.data!.url;
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(currentUrl);
  });
});

describe('useLoadedSliceImage', () => {
  class MockImage {
    static instances: MockImage[] = [];
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    src = '';

    constructor() { MockImage.instances.push(this); }
  }

  beforeEach(() => {
    MockImage.instances = [];
    vi.stubGlobal('Image', MockImage);
  });

  afterEach(() => vi.unstubAllGlobals());

  it('clears readiness on identity change and ignores late load callbacks', () => {
    const first: SliceResult = { requestKey: 'first', url: 'blob:first' };
    const second: SliceResult = { requestKey: 'second', url: 'blob:second' };
    const { result, rerender } = renderHook(({ slice }) => useLoadedSliceImage(slice), {
      initialProps: { slice: first as SliceResult | undefined },
    });

    expect(result.current.image).toBeNull();
    rerender({ slice: second });
    expect(result.current.image).toBeNull();

    act(() => MockImage.instances[0].onload?.());
    expect(result.current.image).toBeNull();

    act(() => MockImage.instances[1].onload?.());
    expect(result.current.requestKey).toBe('second');
    expect(result.current.image).toBe(MockImage.instances[1]);
  });

  it('surfaces image decode errors without retaining the previous image', () => {
    const slice: SliceResult = { requestKey: 'broken', url: 'blob:broken' };
    const { result } = renderHook(() => useLoadedSliceImage(slice));
    act(() => MockImage.instances[0].onerror?.());
    expect(result.current.image).toBeNull();
    expect(result.current.error).toMatch(/decode/i);
  });
});
