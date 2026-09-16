import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { DEFAULT_MANIFOLD_PARAMS, useFeatureManifold } from './useFeatureManifold';
import type { Shape } from '@/stores/annotationStore';

function sampleOk(overrides: Record<string, unknown> = {}) {
  return {
    ok: true,
    json: async () => ({
      sample_id: 'samp-1',
      points: [{ x: 1, y: 2, radius: 5 }],
      k: 24,
      n_picked: 1,
      n_subsample: 100,
      explained_variance: 0.8,
      radius: 5,
      box_size: 64,
      ...overrides,
    }),
  };
}
function heatmapOk() {
  return { ok: true, blob: async () => new Blob(['x']) };
}

beforeEach(() => {
  vi.stubGlobal('URL', Object.assign(URL, {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn(),
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('useFeatureManifold', () => {
  it('starts with default params and empty results', () => {
    vi.stubGlobal('fetch', vi.fn());
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: null }));
    expect(result.current.params).toEqual(DEFAULT_MANIFOLD_PARAMS);
    expect(result.current.points).toEqual([]);
    expect(result.current.heatmapUrl).toBeNull();
    expect(result.current.hasSample).toBe(false);
    expect(result.current.meta).toBeNull();
  });

  it('sample() is a no-op with no featureJobId', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: null }));
    await act(async () => {
      await result.current.sample();
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('sample() posts feature_id/k/box_size, then fetches the heatmap, populating points/meta/heatmapUrl', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sampleOk())
      .mockResolvedValueOnce(heatmapOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));

    await act(async () => {
      await result.current.sample();
    });

    expect(result.current.points).toEqual([{ x: 1, y: 2, radius: 5 }]);
    expect(result.current.heatmapUrl).toBe('blob:mock-url');
    expect(result.current.meta).toEqual({
      k: 24, nPicked: 1, nSubsample: 100, explainedVariance: 0.8, radius: 5, boxSize: 64,
      hasMask: undefined, maskPixels: undefined,
    });
    expect(result.current.hasSample).toBe(true);
    expect(result.current.sampling).toBe(false);

    const [sampleUrl, sampleInit] = fetchMock.mock.calls[0];
    expect(sampleUrl).toBe('/api/ipred/manifold/sample');
    const body = JSON.parse(sampleInit.body);
    expect(body).toEqual({ feature_id: 'feat-1', k: 24, box_size: 64 });
    const [heatUrl] = fetchMock.mock.calls[1];
    expect(heatUrl).toBe('/api/ipred/manifold/samp-1/heatmap.png');
  });

  it('sample() includes shapes in the body when a placement mask is set', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sampleOk())
      .mockResolvedValueOnce(heatmapOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    const shapes: Shape[] = [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1 }];

    act(() => result.current.setPlacementMaskFromShapes(shapes));
    expect(result.current.placementMask).toEqual(shapes);

    await act(async () => {
      await result.current.sample();
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.shapes).toEqual(shapes);
  });

  it('setPlacementMaskFromShapes([]) clears the mask instead of storing an empty array', () => {
    vi.stubGlobal('fetch', vi.fn());
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    act(() => result.current.setPlacementMaskFromShapes([]));
    expect(result.current.placementMask).toBeNull();
  });

  it('clearPlacementMask resets placementMask to null', () => {
    vi.stubGlobal('fetch', vi.fn());
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    act(() => result.current.setPlacementMaskFromShapes([
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1 },
    ]));
    act(() => result.current.clearPlacementMask());
    expect(result.current.placementMask).toBeNull();
  });

  it('sample() surfaces a parsed JSON `detail` error message and revokes any prior heatmap', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({
      ok: false,
      text: async () => JSON.stringify({ detail: 'bad k value' }),
    });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    await act(async () => {
      await result.current.sample();
    });
    expect(result.current.error).toBe('bad k value');
    expect(result.current.heatmapUrl).toBeNull();
    expect(result.current.sampling).toBe(false);
  });

  it('sample() falls back to the raw body text when it is not JSON', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({ ok: false, status: 500, text: async () => 'plain text error' });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    await act(async () => {
      await result.current.sample();
    });
    expect(result.current.error).toBe('plain text error');
  });

  it('sample() treats a "not found"/"unknown feature" detail specially: calls onFeatureJobExpired and shows a recompute hint', async () => {
    const onFeatureJobExpired = vi.fn();
    const fetchMock = vi.fn().mockResolvedValueOnce({
      ok: false,
      text: async () => JSON.stringify({ detail: 'unknown feature id' }),
    });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1', onFeatureJobExpired }));
    await act(async () => {
      await result.current.sample();
    });
    expect(onFeatureJobExpired).toHaveBeenCalledTimes(1);
    expect(result.current.error).toMatch(/Compute again/);
  });

  it('sample() errors when the heatmap fetch itself fails, after a successful sample POST', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sampleOk())
      .mockResolvedValueOnce({ ok: false, status: 500 });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    await act(async () => {
      await result.current.sample();
    });
    expect(result.current.error).toBe('Failed to fetch manifold heatmap');
    expect(result.current.heatmapUrl).toBeNull();
  });

  it('a concurrent sample() call while one is in-flight is ignored', async () => {
    let resolveFirst!: (v: unknown) => void;
    const fetchMock = vi.fn()
      .mockReturnValueOnce(new Promise((resolve) => { resolveFirst = resolve; }))
      .mockResolvedValueOnce(heatmapOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));

    let p1: Promise<void>;
    let p2: Promise<void>;
    act(() => {
      p1 = result.current.sample();
      p2 = result.current.sample();
    });
    expect(fetchMock).toHaveBeenCalledTimes(1); // second call short-circuited by samplingRef

    await act(async () => {
      resolveFirst(sampleOk());
      await Promise.all([p1, p2]);
    });
  });

  it('resets heatmap/points/meta/error/placementMask when featureJobId changes', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sampleOk())
      .mockResolvedValueOnce(heatmapOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result, rerender } = renderHook(
      ({ featureJobId }) => useFeatureManifold({ featureJobId }),
      { initialProps: { featureJobId: 'feat-1' } },
    );
    await act(async () => {
      await result.current.sample();
    });
    expect(result.current.hasSample).toBe(true);

    act(() => { rerender({ featureJobId: 'feat-2' }); });
    expect(result.current.heatmapUrl).toBeNull();
    expect(result.current.points).toEqual([]);
    expect(result.current.meta).toBeNull();
    expect(result.current.placementMask).toBeNull();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('debounces a re-sample 350ms after k/boxSize changes, but only once a sample already exists', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fetchMock = vi.fn()
      .mockResolvedValue(sampleOk())
      // interleave heatmap responses; every other call is the heatmap fetch
      ;
    fetchMock.mockImplementation(async (url: string) => (url.includes('heatmap') ? heatmapOk() : sampleOk()));
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));

    await act(async () => {
      await result.current.sample();
    });
    expect(result.current.hasSample).toBe(true);
    fetchMock.mockClear();

    act(() => { result.current.setParams((p) => ({ ...p, k: 40 })); });
    // Not yet — debounce hasn't elapsed.
    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    expect(fetchMock).not.toHaveBeenCalled();

    await act(async () => { await vi.advanceTimersByTimeAsync(200); });
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });

  it('does not auto re-sample on param change before any sample has been taken', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fetchMock = vi.fn().mockResolvedValue(sampleOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));

    act(() => { result.current.setParams((p) => ({ ...p, k: 40 })); });
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('dismiss() clears the heatmap/points/error', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sampleOk())
      .mockResolvedValueOnce(heatmapOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    await act(async () => {
      await result.current.sample();
    });
    expect(result.current.hasSample).toBe(true);

    act(() => result.current.dismiss());
    expect(result.current.heatmapUrl).toBeNull();
    expect(result.current.points).toEqual([]);
    expect(result.current.error).toBeNull();
    expect(result.current.hasSample).toBe(false);
  });

  it('revokes the heatmap object URL on unmount', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sampleOk())
      .mockResolvedValueOnce(heatmapOk());
    vi.stubGlobal('fetch', fetchMock);
    const { result, unmount } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    await act(async () => {
      await result.current.sample();
    });
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('setShowHeatmap/setShowMarkers/setHeatmapOpacity update their state', () => {
    vi.stubGlobal('fetch', vi.fn());
    const { result } = renderHook(() => useFeatureManifold({ featureJobId: 'feat-1' }));
    act(() => result.current.setShowHeatmap(false));
    expect(result.current.showHeatmap).toBe(false);
    act(() => result.current.setShowMarkers(false));
    expect(result.current.showMarkers).toBe(false);
    act(() => result.current.setHeatmapOpacity(0.9));
    expect(result.current.heatmapOpacity).toBe(0.9);
  });
});
