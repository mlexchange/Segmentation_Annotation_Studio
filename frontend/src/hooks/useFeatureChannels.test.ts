import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { useFeatureChannels } from './useFeatureChannels';
import { useIpredStore, DEFAULT_COMPOSITION_ID } from '@/stores/ipredStore';
import { useConnectionStore } from '@/stores/connectionStore';

const INITIAL_IPRED_STATE = useIpredStore.getState();
const INITIAL_CONN_STATE = useConnectionStore.getState();

function modulesResponse(overrides: Record<string, unknown>[] = []) {
  return {
    ok: true,
    json: async () => ({ modules: overrides }),
  };
}

function preprocessResponse(overrides: Record<string, unknown> = {}) {
  return {
    ok: true,
    json: async () => ({
      feature_id: 'feat-1',
      project_id: 'proj-1',
      setup_id: 'setup-1',
      slice_index: 0,
      n_channels: 3,
      width: 64,
      height: 64,
      labels: ['a', 'b', 'c'],
      cache_hit: false,
      ...overrides,
    }),
  };
}

beforeEach(() => {
  useIpredStore.setState(INITIAL_IPRED_STATE, true);
  useConnectionStore.setState(INITIAL_CONN_STATE, true);
  vi.stubGlobal('URL', Object.assign(URL, {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn(),
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const ARGS = { source: 'x.tif', kind: 'local', sliceIndex: 0, serverUri: null };

describe('useFeatureChannels', () => {
  it('probes SAM availability on mount and sets samAvailable when slimsam is ready', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(modulesResponse([{ id: 'slimsam', ready: true }])));
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(result.current.samAvailable).toBe(true));
  });

  it('samAvailable stays false when slimsam is absent or not ready', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(modulesResponse([{ id: 'slimsam', ready: false }])));
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect((fetch as any).mock.calls.length).toBeGreaterThan(0));
    expect(result.current.samAvailable).toBe(false);
  });

  it('samAvailable stays false when the modules fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('down')));
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect((fetch as any).mock.calls.length).toBeGreaterThan(0));
    expect(result.current.samAvailable).toBe(false);
  });

  it('compute() fails fast with no composition selected', async () => {
    useIpredStore.setState({ preferredCompositionId: '' });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(modulesResponse()));
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await act(async () => {
      await result.current.compute();
    });
    expect(result.current.error).toBe('Select a composition first.');
    expect(result.current.job).toBeNull();
  });

  it('compute() opens a session, preprocesses, and populates job + channelIndex 0', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse()) // sam probe on mount
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) }) // openIpredSession
      .mockResolvedValueOnce(preprocessResponse()) // ipredPreprocess
      .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['x']) }); // channel fetch triggered by channelIndex effect
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await act(async () => {
      await result.current.compute();
    });

    expect(result.current.job?.jobId).toBe('feat-1');
    expect(result.current.job?.channels).toEqual([
      { index: 0, label: 'a' }, { index: 1, label: 'b' }, { index: 2, label: 'c' },
    ]);
    expect(result.current.channelIndex).toBe(0);
    expect(useIpredStore.getState().ipredSessionId).toBe('sess-1');

    await waitFor(() => expect(result.current.channelUrl).toBe('blob:mock-url'));
  });

  it('compute() reuses an existing ipredSessionId instead of opening a new session', async () => {
    useIpredStore.setState({ ipredSessionId: 'existing-sess' });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce(preprocessResponse())
      .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['x']) });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    // Only 3 total calls: sam probe, preprocess, channel fetch — no session POST.
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(result.current.job?.jobId).toBe('feat-1');
  });

  it('compute() builds synthetic channel labels when the server returns none', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce(preprocessResponse({ labels: [], n_channels: 2 }))
      .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['x']) });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    expect(result.current.job?.channels).toEqual([
      { index: 0, label: 'channel 0' }, { index: 1, label: 'channel 1' },
    ]);
  });

  it('compute() sets hasSam true when the composition id mentions sam/slimsam/mark', async () => {
    useIpredStore.setState({ preferredCompositionId: 'comp-slimsam-x' });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce(preprocessResponse())
      .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['x']) });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    expect(result.current.job?.hasSam).toBe(true);
  });

  it('compute() sets error and clears job on a failed preprocess', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce({ ok: false, text: async () => 'boom' });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    expect(result.current.error).toBe('boom');
    expect(result.current.job).toBeNull();
    expect(result.current.computing).toBe(false);
  });

  it('compute() is a silent no-op when there is no source/kind open (guarded before ensureSession)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(modulesResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() =>
      useFeatureChannels({ source: null, kind: null, sliceIndex: 0, serverUri: null }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    expect(result.current.error).toBeNull();
    expect(result.current.job).toBeNull();
  });

  it('selectChannel/cycleChannel/clearSelection manipulate channelIndex', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce(preprocessResponse())
      .mockResolvedValue({ ok: true, blob: async () => new Blob(['x']) });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    expect(result.current.channelIndex).toBe(0);

    act(() => result.current.cycleChannel(1));
    expect(result.current.channelIndex).toBe(1);

    act(() => result.current.cycleChannel(-2)); // wraps around (3 channels)
    expect(result.current.channelIndex).toBe(2);

    act(() => result.current.selectChannel(0));
    expect(result.current.channelIndex).toBe(0);

    act(() => result.current.clearSelection());
    expect(result.current.channelIndex).toBeNull();
  });

  it('cycleChannel is a no-op with no job', () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(modulesResponse()));
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    act(() => result.current.cycleChannel(1));
    expect(result.current.channelIndex).toBeNull();
  });

  it('invalidateJob clears job/channelIndex/channelUrl', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce(preprocessResponse())
      .mockResolvedValue({ ok: true, blob: async () => new Blob(['x']) });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    await waitFor(() => expect(result.current.channelUrl).toBe('blob:mock-url'));

    act(() => result.current.invalidateJob());
    expect(result.current.job).toBeNull();
    expect(result.current.channelIndex).toBeNull();
    expect(result.current.channelUrl).toBeNull();
  });

  it('adoptFeatureBank populates job from an externally-produced feature bank', () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(modulesResponse()));
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    act(() => {
      result.current.adoptFeatureBank({
        featureId: 'ext-1', width: 32, height: 32, labels: ['x', 'y'], setupId: 'setup-sam',
      });
    });
    expect(result.current.job).toEqual({
      jobId: 'ext-1', width: 32, height: 32,
      channels: [{ index: 0, label: 'x' }, { index: 1, label: 'y' }],
      hasSam: true, setupId: 'setup-sam', cacheHit: undefined,
    });
  });

  it('resets job/channelIndex/error and revokes the channel URL when source/kind/slice/serverUri changes', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce(preprocessResponse())
      .mockResolvedValue({ ok: true, blob: async () => new Blob(['x']) });
    vi.stubGlobal('fetch', fetchMock);
    const { result, rerender } = renderHook((props) => useFeatureChannels(props), { initialProps: ARGS });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    await waitFor(() => expect(result.current.channelUrl).toBe('blob:mock-url'));

    act(() => { rerender({ ...ARGS, sliceIndex: 1 }); });
    expect(result.current.job).toBeNull();
    expect(result.current.channelIndex).toBeNull();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('sets an error when the channel PNG fetch itself fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(modulesResponse())
      .mockResolvedValueOnce({ ok: true, json: async () => ({ session_id: 'sess-1', project_id: 'proj-1' }) })
      .mockResolvedValueOnce(preprocessResponse())
      .mockResolvedValueOnce({ ok: false, status: 500 });
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useFeatureChannels(ARGS));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => {
      await result.current.compute();
    });
    await waitFor(() => expect(result.current.error).toMatch(/Channel fetch failed/));
    expect(result.current.channelUrl).toBeNull();
  });
});
