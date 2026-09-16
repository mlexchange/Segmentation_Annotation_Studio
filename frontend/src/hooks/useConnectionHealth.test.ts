import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { useConnectionHealth } from './useConnectionHealth';
import { useConnectionStore } from '@/stores/connectionStore';

beforeEach(() => {
  useConnectionStore.setState({
    kind: null,
    serverUri: null,
    browseContainerPath: null,
    browseFocusPath: null,
    localRoot: null,
    localRel: null,
    label: null,
    sampleCount: null,
    status: 'unknown',
  });
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('useConnectionHealth', () => {
  it('stays unknown and never fetches for a local connection', async () => {
    useConnectionStore.setState({ kind: 'local' });
    renderHook(() => useConnectionHealth());
    await Promise.resolve();
    expect(fetch).not.toHaveBeenCalled();
    expect(useConnectionStore.getState().status).toBe('unknown');
  });

  it('stays unknown when there is no connection at all', async () => {
    renderHook(() => useConnectionHealth());
    await Promise.resolve();
    expect(fetch).not.toHaveBeenCalled();
    expect(useConnectionStore.getState().status).toBe('unknown');
  });

  it('sets status to ok on a successful tiled health check', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
    useConnectionStore.setState({ kind: 'tiled', serverUri: 'http://tiled.example' });
    renderHook(() => useConnectionHealth());
    await waitFor(() => expect(useConnectionStore.getState().status).toBe('ok'));
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/tiled/list?'));
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('server_uri=http'));
  });

  it('sets status to error on a failed response', async () => {
    (fetch as any).mockResolvedValue({ ok: false });
    useConnectionStore.setState({ kind: 'tiled', serverUri: 'http://tiled.example' });
    renderHook(() => useConnectionHealth());
    await waitFor(() => expect(useConnectionStore.getState().status).toBe('error'));
  });

  it('sets status to error when fetch throws', async () => {
    (fetch as any).mockRejectedValue(new Error('network down'));
    useConnectionStore.setState({ kind: 'tiled', serverUri: 'http://tiled.example' });
    renderHook(() => useConnectionHealth());
    await waitFor(() => expect(useConnectionStore.getState().status).toBe('error'));
  });

  it('resets to unknown when the connection is cleared', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
    useConnectionStore.setState({ kind: 'tiled', serverUri: 'http://tiled.example' });
    const { rerender } = renderHook(() => useConnectionHealth());
    await waitFor(() => expect(useConnectionStore.getState().status).toBe('ok'));

    act(() => {
      useConnectionStore.setState({ kind: null, serverUri: null });
      rerender();
    });
    await waitFor(() => expect(useConnectionStore.getState().status).toBe('unknown'));
  });
});
