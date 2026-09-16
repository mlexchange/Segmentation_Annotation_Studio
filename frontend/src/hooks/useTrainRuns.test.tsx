import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useTrainRuns } from './useTrainRuns';

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

const RUN = {
  run_id: 'r1', model_family: 'dlsia_tunet', model_config: {}, classes: [], render: {},
  image_size: 256, hyperparams: {}, source_keys: [], created_at: '2024-01-01', metrics: {},
};

describe('useTrainRuns', () => {
  it('starts empty and loading', () => {
    (fetch as any).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    expect(result.current.isLoading).toBe(true);
    expect(result.current.runs).toEqual([]);
  });

  it('returns the loaded runs list', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => ({ runs: [RUN] }) });
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.runs).toEqual([RUN]);
  });

  it('returns an empty list (not an error) on a failed fetch', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500 });
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.runs).toEqual([]);
  });

  it('defaults to an empty list when the response has no runs key', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => ({}) });
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.runs).toEqual([]);
  });

  it('deleteRun issues a DELETE and refreshes the list on success', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ runs: [RUN] }) })
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ runs: [] }) });
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    let ok: boolean = false;
    await act(async () => {
      ok = await result.current.deleteRun('r1');
    });
    expect(ok).toBe(true);
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/train/runs/r1'), { method: 'DELETE' });
    await waitFor(() => expect(result.current.runs).toEqual([]));
  });

  it('deleteRun returns false on a failed delete', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ runs: [RUN] }) })
      .mockResolvedValueOnce({ ok: false });
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    let ok: boolean = true;
    await act(async () => {
      ok = await result.current.deleteRun('r1');
    });
    expect(ok).toBe(false);
  });

  it('deleteRun returns false when the request throws', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ runs: [RUN] }) })
      .mockRejectedValueOnce(new Error('network down'));
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    let ok: boolean = true;
    await act(async () => {
      ok = await result.current.deleteRun('r1');
    });
    expect(ok).toBe(false);
  });

  it('URL-encodes the run id in the delete request', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ runs: [] }) })
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ runs: [] }) });
    const { result } = renderHook(() => useTrainRuns(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    await act(async () => {
      await result.current.deleteRun('run/with slash');
    });
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining(encodeURIComponent('run/with slash')), expect.anything());
  });
});
