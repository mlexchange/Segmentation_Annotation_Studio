import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { useExportJob } from './useExportJob';

beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('useExportJob', () => {
  it('starts idle', () => {
    const { result } = renderHook(() => useExportJob());
    expect(result.current.state.status).toBe('idle');
    expect(result.current.downloadUrl).toBeNull();
  });

  it('a synchronous (no job_id) response is treated as immediately done', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => ({ ok: true, zip_available: false }) });
    const { result } = renderHook(() => useExportJob());
    await act(async () => { await result.current.start({}); });
    expect(result.current.state.status).toBe('done');
    expect(result.current.state.result).toEqual({ ok: true, zip_available: false });
  });

  it('a job_id response starts polling and reflects progress then done', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'j1' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'running', phase: 'working', done: 1, total: 4 }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'done', phase: 'done', done: 4, total: 4, result: { zip_available: true } }) });

    const { result } = renderHook(() => useExportJob());
    await act(async () => { await result.current.start({}); });
    expect(result.current.state.jobId).toBe('j1');

    await waitFor(() => expect(result.current.state.phase).toBe('working'));
    await waitFor(() => expect(result.current.state.status).toBe('done'), { timeout: 3000 });
    expect(result.current.downloadUrl).toContain('/api/export/download/j1');
  });

  it('a non-ok POST response surfaces a formatted error and does not poll', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 422, text: async () => JSON.stringify({ detail: [{ msg: 'bad', loc: ['body', 'x'] }] }) });
    const { result } = renderHook(() => useExportJob());
    await act(async () => { await result.current.start({}); });
    expect(result.current.state.status).toBe('error');
    expect(result.current.state.error).toBeTruthy();
    expect(fetch).toHaveBeenCalledTimes(1); // no follow-up poll
  });

  it('a network error during start() surfaces as an error state', async () => {
    (fetch as any).mockRejectedValue(new Error('offline'));
    const { result } = renderHook(() => useExportJob());
    await act(async () => { await result.current.start({}); });
    expect(result.current.state.status).toBe('error');
    expect(result.current.state.error).toContain('offline');
  });

  it('a 404 while polling a resumed job clears it and returns to idle', async () => {
    sessionStorage.setItem('exportJob:key1', 'stale-job');
    (fetch as any).mockResolvedValue({ status: 404, ok: false });
    const { result } = renderHook(() => useExportJob('key1'));
    expect(result.current.state.status).toBe('running'); // optimistic placeholder
    await waitFor(() => expect(result.current.state.status).toBe('idle'));
    expect(sessionStorage.getItem('exportJob:key1')).toBeNull();
  });

  it('resumes polling a persisted job id on a fresh mount', async () => {
    sessionStorage.setItem('exportJob:key2', 'resumed-job');
    (fetch as any).mockResolvedValue({ ok: true, json: async () => ({ state: 'done', phase: 'done', result: {} }) });
    const { result } = renderHook(() => useExportJob('key2'));
    await waitFor(() => expect(result.current.state.status).toBe('done'));
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/export/status/resumed-job'));
  });

  it('reset clears state and any persisted job id', async () => {
    sessionStorage.setItem('exportJob:key3', 'j1');
    (fetch as any).mockResolvedValue({ ok: true, json: async () => ({ state: 'running', phase: 'x' }) });
    const { result } = renderHook(() => useExportJob('key3'));
    await waitFor(() => expect(result.current.state.phase).toBe('x'));
    act(() => result.current.reset());
    expect(result.current.state.status).toBe('idle');
    expect(sessionStorage.getItem('exportJob:key3')).toBeNull();
  });

  it('startMaskSync/startIpredBatchTrain/startIpredBatchApply hit their own routes', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => ({ ok: true }) });
    const { result } = renderHook(() => useExportJob());
    await act(async () => { await result.current.startMaskSync({}); });
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining('/api/masks/to-tiled'), expect.anything());

    await act(async () => { await result.current.startIpredBatchTrain({}); });
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining('/api/ipred/batch/train'), expect.anything());

    await act(async () => { await result.current.startIpredBatchApply({}); });
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining('/api/ipred/batch/apply'), expect.anything());
  });
});
