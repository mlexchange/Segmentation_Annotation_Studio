/**
 * useExportJob persistence: a job survives this hook's own component unmounting
 * (e.g. the user switches from Train to Browse and back) as long as `persistKey`
 * is given — the real job keeps running server-side regardless, so the fix is
 * purely about the polling loop reattaching to it rather than starting blank.
 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useExportJob } from './useExportJob';

function response(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response(200, {})));
});

describe('without a persistKey', () => {
  it('behaves exactly as before: starts idle, nothing written to storage', () => {
    const { result } = renderHook(() => useExportJob());
    expect(result.current.state.status).toBe('idle');
    expect(sessionStorage.length).toBe(0);
  });

  it('does not persist a job id across a remount', async () => {
    const { result, unmount } = renderHook(() => useExportJob());
    vi.mocked(fetch).mockResolvedValue(response(200, { job_id: 'job-1' }));

    await act(async () => { await result.current.startJob('/api/train/start', {}); });
    expect(sessionStorage.length).toBe(0);

    unmount();
    const { result: remounted } = renderHook(() => useExportJob());
    expect(remounted.current.state.status).toBe('idle');
  });
});

describe('with a persistKey', () => {
  it('writes the job id to sessionStorage when a job starts', async () => {
    vi.mocked(fetch).mockResolvedValue(response(200, { job_id: 'job-42' }));
    const { result } = renderHook(() => useExportJob('train:start'));

    await act(async () => { await result.current.startJob('/api/train/start', {}); });

    expect(sessionStorage.getItem('exportJob:train:start')).toBe('job-42');
  });

  it('reattaches to a running job on the next mount instead of starting idle', async () => {
    sessionStorage.setItem('exportJob:train:start', 'job-99');
    vi.mocked(fetch).mockResolvedValue(
      response(200, { state: 'running', phase: 'training', done: 5, total: 10, log: ['epoch 5/10'] }),
    );

    const { result } = renderHook(() => useExportJob('train:start'));

    // Immediately (before the resume poll's first tick resolves) it must already
    // read as "running", not "idle" — the whole point is no blank progress bar.
    expect(result.current.state.status).toBe('running');
    expect(result.current.state.jobId).toBe('job-99');

    await waitFor(() => expect(result.current.state.phase).toBe('training'));
    expect(result.current.state.done).toBe(5);
    expect(result.current.state.total).toBe(10);
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/export/status/job-99'));
  });

  it('picks up a job that finished while unmounted, without ever showing a blank bar', async () => {
    sessionStorage.setItem('exportJob:train:start', 'job-done');
    vi.mocked(fetch).mockResolvedValue(
      response(200, { state: 'done', phase: 'done', done: 10, total: 10, result: { run_id: 'r1' } }),
    );

    const { result } = renderHook(() => useExportJob('train:start'));
    await waitFor(() => expect(result.current.state.status).toBe('done'));
    expect(result.current.state.result).toEqual({ run_id: 'r1' });
  });

  it('clears the persisted id and goes idle if the job is gone (404), not an error', async () => {
    sessionStorage.setItem('exportJob:train:start', 'job-expired');
    vi.mocked(fetch).mockResolvedValue(response(404, {}));

    const { result } = renderHook(() => useExportJob('train:start'));
    await waitFor(() => expect(result.current.state.status).toBe('idle'));
    expect(sessionStorage.getItem('exportJob:train:start')).toBeNull();
  });

  it('does not resume a job that was never started (no stored id)', () => {
    const { result } = renderHook(() => useExportJob('train:start'));
    expect(result.current.state.status).toBe('idle');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('reset() clears the persisted id so a later mount does not resume it', async () => {
    vi.mocked(fetch).mockResolvedValue(response(200, { job_id: 'job-1' }));
    const { result } = renderHook(() => useExportJob('train:probe'));
    await act(async () => { await result.current.startJob('/api/train/estimate-batch', {}); });
    expect(sessionStorage.getItem('exportJob:train:probe')).toBe('job-1');

    act(() => result.current.reset());

    expect(sessionStorage.getItem('exportJob:train:probe')).toBeNull();
    expect(result.current.state.status).toBe('idle');
  });

  it('different persistKeys never collide with each other', async () => {
    vi.mocked(fetch).mockResolvedValue(response(200, { job_id: 'job-a' }));
    const a = renderHook(() => useExportJob('train:start'));
    await act(async () => { await a.result.current.startJob('/api/train/start', {}); });

    vi.mocked(fetch).mockResolvedValue(response(200, { job_id: 'job-b' }));
    const b = renderHook(() => useExportJob('train:probe'));
    await act(async () => { await b.result.current.startJob('/api/train/estimate-batch', {}); });

    expect(sessionStorage.getItem('exportJob:train:start')).toBe('job-a');
    expect(sessionStorage.getItem('exportJob:train:probe')).toBe('job-b');
  });

  it('a synchronous (job-id-less) response clears any previously-persisted id', async () => {
    sessionStorage.setItem('exportJob:train:start', 'stale-job');
    vi.mocked(fetch).mockResolvedValue(response(200, { some: 'sync-result' }));

    const { result } = renderHook(() => useExportJob('train:start'));
    // The resume attempt for 'stale-job' fires first; let it settle before
    // starting a brand new (synchronous) job on the same key.
    await waitFor(() => expect(fetch).toHaveBeenCalled());

    await act(async () => { await result.current.startJob('/api/train/start', {}); });

    expect(sessionStorage.getItem('exportJob:train:start')).toBeNull();
  });
});
