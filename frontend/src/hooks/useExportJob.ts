/**
 * useExportJob — drive a background COCO/mask export and stream its progress.
 *
 * POSTs to /api/export/coco (which returns a job_id), then polls
 * /api/export/status/{job_id} until done/error, exposing phase, done/total, and
 * a live log for the UI. When finished, `downloadUrl` points at the .zip so the
 * browser's save dialog can write it to the user's machine. The server-side
 * (Tiled/EXPORT_ROOT) copy is always written regardless.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE } from '@/config';

export interface ExportJobState {
  status: 'idle' | 'running' | 'done' | 'error';
  phase: string;
  done: number;
  total: number;
  log: string[];
  result: Record<string, unknown> | null;
  error: string | null;
  jobId: string | null;
}

const IDLE: ExportJobState = {
  status: 'idle', phase: '', done: 0, total: 0, log: [], result: null, error: null, jobId: null,
};

/**
 * Drives a background COCO export: returns the live job `state`, a `start` action,
 * `reset`, and a `downloadUrl` for the result zip when available.
 */
export function useExportJob() {
  const [state, setState] = useState<ExportJobState>(IDLE);
  const timer = useRef<number | null>(null);

  /** Cancel any pending poll timeout. */
  const clearTimer = () => {
    if (timer.current !== null) { window.clearTimeout(timer.current); timer.current = null; }
  };
  useEffect(() => clearTimer, []);

  /** Stop polling and return state to idle. */
  const reset = useCallback(() => { clearTimer(); setState(IDLE); }, []);

  /** Recursively poll /api/export/status/{jobId} every 500ms until done/error. */
  const poll = useCallback((jobId: string) => {
    const tick = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/export/status/${jobId}`);
        if (!r.ok) { setState((s) => ({ ...s, status: 'error', error: `Status ${r.status}` })); return; }
        const j = await r.json();
        const status: ExportJobState['status'] =
          j.state === 'done' ? 'done' : j.state === 'error' ? 'error' : 'running';
        setState((s) => ({
          ...s, status, phase: j.phase ?? '', done: j.done ?? 0, total: j.total ?? 0,
          log: j.log ?? [], result: j.result ?? null, error: j.error ?? null,
        }));
        if (j.state === 'done' || j.state === 'error') return; // stop polling
      } catch (e) {
        setState((s) => ({ ...s, status: 'error', error: String(e) }));
        return;
      }
      timer.current = window.setTimeout(tick, 500);
    };
    tick();
  }, []);

  /** POST a job-starting request to `path`, then poll the shared status route. */
  const run = useCallback(async (path: string, payload: unknown) => {
    clearTimer();
    setState({ ...IDLE, status: 'running', phase: 'queued' });
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const msg = await res.text(); setState((s) => ({ ...s, status: 'error', error: msg })); return; }
      const data = await res.json();
      if (data.job_id) {
        setState((s) => ({ ...s, jobId: data.job_id }));
        poll(data.job_id);
      } else {
        // Synchronous response (e.g. dry-run) — treat as immediately done.
        setState((s) => ({ ...s, status: 'done', result: data }));
      }
    } catch (e) {
      setState((s) => ({ ...s, status: 'error', error: String(e) }));
    }
  }, [poll]);

  /** Start a real COCO export. Returns once the job is queued (progress via state). */
  const start = useCallback((payload: unknown) => run('/api/export/coco', payload), [run]);

  /** Write rasterized masks into Tiled (standalone). Shares the status polling. */
  const startMaskSync = useCallback((payload: unknown) => run('/api/masks/to-tiled', payload), [run]);

  const downloadUrl =
    state.jobId && (state.result as { zip_available?: boolean } | null)?.zip_available
      ? `${API_BASE}/api/export/download/${state.jobId}`
      : null;

  return { state, start, startMaskSync, reset, downloadUrl };
}
