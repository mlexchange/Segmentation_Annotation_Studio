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
import { formatApiError } from '@/lib/apiError';

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

const storageKeyFor = (persistKey: string) => `exportJob:${persistKey}`;

/**
 * Drives a background COCO export: returns the live job `state`, a `start` action,
 * `reset`, and a `downloadUrl` for the result zip when available.
 *
 * `persistKey`, when given, survives this component unmounting mid-job — e.g. the
 * user switches from Train to Browse and back while training runs in the
 * background. Without it, the job keeps running server-side (it's independent of
 * any client), but the polling loop lived in this hook's local state, so a fresh
 * mount used to come back to a blank "idle" progress bar with no way to tell a
 * job was still going. The job id is stashed in sessionStorage on start and
 * reattached to the same polling loop on the next mount; callers using the same
 * `persistKey` for genuinely different jobs (e.g. a different sample) should
 * include whatever varies in the key so they don't reconnect to the wrong one.
 */
export function useExportJob(persistKey?: string) {
  const storageKey = persistKey ? storageKeyFor(persistKey) : null;

  const [state, setState] = useState<ExportJobState>(() => {
    const savedId = storageKey ? sessionStorage.getItem(storageKey) : null;
    // Placeholder until the resume effect's first tick fills in the real
    // phase/progress/result — avoids a flash of "idle" for a job that, most of
    // the time, is still genuinely running.
    return savedId ? { ...IDLE, status: 'running', jobId: savedId } : IDLE;
  });
  const timer = useRef<number | null>(null);

  /** Cancel any pending poll timeout. */
  const clearTimer = () => {
    if (timer.current !== null) { window.clearTimeout(timer.current); timer.current = null; }
  };
  useEffect(() => clearTimer, []);

  /** Stop polling and return state to idle. */
  const reset = useCallback(() => {
    clearTimer();
    if (storageKey) sessionStorage.removeItem(storageKey);
    setState(IDLE);
  }, [storageKey]);

  /** Recursively poll /api/export/status/{jobId} every 500ms until done/error. */
  const poll = useCallback((jobId: string) => {
    const tick = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/export/status/${jobId}`);
        if (r.status === 404) {
          // Only reachable when resuming a persisted id: the job registry is
          // in-memory and TTL-pruned, so a job from long enough ago (or from
          // before a server restart) is gone. That's not a failure worth
          // showing — there's simply nothing left to resume.
          if (storageKey) sessionStorage.removeItem(storageKey);
          setState(IDLE);
          return;
        }
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
  }, [storageKey]);

  // Reattach the polling loop to a job that was already running when this hook
  // last mounted. Mount-only: a job started via run()/startJob() below already
  // calls poll() itself and must not be double-polled by this effect too.
  useEffect(() => {
    if (state.jobId && state.status === 'running') poll(state.jobId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
      if (!res.ok) {
        // Rejected requests come back as FastAPI validation JSON; show the field
        // and message rather than the raw body.
        const msg = formatApiError(await res.text(), `Request failed (${res.status}).`);
        setState((s) => ({ ...s, status: 'error', error: msg }));
        return;
      }
      const data = await res.json();
      if (data.job_id) {
        if (storageKey) sessionStorage.setItem(storageKey, data.job_id);
        setState((s) => ({ ...s, jobId: data.job_id }));
        poll(data.job_id);
      } else {
        // Synchronous response (e.g. dry-run) — treat as immediately done.
        if (storageKey) sessionStorage.removeItem(storageKey);
        setState((s) => ({ ...s, status: 'done', result: data }));
      }
    } catch (e) {
      setState((s) => ({ ...s, status: 'error', error: String(e) }));
    }
  }, [poll, storageKey]);

  /** Start a real COCO export. Returns once the job is queued (progress via state). */
  const start = useCallback((payload: unknown) => run('/api/export/coco', payload), [run]);

  /** Write rasterized masks into Tiled (standalone). Shares the status polling. */
  const startMaskSync = useCallback((payload: unknown) => run('/api/masks/to-tiled', payload), [run]);

  const downloadUrl =
    state.jobId && (state.result as { zip_available?: boolean } | null)?.zip_available
      ? `${API_BASE}/api/export/download/${state.jobId}`
      : null;

  return {
    state,
    start,
    startMaskSync,
    /** Start a job at an arbitrary path (e.g. /api/train/start, /api/train/infer)
     *  that returns a job_id and shares this same status-polling machinery. */
    startJob: run,
    reset,
    downloadUrl,
  };
}
