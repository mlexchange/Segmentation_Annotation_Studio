/**
 * DenoiseBakeModal — write the current denoise settings out as a new dataset.
 *
 * The Annotate preview is deliberately non-destructive: it changes what you see
 * and what the intensity tools act on, but exports still use the original
 * pixels. This is the other half — it produces a first-class dataset you can
 * open, annotate and export, sitting next to its source in Browse.
 *
 * The job runs on the shared export-job registry, so it reports progress and
 * cancels through the same routes everything else long-running here uses.
 */
import { useCallback, useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { X, Warning, CheckCircle, CircleNotch } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import type { DenoiseOpts } from '@/stores/datasetStore';

interface JobState {
  state: 'pending' | 'running' | 'done' | 'error';
  phase: string;
  done: number;
  total: number;
  error: string | null;
  result: { path?: string; n_slices?: number; cancelled?: boolean } | null;
}

export interface DenoiseBakeModalProps {
  open: boolean;
  onClose: () => void;
  source: string;
  serverUri: string | null;
  denoise: DenoiseOpts;
  /** Slice count, so the estimate is honest about how long this will take. */
  nSlices: number;
  /** Method label for display (e.g. "Total variation"). */
  methodLabel: string;
}

/** `browse/foo` -> `browse/foo_denoised`, matching the backend default. */
function defaultTarget(source: string): string {
  return `${source.replace(/\/+$/, '')}_denoised`;
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') return body.detail;
    return JSON.stringify(body);
  } catch {
    return `Request failed (${res.status})`;
  }
}

export default function DenoiseBakeModal({
  open, onClose, source, serverUri, denoise, nSlices, methodLabel,
}: DenoiseBakeModalProps) {
  const queryClient = useQueryClient();
  const [target, setTarget] = useState(() => defaultTarget(source));
  const [description, setDescription] = useState('');
  const [job, setJob] = useState<JobState | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setTarget(defaultTarget(source));
      setJob(null);
      setJobId(null);
      setError(null);
    }
  }, [open, source]);

  const start = useCallback(async () => {
    setError(null);
    setJob({ state: 'pending', phase: 'Starting', done: 0, total: nSlices, error: null, result: null });
    try {
      const res = await fetch(`${API_BASE}/api/denoise/bake`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          server_uri: serverUri,
          method: denoise.method,
          strength: denoise.strength,
          target_path: target,
          description,
        }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const { job_id: id } = await res.json();
      setJobId(id);

      for (;;) {
        await new Promise((r) => setTimeout(r, 1000));
        const s = await fetch(`${API_BASE}/api/export/status/${id}`);
        if (!s.ok) throw new Error(await readError(s));
        const status: JobState = await s.json();
        setJob(status);
        if (status.state === 'error') throw new Error(status.error || 'Bake failed');
        if (status.state === 'done') break;
      }
      // A new dataset exists — let Browse pick it up without a reload.
      await queryClient.invalidateQueries({ queryKey: ['browse'] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setJob(null);
    }
  }, [source, serverUri, denoise, target, description, nSlices, queryClient]);

  const cancel = useCallback(async () => {
    if (!jobId) return;
    // Cooperative: the job stops at its next slice boundary and discards the
    // partial dataset, rather than leaving something that looks complete.
    await fetch(`${API_BASE}/api/export/cancel/${jobId}`, { method: 'POST' });
  }, [jobId]);

  if (!open) return null;

  const running = job !== null && job.state !== 'done';
  const finished = job?.state === 'done';
  const percent = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-lg rounded-lg border border-sky-800 bg-sky-950 p-5 text-sky-100 shadow-xl">
        <div className="mb-3 flex items-start justify-between">
          <div>
            <h2 className="text-sm font-semibold">Save denoised copy</h2>
            <p className="mt-1 text-xs text-sky-300/80">
              {methodLabel}, strength {denoise.strength.toFixed(2)} — applied to all{' '}
              {nSlices} slice{nSlices === 1 ? '' : 's'} and written as a new dataset.
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="text-sky-400 hover:text-sky-200">
            <X size={16} />
          </button>
        </div>

        {!running && !finished && (
          <>
            <label className="mb-1 block text-xs text-sky-300">Destination</label>
            <input
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="mb-3 w-full rounded border border-sky-800 bg-sky-900/50 px-2 py-1 text-xs"
            />
            <label className="mb-1 block text-xs text-sky-300">Tags (optional, comma-separated)</label>
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. denoised, tv"
              className="mb-4 w-full rounded border border-sky-800 bg-sky-900/50 px-2 py-1 text-xs"
            />
            <p className="mb-4 text-[11px] leading-snug text-sky-400/70">
              The original dataset is not modified. This writes a copy — on a large
              volume it can take a while, and it can be cancelled.
            </p>
            <div className="flex justify-end gap-2">
              <button type="button" onClick={onClose} className="rounded px-3 py-1.5 text-xs text-sky-300 hover:bg-sky-900">
                Cancel
              </button>
              <button
                type="button"
                onClick={start}
                disabled={!target.trim()}
                className="rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50"
              >
                Save copy
              </button>
            </div>
          </>
        )}

        {running && (
          <div className="text-xs">
            <p className="mb-2 flex items-center gap-2">
              <CircleNotch size={14} className="animate-spin" />
              {job.phase} — {job.done}/{job.total}
            </p>
            <div className="mb-4 h-2 w-full overflow-hidden rounded bg-sky-900">
              <div className="h-full bg-sky-500 transition-[width] duration-300" style={{ width: `${percent}%` }} />
            </div>
            <div className="flex justify-end">
              <button type="button" onClick={cancel} className="rounded border border-sky-700 px-3 py-1.5 text-xs hover:bg-sky-900">
                Stop
              </button>
            </div>
          </div>
        )}

        {finished && (
          <div className="text-xs">
            <p className="mb-3 flex items-center gap-2 text-emerald-300">
              <CheckCircle size={16} />
              {job.result?.cancelled
                ? 'Stopped — the partial copy was discarded.'
                : `Saved ${job.result?.n_slices ?? job.done} slices to ${job.result?.path ?? target}.`}
            </p>
            <div className="flex justify-end">
              <button type="button" onClick={onClose} className="rounded bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500">
                Done
              </button>
            </div>
          </div>
        )}

        {error && (
          <p className="mt-3 flex items-start gap-1 text-xs text-amber-300">
            <Warning size={14} className="mt-0.5 shrink-0" />
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
