/**
 * DenoiseBakeModal — confirm-then-run for "apply this denoise filter to every
 * slice and save the result as a new dataset".
 *
 * Distinct from the Annotate-tab preview in a way worth being explicit about in
 * the UI: the preview is display-only and leaves the source untouched, while
 * this writes real new data. The result is a sibling dataset under `browse/`
 * (not a hidden sidecar) precisely so it can be opened, annotated, trained on
 * and exported like any ingested sample.
 *
 * Follows ResetTiledModal's visual conventions, with the job-progress + cancel
 * pattern used by the Train tab's long-running jobs.
 */
import { useState } from 'react';
import { Archive, X, CircleDashed, CheckCircle } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { formatApiError } from '@/lib/apiError';
import { useExportJob } from '@/hooks/useExportJob';

/** What to bake: either a classical filter at a strength, or a trained
 *  denoiser run (`method: 'model'`, where `strength` is unused). */
export interface DenoiseBakeTarget {
  method: string;
  label: string;
  strength: number;
  /** Required when `method === 'model'`. */
  runId?: string;
}

export interface DenoiseBakeModalProps {
  source: string;
  serverUri: string | null;
  target: DenoiseBakeTarget;
  nSlices: number;
  onClose: () => void;
}

/** `browse/dataset` -> `browse/dataset_denoised` (mirrors the backend's
 *  `denoise_bake.default_target_path` — kept as a separate implementation
 *  since it only needs to pre-fill an editable text field, not enforce the
 *  contract, but it should still agree with the server in the common case). */
export function defaultTarget(source: string): string {
  const parts = source.trim().replace(/^\/+|\/+$/g, '').split('/').filter(Boolean);
  if (parts.length === 0) return '';
  parts[parts.length - 1] = `${parts[parts.length - 1]}_denoised`;
  return parts.join('/');
}

export default function DenoiseBakeModal({
  source, serverUri, target, nSlices, onClose,
}: DenoiseBakeModalProps) {
  const { method, label: methodLabel, strength, runId } = target;
  const isModel = method === 'model';
  const [targetPath, setTargetPath] = useState(() => defaultTarget(source));
  const [description, setDescription] = useState('');
  const [startError, setStartError] = useState<string | null>(null);

  // Scoped per source: navigating away and back should only ever reattach to a
  // bake for the sample that's open now.
  const { state: job, startJob } = useExportJob(`annotate:denoise-bake:${source}`);

  const running = job.status === 'running';
  const done = job.status === 'done';

  const handleStart = async () => {
    setStartError(null);
    try {
      await startJob('/api/denoise/bake', {
        source,
        server_uri: serverUri,
        method,
        strength,
        target_path: targetPath.trim() || null,
        description: description.trim(),
        ...(runId ? { run_id: runId } : {}),
      });
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleCancel = () => {
    if (job.jobId) void fetch(`${API_BASE}/api/train/cancel/${job.jobId}`, { method: 'POST' });
  };

  const errorCount = Array.isArray(job.result?.errors) ? job.result!.errors.length : 0;
  const wasCancelled = job.result?.cancelled === true;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-md flex flex-col max-h-[90vh]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <div className="flex items-center gap-2 text-sky-700 font-semibold">
            <Archive size={18} />
            Save denoised copy
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={running}
            className="text-gray-400 hover:text-gray-700 transition-colors disabled:opacity-50"
          >
            <X size={18} />
          </button>
        </div>

        <div className="overflow-y-auto px-5 py-4 space-y-3">
          {done ? (
            <div className="flex flex-col items-center gap-2 py-4 text-center">
              <CheckCircle size={32} className="text-emerald-600" />
              <p className="text-sm text-gray-700">
                Wrote {String(job.result?.n_slices ?? 0)} slice
                {job.result?.n_slices === 1 ? '' : 's'} to{' '}
                <span className="font-mono text-xs">{String(job.result?.path ?? targetPath)}</span>.
              </p>
              {wasCancelled && (
                <p className="text-xs text-amber-700">
                  Cancelled partway — only the slices written so far are in the new dataset.
                </p>
              )}
              {errorCount > 0 && (
                <p className="text-xs text-amber-700">
                  {errorCount} slice{errorCount === 1 ? '' : 's'} could not be filtered and were
                  copied through unchanged.
                </p>
              )}
              <p className="text-xs text-gray-500">
                Open it from Browse to annotate it. Its annotations are separate from the
                original's.
              </p>
            </div>
          ) : (
            <>
              <p className="text-sm text-gray-700">
                Applies <strong>{methodLabel}</strong>
                {isModel ? '' : ` at ${Math.round(strength * 100)}% strength`} to all {nSlices}{' '}
                slice{nSlices === 1 ? '' : 's'} and saves the result as a new dataset. The original
                is never modified.
              </p>
              {isModel && (
                <p className="text-[11px] leading-snug text-amber-700">
                  A trained denoiser writes 8-bit output — the scale it actually works in — rather
                  than the source's higher bit depth, which it never had access to. It also needs
                  the GPU for the whole run, so training and inference can't run at the same time.
                </p>
              )}

              <label className="block space-y-1">
                <span className="text-xs font-medium text-gray-600">New dataset path</span>
                <input
                  type="text"
                  value={targetPath}
                  onChange={(e) => setTargetPath(e.target.value)}
                  disabled={running}
                  className="w-full rounded-md border border-gray-300 px-2 py-1.5 text-xs font-mono text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
                />
              </label>

              <label className="block space-y-1">
                <span className="text-xs font-medium text-gray-600">
                  Tags <span className="text-gray-400">(optional, comma-separated)</span>
                </span>
                <input
                  type="text"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  disabled={running}
                  placeholder="e.g. denoised, sand, glass"
                  className="w-full rounded-md border border-gray-300 px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
                />
                <span className="block text-[11px] text-gray-400">
                  Searchable in Browse, and pre-created as annotation classes.
                </span>
              </label>

              <p className="text-[11px] text-gray-500">
                This can take a while on a large stack — roughly the per-slice filter cost times{' '}
                {nSlices}. It runs in the background and can be cancelled.
              </p>

              {running && (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <span className="capitalize">{job.phase || 'working'}…</span>
                    {job.total > 0 && (
                      <span className="tabular-nums">
                        {job.done}/{job.total} slices
                      </span>
                    )}
                  </div>
                  <div className="h-1.5 w-full rounded bg-gray-200 overflow-hidden">
                    <div
                      className="h-full bg-sky-500 transition-all"
                      style={{
                        width: `${job.total > 0 ? Math.max(3, Math.round((job.done / job.total) * 100)) : 5}%`,
                      }}
                    />
                  </div>
                </div>
              )}

              {job.status === 'error' && (
                <p className="text-sm text-red-600 break-words">{job.error}</p>
              )}
              {startError && <p className="text-sm text-red-600 break-words">{startError}</p>}
            </>
          )}
        </div>

        <div className="flex justify-end gap-2 px-5 py-4 border-t border-gray-200 bg-gray-50 rounded-b-xl">
          {done ? (
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white font-medium hover:bg-sky-700 transition-colors"
            >
              Close
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={running ? handleCancel : onClose}
                className="px-4 py-2 text-sm rounded-md border border-gray-200 text-gray-700 hover:bg-gray-100 transition-colors"
              >
                {running ? 'Cancel job' : 'Cancel'}
              </button>
              <button
                type="button"
                onClick={handleStart}
                disabled={running || !targetPath.trim()}
                className="flex items-center gap-2 px-4 py-2 text-sm rounded-md bg-sky-600 text-white font-medium hover:bg-sky-700 transition-colors disabled:opacity-60"
              >
                {running ? (
                  <>
                    <CircleDashed size={16} className="animate-spin" />
                    Denoising…
                  </>
                ) : (
                  <>
                    <Archive size={16} />
                    Denoise &amp; save
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
