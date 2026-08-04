/**
 * LoadMasksPanel — read a previously-written `<stem>__masks` Tiled sidecar
 * container back into editable annotations ("Load saved masks").
 *
 * This is the read-back half of "Write masks to Tiled" (Train tab's
 * InferencePanel): that button writes predicted (or, via the Export modal,
 * hand-annotated) masks into a hidden Tiled sidecar for downstream consumers
 * — until now there was no way back from there into Annotate. The backend
 * vectorizes the saved masks into the SAME {classes, slices} shape a live
 * inference job produces, so this reuses AnnotatePage's existing
 * `handleApplyModelPredictions` import path unchanged.
 *
 * Loaded shapes are always ADDED alongside whatever is already on canvas,
 * never replacing it — the same Tiled container can also hold hand-drawn
 * ground truth (from the Export modal's "Write masks to Tiled"), so loading
 * it back can duplicate regions you already have. A save happens first
 * (matching ApplyModelPanel's saveThen), so the load is always one Version
 * History restore away from undone, and when the sample already has shapes
 * on any of the saved slices, the button flips to an inline confirmation
 * instead of loading silently.
 */
import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Stack } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useExportJob } from '@/hooks/useExportJob';
import { getClassPalette } from '@/lib/classColors';
import type { RunClass } from '@/lib/importPredictions';

interface MasksPreview {
  available: boolean;
  path?: string;
  n_slices?: number;
  slice_indices?: number[];
  updated_at?: string | null;
  classes?: Array<{ id: number; name: string; color: string | null }>;
  error?: string;
}

interface LoadMasksPanelProps {
  source: string | null;
  serverUri: string | null;
  /** Total slices in the currently-open sample — masks saved for a source
   *  that was since re-ingested shallower can reference slices that no
   *  longer exist; those are filtered out client-side rather than crashing
   *  the import. */
  nSlices: number;
  /** True if any of the given (real) slice indices already has annotations —
   *  drives the duplicate-import confirmation. */
  hasShapesOnSlices: (indices: number[]) => boolean;
  needsSaveBeforeApply: boolean;
  onEnsureSaved: () => Promise<boolean>;
  onImportPredictions: (runClasses: RunClass[], slices: Record<string, unknown>) => void;
}

async function fetchPreview(source: string, serverUri: string | null): Promise<MasksPreview> {
  const params = new URLSearchParams({ source });
  if (serverUri) params.set('server_uri', serverUri);
  const res = await fetch(`${API_BASE}/api/masks/from-tiled/preview?${params}`);
  if (!res.ok) return { available: false, error: `Request failed (${res.status}).` };
  return res.json();
}

export default function LoadMasksPanel({
  source, serverUri, nSlices, hasShapesOnSlices, needsSaveBeforeApply, onEnsureSaved, onImportPredictions,
}: LoadMasksPanelProps) {
  const [isSavingFirst, setIsSavingFirst] = useState(false);
  const [confirmDuplicate, setConfirmDuplicate] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [loadedCount, setLoadedCount] = useState<number | null>(null);
  const [skippedInfo, setSkippedInfo] = useState<{ outOfRange: number; unreadable: number }>({ outOfRange: 0, unreadable: 0 });
  const importedForJobRef = useRef<string | null>(null);

  const preview = useQuery({
    queryKey: ['masks-from-tiled-preview', source, serverUri],
    queryFn: () => fetchPreview(source!, serverUri),
    enabled: !!source,
    staleTime: 30_000,
  });

  // Scoped per sample, like ApplyModelPanel's jobs: navigating away to Browse
  // and back remounts this panel, and resume should only ever reattach to a
  // job for the sample that's open right now.
  const { state: job, startJob, reset: resetJob } = useExportJob(
    source ? `annotate:load-masks:${source}` : undefined,
  );

  // A different sample invalidates any finished job's relevance.
  useEffect(() => {
    resetJob();
    setLocalError(null);
    setLoadedCount(null);
    setConfirmDuplicate(false);
    importedForJobRef.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // Import once per completed job.
  useEffect(() => {
    if (job.status !== 'done' || !job.jobId || importedForJobRef.current === job.jobId) return;
    importedForJobRef.current = job.jobId;
    const palette = getClassPalette();
    const rawClasses = Array.isArray(job.result?.classes) ? (job.result!.classes as Array<{ classId: number; label: string; color: string | null }>) : [];
    const runClasses: RunClass[] = rawClasses.map((c, i) => ({
      classId: c.classId,
      label: c.label,
      color: c.color ?? palette[i % palette.length],
    }));
    const rawSlices = (job.result?.slices ?? {}) as Record<string, unknown>;
    const slices: Record<string, unknown> = {};
    let outOfRange = 0;
    for (const [key, shapes] of Object.entries(rawSlices)) {
      if (Number(key) >= nSlices) { outOfRange += 1; continue; }
      slices[key] = shapes;
    }
    onImportPredictions(runClasses, slices);
    setLoadedCount(typeof job.result?.n_shapes === 'number' ? job.result.n_shapes : 0);
    const errors = Array.isArray(job.result?.errors) ? job.result!.errors as unknown[] : [];
    setSkippedInfo({ outOfRange, unreadable: errors.length });
  }, [job.status, job.jobId, job.result, onImportPredictions, nSlices]);

  const busy = isSavingFirst || job.status === 'running';

  const startLoad = () => {
    if (!source) return;
    resetJob();
    setLocalError(null);
    setLoadedCount(null);
    void (async () => {
      setLocalError(null);
      if (needsSaveBeforeApply) {
        setIsSavingFirst(true);
        const ok = await onEnsureSaved();
        setIsSavingFirst(false);
        if (!ok) {
          setLocalError('Could not save first — try Save, then try again.');
          return;
        }
      }
      void startJob('/api/masks/from-tiled', { kind: 'tiled', source, server_uri: serverUri });
    })();
  };

  const handleClick = () => {
    if (!source || !preview.data?.available) return;
    const savedIndices = preview.data.slice_indices ?? [];
    if (!confirmDuplicate && hasShapesOnSlices(savedIndices)) {
      setConfirmDuplicate(true);
      return;
    }
    setConfirmDuplicate(false);
    startLoad();
  };

  const handleCancel = () => {
    if (job.jobId) void fetch(`${API_BASE}/api/train/cancel/${job.jobId}`, { method: 'POST' });
  };

  if (!source) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Saved masks</label>

      {preview.isLoading && <p className="text-xs text-gray-400">Checking for saved masks…</p>}

      {preview.data && !preview.data.available && (
        <p className="text-xs text-gray-400">No saved masks for this sample yet — write them from the Train tab.</p>
      )}

      {preview.data?.available && (
        <>
          <p className="text-xs text-gray-500">
            {preview.data.n_slices ?? (preview.data.slice_indices ?? []).length} slice
            {(preview.data.n_slices ?? 0) === 1 ? '' : 's'} of saved masks
            {preview.data.updated_at ? ` · updated ${new Date(preview.data.updated_at).toLocaleDateString()}` : ''}
          </p>

          {confirmDuplicate ? (
            <div className="rounded-md border border-amber-300 bg-amber-50 p-2 space-y-1.5">
              <p className="text-xs text-amber-800">
                This sample already has annotations on those slices — loading adds the saved masks alongside them
                (a version is saved first, so this can be undone from Version History).
              </p>
              <div className="flex gap-1.5">
                <button
                  type="button"
                  onClick={handleClick}
                  className="flex-1 px-3 py-1.5 rounded-md text-xs font-medium bg-amber-600 text-white hover:bg-amber-500 transition-colors"
                >
                  Load anyway
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDuplicate(false)}
                  className="px-3 py-1.5 rounded-md text-xs font-medium border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={handleClick}
              disabled={busy}
              title="Reads the saved masks back as editable annotations, added alongside anything already here. Saves a version first, so it can be undone from Version History."
              className="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium bg-sky-100 text-sky-700 hover:bg-sky-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Stack size={16} />
              {isSavingFirst ? 'Saving…' : job.status === 'running' ? 'Loading…' : 'Load saved masks'}
            </button>
          )}

          {job.status === 'running' && (
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>
                Loading{job.total > 0 ? `: ${job.done}/${job.total} slices` : '…'}
              </span>
              <button type="button" onClick={handleCancel} className="text-gray-400 hover:text-gray-600 underline">
                Cancel
              </button>
            </div>
          )}
        </>
      )}

      {job.status === 'error' && <p className="text-xs text-red-600 break-words">{job.error}</p>}
      {localError && <p className="text-xs text-red-600 break-words">{localError}</p>}
      {job.status === 'done' && loadedCount !== null && (
        <p className="text-xs text-emerald-600">
          Loaded {loadedCount} region{loadedCount === 1 ? '' : 's'} from saved masks — review and edit as needed.
          {skippedInfo.outOfRange > 0 && ` (${skippedInfo.outOfRange} slice${skippedInfo.outOfRange === 1 ? '' : 's'} out of range, skipped)`}
          {skippedInfo.unreadable > 0 && ` (${skippedInfo.unreadable} slice${skippedInfo.unreadable === 1 ? '' : 's'} could not be read)`}
        </p>
      )}
    </div>
  );
}
