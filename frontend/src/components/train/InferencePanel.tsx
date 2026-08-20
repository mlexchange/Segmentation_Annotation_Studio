/**
 * InferencePanel — run a saved fine-tuned run over the current sample, with
 * an overlay preview, then import the predictions as editable annotations
 * and/or push them into Tiled as masks.
 */
import { useEffect, useMemo, useState } from 'react';
import { API_BASE } from '@/config';
import { useExportJob } from '@/hooks/useExportJob';
import { buildSliceUrl } from '@/hooks/useImageSlice';
import { useDatasetStore } from '@/stores/datasetStore';
import type { RunClass } from '@/lib/importPredictions';
import JobProgressBar from './JobProgressBar';

const MAX_SLICE_INDICES = 2000;

interface InferencePanelProps {
  selectedRunId: string | null;
  hasOpenSample: boolean;
  isTiledSource: boolean;
  source: string | null;
  serverUri: string | null;
  currentSlice: number;
  nSlices: number;
  baseImageUrl: string | null;
  onImportPredictions: (runClasses: RunClass[], slices: Record<string, unknown>) => void;
}

type Scope = 'current' | 'range' | 'all';

export default function InferencePanel({
  selectedRunId, hasOpenSample, isTiledSource, source, serverUri, currentSlice, nSlices, baseImageUrl, onImportPredictions,
}: InferencePanelProps) {
  const [scope, setScope] = useState<Scope>('current');
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(Math.max(0, nSlices - 1));
  const [previewSlice, setPreviewSlice] = useState<number | null>(null);
  const [opacity, setOpacity] = useState(0.7);
  // Set right after "Import as annotations" is clicked; cleared on a new run
  // so a stale count from a previous job never lingers. There was previously
  // no feedback at all here — the shapes DID land in the store, but nothing
  // told the user it worked (see TrainPage's useDraftSync mount for the
  // matching fix that makes the import actually persist, not just appear).
  const [importedCount, setImportedCount] = useState<number | null>(null);

  // Fixed keys: same reasoning as TrainPage's jobs — once started, an infer job
  // is bound to its own job_id independent of anything selected afterward.
  const { state: job, startJob, reset } = useExportJob('train:infer');
  const { state: writeJob, startJob: startWriteJob } = useExportJob('train:infer-write');

  useEffect(() => {
    setRangeEnd(Math.max(0, nSlices - 1));
  }, [nSlices]);

  const sliceIndices = (): number[] => {
    if (scope === 'current') return [currentSlice];
    if (scope === 'range') {
      const lo = Math.max(0, Math.min(rangeStart, rangeEnd));
      const hi = Math.min(nSlices - 1, Math.max(rangeStart, rangeEnd));
      return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
    }
    return Array.from({ length: Math.min(nSlices, MAX_SLICE_INDICES) }, (_, i) => i);
  };

  const handleRunInference = () => {
    if (!selectedRunId || !source) return;
    reset();
    setImportedCount(null);
    void startJob('/api/train/infer', {
      run_id: selectedRunId,
      kind: isTiledSource ? 'tiled' : 'local',
      source,
      server_uri: serverUri,
      slice_indices: sliceIndices(),
    });
  };

  const handleCancelInference = () => {
    if (job.jobId) void fetch(`${API_BASE}/api/train/cancel/${job.jobId}`, { method: 'POST' });
  };

  const previewSlices: number[] = Array.isArray(job.result?.preview_slices)
    ? (job.result!.preview_slices as number[])
    : [];
  const activePreviewSlice = previewSlice ?? previewSlices[0] ?? null;
  // Keyed on the job, deliberately NOT on selectedRunId: the overlay is cached
  // server-side against this job_id (which run produced it is already baked in),
  // and selectedRunId is page-local state that resets to null when the Train tab
  // remounts — requiring it here made a finished job's overlay vanish on the way
  // back to the tab, leaving just the bare base image under a "done" progress bar.
  const previewUrl = job.status === 'done' && activePreviewSlice != null && job.jobId
    ? `${API_BASE}/api/train/infer/preview/${job.jobId}/${activePreviewSlice}`
    : null;

  // The image UNDER the overlay must follow the preview-slice slider, not the
  // slice open in the app. `baseImageUrl` is bound to the open slice, so on a
  // multi-slice job the overlay advanced while the picture beneath it stayed
  // put — predictions from slice N drawn over the pixels of slice 0. Rebuilt
  // here from activePreviewSlice; the prop stays as the fallback for the
  // single-slice case and before a job has produced any preview.
  const { kind, renderOpts } = useDatasetStore();
  const basePreviewUrl = useMemo(() => {
    if (activePreviewSlice == null || activePreviewSlice === currentSlice) return baseImageUrl;
    if (!source || !kind) return baseImageUrl;
    return buildSliceUrl(source, kind, activePreviewSlice, renderOpts, serverUri);
  }, [activePreviewSlice, currentSlice, source, kind, renderOpts, serverUri, baseImageUrl]);

  const totalShapes = typeof job.result?.n_shapes === 'number' ? job.result.n_shapes : 0;

  // Gates the "start a new job" controls only. Progress/results for a job
  // already running or finished must stay visible on their own — e.g. after
  // navigating away and back, TrainPage's selectedRunId resets to null before
  // the run list settles, but a stack-wide inference started earlier keeps
  // running server-side and this hook already reattached to it (see
  // useExportJob's persistKey). Gating the whole section on selectedRunId
  // used to hide that reattached job behind "select a run", making a
  // still-running (or already-finished) job look like it vanished.
  const canStartNewJob = hasOpenSample && !!selectedRunId;
  const hasJob = job.status !== 'idle';

  return (
    <div className="space-y-3">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Inference</p>
      {!hasOpenSample && !hasJob && (
        <p className="text-xs text-slate-400">Open a sample in Browse/Annotate to run inference on it.</p>
      )}
      {hasOpenSample && !selectedRunId && !hasJob && (
        <p className="text-xs text-slate-400">Select a saved run above to enable inference.</p>
      )}

      {canStartNewJob && (
        <>
          <div className="flex flex-col gap-1.5">
            {([
              ['current', `Current slice (${currentSlice})`],
              ['range', 'Slice range'],
              ['all', `All slices (${nSlices})`],
            ] as const).map(([value, label]) => (
              <label key={value} className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input type="radio" name="infer-scope" className="accent-sky-500" checked={scope === value} onChange={() => setScope(value)} />
                {label}
              </label>
            ))}
            {scope === 'range' && (
              <div className="flex items-center gap-2 pl-5 text-xs text-slate-300">
                <input
                  type="number" min={0} max={nSlices - 1} value={rangeStart}
                  onChange={(e) => setRangeStart(Number(e.target.value))}
                  className="w-16 rounded border border-slate-600 bg-slate-900/60 px-1.5 py-1"
                />
                <span>to</span>
                <input
                  type="number" min={0} max={nSlices - 1} value={rangeEnd}
                  onChange={(e) => setRangeEnd(Number(e.target.value))}
                  className="w-16 rounded border border-slate-600 bg-slate-900/60 px-1.5 py-1"
                />
              </div>
            )}
          </div>

          <button
            type="button" onClick={handleRunInference}
            disabled={job.status === 'running'}
            className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white hover:bg-sky-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {job.status === 'running' ? 'Predicting…' : 'Run inference'}
          </button>
        </>
      )}

      {hasJob && (
        <>
          <div className="flex items-start gap-2">
            <div className="flex-1"><JobProgressBar job={job} unit="slices" /></div>
            {job.status === 'running' && (
              <button
                type="button" onClick={handleCancelInference}
                className="shrink-0 px-3 py-1.5 text-xs rounded-md border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
              >
                Cancel
              </button>
            )}
          </div>

          {job.status === 'done' && (
            <div className="space-y-2 rounded-lg border border-slate-700 bg-slate-900/40 p-2">
              <p className="text-xs text-slate-300">
                {totalShapes} predicted region{totalShapes !== 1 ? 's' : ''}
                {job.result?.cancelled === true ? ' (cancelled — partial result)' : ''}.
              </p>
              {previewSlices.length > 0 && basePreviewUrl && (
                <>
                  <div className="relative w-full overflow-hidden rounded border border-slate-700 bg-black">
                    <img src={basePreviewUrl} alt="" className="block w-full" />
                    {previewUrl && (
                      <img src={previewUrl} alt="" className="absolute inset-0 w-full" style={{ opacity }} />
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-300">
                    <span>Opacity</span>
                    <input
                      type="range" min={0} max={1} step={0.05} value={opacity}
                      onChange={(e) => setOpacity(Number(e.target.value))} className="flex-1 accent-sky-500"
                    />
                  </div>
                  {previewSlices.length > 1 && (
                    <div className="flex items-center gap-2 text-xs text-slate-300">
                      <span>Slice</span>
                      <input
                        type="range" min={0} max={previewSlices.length - 1}
                        value={Math.max(0, previewSlices.indexOf(activePreviewSlice ?? previewSlices[0]))}
                        onChange={(e) => setPreviewSlice(previewSlices[Number(e.target.value)])}
                        className="flex-1 accent-sky-500"
                      />
                      <span className="tabular-nums">{activePreviewSlice}</span>
                    </div>
                  )}
                </>
              )}
              {hasOpenSample && (
                <div className="flex flex-wrap gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => {
                      const classes = Array.isArray(job.result?.classes) ? (job.result!.classes as RunClass[]) : [];
                      const slices = (job.result?.slices ?? {}) as Record<string, unknown>;
                      onImportPredictions(classes, slices);
                      setImportedCount(totalShapes);
                    }}
                    className="px-3 py-1.5 text-xs rounded-md bg-sky-600 text-white hover:bg-sky-500 transition-colors"
                  >
                    Import as annotations
                  </button>
                  {isTiledSource && (
                    <button
                      type="button"
                      disabled={writeJob.status === 'running'}
                      onClick={() => job.jobId && void startWriteJob(`/api/train/infer/write-tiled/${job.jobId}`, {})}
                      className="px-3 py-1.5 text-xs rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-900/30 transition-colors disabled:opacity-50"
                    >
                      {writeJob.status === 'running' ? 'Writing…' : 'Write masks to Tiled'}
                    </button>
                  )}
                </div>
              )}
              {importedCount !== null && (
                <p className="text-xs text-emerald-400">
                  Imported {importedCount} region{importedCount === 1 ? '' : 's'} as annotations — they're on the
                  Annotate tab and autosaved.
                </p>
              )}
              {writeJob.status === 'done' && (
                <p className="text-xs text-emerald-400">
                  Masks saved to Tiled
                  {typeof writeJob.result?.n_slices === 'number' ? ` (${writeJob.result.n_slices} slices)` : ''} —
                  load them in Annotate anytime with "Load saved masks".
                </p>
              )}
              <JobProgressBar job={writeJob} unit="steps" />
            </div>
          )}
        </>
      )}
    </div>
  );
}
