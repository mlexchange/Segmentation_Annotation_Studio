/**
 * ApplyModelPanel — the correct-and-refine loop for the currently-open image,
 * without leaving Annotate:
 *
 *   Apply to this image   → run a saved model here, import its predictions
 *   (fix the mistakes by hand)
 *   Fine-tune & apply     → continue training THAT model on the corrections,
 *                           then apply the improved model back to this image
 *
 * "Fine-tune & apply" warm-starts from the selected run's saved weights rather
 * than training from scratch (backend: `resume_from_run_id`), so each pass
 * builds on the last instead of relearning from zero. It always writes a new
 * run — the one being continued from is never modified, so a refinement that
 * makes things worse is discardable.
 *
 * Both actions save a version first, so any import is one Version History
 * restore away from being undone rather than only an in-session Ctrl+Z.
 */
import { useEffect, useRef, useState } from 'react';
import { MagicWand, Student } from '@phosphor-icons/react';
import { useTrainRuns } from '@/hooks/useTrainRuns';
import { useExportJob } from '@/hooks/useExportJob';
import { FAMILY_LABELS } from '@/components/train/RunsPanel';
import { canContinueFineTuning, fineTuningBlockedReason } from '@/lib/runCompatibility';
import type { AnnotationClass } from '@/stores/classStore';
import type { RunClass } from '@/lib/importPredictions';

const DEFAULT_REFINE_EPOCHS = 15;

interface ApplyModelPanelProps {
  hasOpenSample: boolean;
  isTiledSource: boolean;
  source: string | null;
  serverUri: string | null;
  currentSlice: number;
  /** This image's current class list — decides which runs can be continued. */
  currentClasses: AnnotationClass[];
  /** True if the current annotation state has no matching saved version yet. */
  needsSaveBeforeApply: boolean;
  /** Creates a new version from the current state; false means it failed. */
  onEnsureSaved: () => Promise<boolean>;
  /** Builds the /api/train/start `sources` + `classes` for this sample, or a
   *  message explaining why it can't (e.g. nothing annotated yet). */
  buildTrainingPayload: () => { sources: unknown[]; classes: unknown[] } | { error: string };
  onImportPredictions: (runClasses: RunClass[], slices: Record<string, unknown>) => void;
}

export default function ApplyModelPanel({
  hasOpenSample, isTiledSource, source, serverUri, currentSlice, currentClasses,
  needsSaveBeforeApply, onEnsureSaved, buildTrainingPayload, onImportPredictions,
}: ApplyModelPanelProps) {
  const { runs, invalidate: refreshRuns } = useTrainRuns();
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [refineEpochs, setRefineEpochs] = useState(DEFAULT_REFINE_EPOCHS);
  const [isSavingFirst, setIsSavingFirst] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [appliedCount, setAppliedCount] = useState<number | null>(null);

  // Scoped by `source`, unlike Train's fixed job keys: Annotate stays mounted
  // while the user switches samples (no remount, so no persistence need there),
  // but navigating away to Browse and back DOES remount this panel — scoping by
  // sample means that resume can only ever reattach to a job for the sample
  // that's open right now, not silently import a stale job's result onto
  // whatever sample happens to be open when the user comes back.
  const { state: inferJob, startJob: startInferJob, reset: resetInfer } = useExportJob(
    source ? `annotate:apply-infer:${source}` : undefined,
  );
  const { state: trainJob, startJob: startTrainJob, reset: resetTrain } = useExportJob(
    source ? `annotate:finetune:${source}` : undefined,
  );
  const importedForJobRef = useRef<string | null>(null);
  // Set when a fine-tune is running so its resulting run is applied (and
  // selected) automatically once training finishes.
  const awaitingTrainedRunRef = useRef(false);

  // Default to the newest run that can actually be CONTINUED on this image's
  // classes, falling back to the newest overall. Defaulting to "newest" alone
  // lands on a run trained for a different taxonomy whenever one happens to be
  // more recent, and every fine-tune attempt from it can only ever fail.
  // Also self-heals if the selected run disappears (deleted from the Train tab).
  useEffect(() => {
    if (runs.length === 0) {
      if (selectedRunId !== null) setSelectedRunId(null);
      return;
    }
    if (selectedRunId === null || !runs.some((r) => r.run_id === selectedRunId)) {
      const compatible = runs.find((r) => canContinueFineTuning(r.classes, currentClasses));
      setSelectedRunId((compatible ?? runs[0]).run_id);
    }
  }, [runs, selectedRunId, currentClasses]);

  // A different run or image invalidates any finished job's relevance.
  useEffect(() => {
    resetInfer();
    setLocalError(null);
    setAppliedCount(null);
    importedForJobRef.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRunId, source]);

  /** Predict on this slice with `runId` (already-saved state assumed). */
  const runInference = (runId: string) => {
    void startInferJob('/api/train/infer', {
      run_id: runId,
      kind: isTiledSource ? 'tiled' : 'local',
      source,
      server_uri: serverUri,
      slice_indices: [currentSlice],
    });
  };

  // Chain fine-tune → apply: adopt the new run, then predict with it.
  useEffect(() => {
    if (trainJob.status !== 'done' || !awaitingTrainedRunRef.current) return;
    awaitingTrainedRunRef.current = false;
    refreshRuns();
    const newRunId = typeof trainJob.result?.run_id === 'string' ? trainJob.result.run_id : null;
    if (!newRunId) {
      setLocalError('Fine-tuning finished but produced no run to apply.');
      return;
    }
    // Selecting the new run also resets the infer job via the effect above, so
    // start the prediction from here rather than racing that reset.
    setSelectedRunId(newRunId);
    runInference(newRunId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainJob.status, trainJob.result]);

  // Import predictions exactly once per completed inference job.
  useEffect(() => {
    if (inferJob.status !== 'done' || !inferJob.jobId || importedForJobRef.current === inferJob.jobId) return;
    importedForJobRef.current = inferJob.jobId;
    const runClasses = Array.isArray(inferJob.result?.classes) ? (inferJob.result!.classes as RunClass[]) : [];
    const slices = (inferJob.result?.slices ?? {}) as Record<string, unknown>;
    onImportPredictions(runClasses, slices);
    setAppliedCount(typeof inferJob.result?.n_shapes === 'number' ? inferJob.result.n_shapes : 0);
  }, [inferJob.status, inferJob.jobId, inferJob.result, onImportPredictions]);

  const selectedRun = runs.find((r) => r.run_id === selectedRunId) ?? null;
  // Applying works across taxonomies (predictions are remapped by label), but
  // CONTINUING to train needs the class lists to line up — see runCompatibility.
  const fineTuneBlocked = selectedRun
    ? fineTuningBlockedReason(selectedRun.classes, currentClasses)
    : null;

  /** Save first (so any import is undoable from Version History), then act. */
  const saveThen = async (action: () => void) => {
    setLocalError(null);
    setAppliedCount(null);
    if (needsSaveBeforeApply) {
      setIsSavingFirst(true);
      const ok = await onEnsureSaved();
      setIsSavingFirst(false);
      if (!ok) {
        setLocalError('Could not save first — try Save, then try again.');
        return;
      }
    }
    action();
  };

  const handleApply = () => {
    if (!selectedRunId || !source) return;
    resetInfer();
    void saveThen(() => runInference(selectedRunId));
  };

  const handleFineTuneAndApply = () => {
    if (!selectedRunId || !selectedRun || !source || fineTuneBlocked) return;
    const payload = buildTrainingPayload();
    if ('error' in payload) {
      setLocalError(payload.error);
      return;
    }
    // The server overrides every architecture-defining setting (arch,
    // checkpoint, LoRA shapes, patch size, tiling) from the run being resumed,
    // so only the genuinely re-tunable epoch count matters here. arch/checkpoint
    // are still echoed back because the DINOv3 schema requires them present.
    const model = selectedRun.model_family === 'dinov3_lora'
      ? {
          model_family: 'dinov3_lora' as const,
          arch: String(selectedRun.model_config.arch ?? ''),
          checkpoint: String(selectedRun.model_config.checkpoint ?? ''),
          hyperparams: { epochs: refineEpochs },
        }
      : { model_family: 'dlsia_tunet' as const, hyperparams: { epochs: refineEpochs } };

    resetInfer();
    resetTrain();
    void saveThen(() => {
      awaitingTrainedRunRef.current = true;
      void startTrainJob('/api/train/start', {
        sources: payload.sources,
        classes: payload.classes,
        model,
        resume_from_run_id: selectedRunId,
      });
    });
  };

  const busy = isSavingFirst || inferJob.status === 'running' || trainJob.status === 'running';
  const jobError = inferJob.status === 'error' ? inferJob.error : trainJob.status === 'error' ? trainJob.error : null;

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Apply a trained model</label>
      {runs.length === 0 ? (
        <p className="text-xs text-gray-400">No saved runs yet — train one in the Train tab first.</p>
      ) : (
        <>
          <select
            value={selectedRunId ?? ''}
            onChange={(e) => setSelectedRunId(e.target.value || null)}
            disabled={busy}
            className="w-full rounded-md border border-gray-300 bg-white px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
          >
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {FAMILY_LABELS[run.model_family] ?? run.model_family} — {new Date(run.created_at).toLocaleDateString()}
                {typeof run.metrics.val_miou === 'number' ? ` (mIoU ${run.metrics.val_miou.toFixed(2)})` : ''}
                {/* Marked, not hidden: these runs are still perfectly usable
                    with "Apply", just not continuable. */}
                {canContinueFineTuning(run.classes, currentClasses) ? '' : ' — other classes'}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={handleApply}
            disabled={!hasOpenSample || !selectedRunId || busy}
            title="Runs the selected model on this image and adds its predictions as editable annotations. Saves a version first, so it can be undone from Version History."
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium bg-violet-600 text-white hover:bg-violet-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <MagicWand size={16} />
            {isSavingFirst ? 'Saving…' : inferJob.status === 'running' ? 'Applying…' : 'Apply to this image'}
          </button>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={handleFineTuneAndApply}
              disabled={!hasOpenSample || !selectedRunId || busy || !!fineTuneBlocked}
              title={
                fineTuneBlocked
                ?? "Continues training the selected model on this image's current annotations (including your corrections), then applies the improved model here. Saves as a new run — the original is left untouched."
              }
              className="flex flex-1 items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium bg-violet-100 text-violet-700 hover:bg-violet-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Student size={16} />
              {trainJob.status === 'running' ? 'Fine-tuning…' : 'Fine-tune & apply'}
            </button>
            <label className="flex items-center gap-1 text-xs text-gray-500" title="Extra training epochs to run on top of the existing model">
              <input
                type="number"
                min={1}
                max={500}
                value={refineEpochs}
                onChange={(e) => setRefineEpochs(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
                disabled={busy}
                className="w-14 rounded-md border border-gray-300 bg-white px-1.5 py-1 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
              />
              ep
            </label>
          </div>

          {trainJob.status === 'running' && (
            <p className="text-xs text-gray-500">
              Fine-tuning: {trainJob.phase || 'working'}…
              {trainJob.total > 0 ? ` ${trainJob.done}/${trainJob.total} batches` : ''}
            </p>
          )}
          {inferJob.status === 'running' && (
            <p className="text-xs text-gray-500">
              Predicting: {inferJob.phase || 'working'}…{inferJob.total > 0 ? ` ${inferJob.done}/${inferJob.total}` : ''}
            </p>
          )}
          {/* Amber, not red: nothing has gone wrong — "Apply" still works with
              this run, only continuing its training doesn't. */}
          {fineTuneBlocked && !busy && (
            <p className="text-xs text-amber-700 break-words">{fineTuneBlocked}</p>
          )}
          {jobError && <p className="text-xs text-red-600 break-words">{jobError}</p>}
          {localError && <p className="text-xs text-red-600 break-words">{localError}</p>}
          {inferJob.status === 'done' && appliedCount !== null && (
            <p className="text-xs text-emerald-600">
              Applied {appliedCount} predicted region{appliedCount === 1 ? '' : 's'} — review and edit as needed.
            </p>
          )}
        </>
      )}
    </div>
  );
}
