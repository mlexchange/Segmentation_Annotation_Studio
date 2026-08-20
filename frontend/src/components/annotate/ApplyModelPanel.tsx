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
 *
 * "Fine-tune & apply" is a training entry point, so it carries the same
 * opt-in "Train on denoised input" checkbox as the Train tab — without it, a
 * fine-tune would silently drop the setting, because the server records
 * `denoise` from the REQUEST even on a resume (unlike the architecture
 * settings, which it inherits from the parent run). "Apply" takes no such
 * option: inference reapplies whatever the run itself recorded.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { MagicWand, Student } from '@phosphor-icons/react';
import { useTrainRuns } from '@/hooks/useTrainRuns';
import { useExportJob } from '@/hooks/useExportJob';
import { useTrainCapability } from '@/hooks/useTrainCapability';
import { useDenoiseStore } from '@/stores/denoiseStore';
import { FAMILY_LABELS } from '@/components/train/RunsPanel';
import TrainDenoiseToggle from '@/components/train/TrainDenoiseToggle';
import { canContinueFineTuning, fineTuningBlockedReason, isSegmentationRun } from '@/lib/runCompatibility';
import { denoiseMethodLabel, trainDenoisePayload } from '@/lib/trainDenoiseOption';
import type { AnnotationClass } from '@/stores/classStore';
import type { RunClass } from '@/lib/importPredictions';
import type { TrainingSourceItem } from '@/lib/gatherTrainingSources';

const DEFAULT_REFINE_EPOCHS = 15;

/** Which of this sample's annotated slices "Fine-tune & apply" trains on.
 *  Mirrors InferencePanel's inference-scope selector (current/range/all). */
type TrainScope = 'current' | 'range' | 'all';

interface ApplyModelPanelProps {
  hasOpenSample: boolean;
  isTiledSource: boolean;
  source: string | null;
  serverUri: string | null;
  currentSlice: number;
  /** Total slices in the currently-open sample — bounds the fine-tune scope's
   *  range inputs and labels "All slices (N)". */
  nSlices: number;
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
  hasOpenSample, isTiledSource, source, serverUri, currentSlice, nSlices, currentClasses,
  needsSaveBeforeApply, onEnsureSaved, buildTrainingPayload, onImportPredictions,
}: ApplyModelPanelProps) {
  const { runs: allRuns, invalidate: refreshRuns } = useTrainRuns();
  const { capability } = useTrainCapability();
  // Read-only here: the Denoise panel above owns this setting.
  const denoise = useDenoiseStore((s) => s.denoise);
  // Apply/fine-tune here both key off a class list and write predicted SHAPES
  // back onto the canvas — neither makes sense for a saved denoiser run (see
  // the Learned Denoiser panel's own run picker for that flow), so exclude
  // them from this picker the same way TrainPage's does.
  const runs = useMemo(() => allRuns.filter(isSegmentationRun), [allRuns]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [refineEpochs, setRefineEpochs] = useState(DEFAULT_REFINE_EPOCHS);
  const [isSavingFirst, setIsSavingFirst] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [appliedCount, setAppliedCount] = useState<number | null>(null);
  // Default 'all': preserves the pre-existing behavior (fine-tune on every
  // annotated slice of this sample) for anyone who never touches this control.
  const [trainScope, setTrainScope] = useState<TrainScope>('all');
  // Off by default, exactly as on the Train tab — fine-tuning is training, and
  // baking a filter into the model's input has to be asked for explicitly.
  const [trainOnDenoised, setTrainOnDenoised] = useState(false);
  const [trainRangeStart, setTrainRangeStart] = useState(0);
  const [trainRangeEnd, setTrainRangeEnd] = useState(Math.max(0, nSlices - 1));

  useEffect(() => {
    setTrainRangeEnd(Math.max(0, nSlices - 1));
  }, [nSlices]);

  /** null = 'all' (no filtering — use every annotated slice, current behavior);
   *  otherwise the concrete list of slice indices to restrict training to. */
  const trainSliceIndices = (): number[] | null => {
    if (trainScope === 'all') return null;
    if (trainScope === 'current') return [currentSlice];
    const lo = Math.max(0, Math.min(trainRangeStart, trainRangeEnd));
    const hi = Math.min(nSlices - 1, Math.max(trainRangeStart, trainRangeEnd));
    return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
  };

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
  // Input denoising baked into the selected run at training time. Surfaced
  // read-only: inference reapplies it off the run itself, so there is no
  // choice to offer here — only the fact of it, which is otherwise invisible
  // and would make two identical-looking runs behave differently.
  const selectedRunDenoise = selectedRun?.denoise ?? null;

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

    // buildTrainingPayload() always gathers every annotated slice of THIS
    // sample (there's exactly one source item, this image); when the user has
    // narrowed the scope to a single slice or a range, filter it down here
    // rather than changing what buildTrainingPayload itself means elsewhere.
    let sources = payload.sources as TrainingSourceItem[];
    const scopedIndices = trainSliceIndices();
    if (scopedIndices !== null) {
      const allowed = new Set(scopedIndices.map(String));
      const item = sources[0];
      const scopedSlices = Object.fromEntries(
        Object.entries(item.slices).filter(([sliceKey]) => allowed.has(sliceKey)),
      );
      const scopedShapeCount = Object.values(scopedSlices).reduce((n, shapes) => n + shapes.length, 0);
      if (scopedShapeCount === 0) {
        setLocalError(
          scopedIndices.length === 1
            ? `No annotations on slice ${scopedIndices[0]} — choose a different scope or annotate here first.`
            : `No annotations on slices ${scopedIndices[0]}–${scopedIndices[scopedIndices.length - 1]} — choose a different scope.`,
        );
        return;
      }
      sources = [{
        ...item,
        slices: scopedSlices,
        split_by_slice: Object.fromEntries(
          Object.entries(item.split_by_slice).filter(([sliceKey]) => allowed.has(sliceKey)),
        ),
        negative_slices: item.negative_slices.filter((sliceKey) => allowed.has(sliceKey)),
      }];
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
        sources,
        classes: payload.classes,
        model,
        resume_from_run_id: selectedRunId,
        // Unlike the architecture settings above, the server does NOT inherit
        // this from the run being resumed — it records whatever the request
        // carries. Omitted entirely when the checkbox is off, which keeps a
        // plain fine-tune identical to what this panel sent before.
        ...trainDenoisePayload(trainOnDenoised, denoise),
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

          {selectedRunDenoise && (
            <p className="text-[11px] leading-snug text-gray-500">
              Trained on{' '}
              {denoiseMethodLabel(selectedRunDenoise.method, capability.denoise.methods)}-denoised
              input ({Math.round(selectedRunDenoise.strength * 100)}%). Applying it reapplies the
              same filter automatically — nothing to set here.
            </p>
          )}

          <div className="flex flex-col gap-1 pt-0.5">
            <span className="text-xs text-gray-400">Fine-tune on</span>
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {([
                ['current', `This slice (${currentSlice})`],
                ['range', 'Slice range'],
                ['all', `All annotated slices`],
              ] as const).map(([value, label]) => (
                <label key={value} className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer">
                  <input
                    type="radio" name="finetune-scope" className="accent-violet-600"
                    checked={trainScope === value} disabled={busy}
                    onChange={() => setTrainScope(value)}
                  />
                  {label}
                </label>
              ))}
            </div>
            {trainScope === 'range' && (
              <div className="flex items-center gap-2 pl-5 text-xs text-gray-600">
                <input
                  type="number" min={0} max={nSlices - 1} value={trainRangeStart} disabled={busy}
                  onChange={(e) => setTrainRangeStart(Number(e.target.value))}
                  className="w-16 rounded border border-gray-300 bg-white px-1.5 py-1 disabled:opacity-50"
                />
                <span>to</span>
                <input
                  type="number" min={0} max={nSlices - 1} value={trainRangeEnd} disabled={busy}
                  onChange={(e) => setTrainRangeEnd(Number(e.target.value))}
                  className="w-16 rounded border border-gray-300 bg-white px-1.5 py-1 disabled:opacity-50"
                />
              </div>
            )}
          </div>

          <TrainDenoiseToggle
            checked={trainOnDenoised}
            onChange={setTrainOnDenoised}
            disabled={busy}
            variant="light"
          />

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={handleFineTuneAndApply}
              disabled={!hasOpenSample || !selectedRunId || busy || !!fineTuneBlocked}
              title={
                fineTuneBlocked
                ?? "Continues training the selected model on this image's annotations (including your corrections) within the scope selected below, then applies the improved model here. Saves as a new run — the original is left untouched."
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
