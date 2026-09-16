/**
 * TrainPage — fine-tune a dlsia TUNet model on this session's annotated
 * samples, then run inference with any saved run: preview the predicted
 * masks, import them as editable annotations, or push them to Tiled.
 *
 * dlsia TUNet only — DINOv3 LoRA is deferred (see Phase 5.5 in the
 * integration plan), so there is no model-family picker here.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router';
import { v4 as uuidv4 } from 'uuid';
import { Brain } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { buildSourceKey } from '@/lib/sourceKey';
import { useTrainCapability } from '@/hooks/useTrainCapability';
import { useTrainRuns } from '@/hooks/useTrainRuns';
import { useExportJob } from '@/hooks/useExportJob';
import { useDraftSync } from '@/hooks/useDraftSync';
import { useImageSlice } from '@/hooks/useImageSlice';
import { gatherTrainingSources, listAnnotatedSourceKeys } from '@/lib/gatherTrainingSources';
import { buildModelConfig, trainConfigSignature, validateTrainConfig } from '@/lib/trainModelConfig';
import { trainDenoisePayload } from '@/lib/trainDenoiseOption';
import { isSegmentationRun } from '@/lib/runCompatibility';
import { remapPredictedShapes, type RunClass } from '@/lib/importPredictions';
import type { Shape } from '@/stores/annotationStore';
import CapabilityBanner from '@/components/train/CapabilityBanner';
import TrainingDataPanel from '@/components/train/TrainingDataPanel';
import HyperparamsPanel, { type HyperparamsState } from '@/components/train/HyperparamsPanel';
import JobProgressBar from '@/components/train/JobProgressBar';
import RunsPanel from '@/components/train/RunsPanel';
import TrainDenoiseToggle from '@/components/train/TrainDenoiseToggle';
import InferencePanel from '@/components/train/InferencePanel';

const DEFAULT_HYPERPARAMS: HyperparamsState = {
  // 512: tile count scales with 1/size², so a bigger window means far fewer
  // forward passes per slice and more context in each.
  epochs: 60, lr: 1e-3, batch_size: 4, image_size: 512, flip_augment: true, tiling: true,
  depth: 4, base_channels: 8, growth_rate: 1.5,
};

export default function TrainPage() {
  const { source, kind, serverUri, meta, currentSlice, renderOpts } = useDatasetStore();
  const { byImage, splitBySlice, negativeSlices, addShapes } = useAnnotationStore();
  const { classes, setClasses } = useClassStore();

  // The Annotate tab's denoise setting (Phase 2's DenoisePanel) — only ever
  // read here, and only when "Train on denoised input" is ticked below.
  const denoise = useDatasetStore((s) => s.denoise);

  const { capability } = useTrainCapability();
  const { runs: allRuns, invalidate: refreshRuns, deleteRun } = useTrainRuns();
  // This tab's Runs/Inference card is segmentation-only (it imports predicted
  // SHAPES) — a saved denoiser run has no class list to remap predictions
  // against, so it's excluded here.
  const runs = useMemo(() => allRuns.filter(isSegmentationRun), [allRuns]);
  // persistKey: survives switching to another tab and back mid-job — see
  // useExportJob's docstring. Fixed keys (not scoped to a sample) are correct
  // here: once submitted, a job is already bound to its own run_id server-side,
  // independent of anything that changes in this page afterward, and only one
  // training/probe job can run at a time regardless.
  const { state: trainJob, startJob: startTrainJob } = useExportJob('train:start');
  const { state: probeJob, startJob: startProbeJob, reset: resetProbeJob } = useExportJob('train:probe');
  // Set at handleEstimateBatch time, compared against the current config when
  // the probe finishes — see the adoption effect below.
  const probeConfigSignature = useRef<string | null>(null);

  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());
  const [hyperparams, setHyperparams] = useState<HyperparamsState>(DEFAULT_HYPERPARAMS);
  const [runName, setRunName] = useState('');
  // Off by default: this one changes what the model LEARNS, not just what's on
  // screen, so it has to be asked for explicitly.
  const [trainOnDenoised, setTrainOnDenoised] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [dataError, setDataError] = useState<string | null>(null);

  // "Run inference" from Annotate/Browse jumps straight to the Runs/Inference
  // card via ?focus=infer, since that's the section they actually asked for.
  const [searchParams] = useSearchParams();
  const inferenceSectionRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (searchParams.get('focus') === 'infer') {
      inferenceSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isTiledSource = kind === 'tiled';
  const sourceKey = source && kind ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri) : null;
  // "Import as annotations" below writes shapes (and possibly new classes)
  // straight into the store for this sourceKey — without this, nothing here
  // autosaves them: AnnotatePage is the only OTHER place that mounts
  // useDraftSync, and routes render exactly one page at a time, so navigating
  // to Annotate afterward would establish a "clean" baseline that already
  // includes the import, and the debounced PUT would never fire. This closes
  // that gap the same way every edit in Annotate is already persisted.
  useDraftSync(sourceKey);
  const candidates = useMemo(() => listAnnotatedSourceKeys(byImage), [byImage]);

  // Auto-select the currently open sample the first time it becomes available.
  useEffect(() => {
    if (sourceKey && candidates.some((c) => c.sourceKey === sourceKey)) {
      setSelectedKeys((prev) => (prev.size === 0 ? new Set([sourceKey]) : prev));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, candidates.length]);

  const toggleKey = (key: string) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  };

  const handleHyperparamsChange = (updates: Partial<HyperparamsState>) =>
    setHyperparams((prev) => ({ ...prev, ...updates }));

  const handleStartTraining = () => {
    setDataError(null);
    let sources;
    try {
      sources = gatherTrainingSources(Array.from(selectedKeys), byImage, splitBySlice, negativeSlices);
    } catch (err) {
      setDataError(err instanceof Error ? err.message : String(err));
      return;
    }
    // Catch these here; the server would otherwise reject them with a validation
    // error that names the schema field rather than the visible control.
    const configError = validateTrainConfig(hyperparams);
    if (configError) {
      setDataError(configError);
      return;
    }
    const model = buildModelConfig(hyperparams);

    void startTrainJob('/api/train/start', {
      sources,
      classes,
      model,
      run_name: runName.trim() || null,
      // Adds nothing at all when the checkbox is off, so an un-denoised run is
      // byte-identical to what this page sent before the option existed.
      ...trainDenoisePayload(trainOnDenoised, denoise),
    });
  };

  // Refresh the runs list once a training job finishes.
  useEffect(() => {
    if (trainJob.status === 'done') refreshRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainJob.status]);

  // Shared registry — same cancel route every background job in this app uses.
  const handleCancelTrain = () => {
    if (trainJob.jobId) void fetch(`${API_BASE}/api/export/cancel/${trainJob.jobId}`, { method: 'POST' });
  };

  const handleCancelProbe = () => {
    if (probeJob.jobId) void fetch(`${API_BASE}/api/export/cancel/${probeJob.jobId}`, { method: 'POST' });
  };

  // Both Start and Estimate hit the same server-side gates (ML_LOCK, torch, dlsia)
  // and must agree on when they're blocked — checking a different subset per
  // button just means the one that under-checks fires a request the server was
  // always going to 409/503 back. `capability.busy` is the server's own ML_LOCK
  // state (catches jobs from other tabs/clients); the two job statuses cover this
  // tab's own in-flight request before that poll has caught up.
  const trainOrProbeRunning = trainJob.status === 'running' || probeJob.status === 'running';
  const deviceBusy = trainOrProbeRunning || capability.busy;
  const needsDlsia = !capability.dlsia.available;
  const startDisabled = deviceBusy || !capability.torch_available || needsDlsia;
  const estimateDisabledReason = trainOrProbeRunning
    ? 'A training run or another batch-size probe is already using the device'
    : capability.busy
      ? 'Another training or inference job is using the device'
      : !capability.torch_available
        ? 'Estimating is unavailable: torch is not installed on this server'
        : needsDlsia
          ? 'Estimating is unavailable: dlsia is not installed on this server'
          : null;

  /** Measure the largest batch size this model config fits, then adopt it.
   *  Reuses the shared export-job plumbing, so progress lines stream in as the
   *  probe steps 1, 2, 4, 8… (see backend/batch_probe.py). */
  const handleEstimateBatch = () => {
    setDataError(null);
    const configError = validateTrainConfig(hyperparams);
    if (configError) {
      setDataError(configError);
      return;
    }
    const model = buildModelConfig(hyperparams);
    probeConfigSignature.current = trainConfigSignature(hyperparams);
    void startProbeJob('/api/train/estimate-batch', {
      model,
      // Head width barely moves memory, so the probe works before any classes exist.
      n_classes: Math.max(1, classes.length || 2),
    });
  };

  // Adopt the measured value once the probe finishes — but only if the config
  // it measured is still the one that would be submitted. The user is free to
  // change patch-size/tiling while a probe is running; if they did, this
  // result describes a different memory footprint and must not silently
  // overwrite batch_size (a cancelled probe's partial measurement is excluded
  // the same way — see batch_probe.py's `cancelled` flag).
  useEffect(() => {
    const suggested = probeJob.result?.suggested_batch_size;
    const cancelled = probeJob.result?.cancelled === true;
    const currentSignature = trainConfigSignature(hyperparams);
    if (
      probeJob.status === 'done' && typeof suggested === 'number' && !cancelled
      && probeConfigSignature.current === currentSignature
    ) {
      setHyperparams((prev) => ({ ...prev, batch_size: suggested }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [probeJob.status, probeJob.result]);

  /** Latest probe line: its own error, the streaming log tail, or the summary. */
  const batchEstimateNote = useMemo(() => {
    if (probeJob.status === 'error') return probeJob.error ?? 'Could not estimate a batch size.';
    if (probeJob.status === 'running') return probeJob.log.at(-1) ?? 'Loading the model…';
    if (probeJob.status === 'done') {
      const note = probeJob.result?.note;
      return typeof note === 'string' ? note : null;
    }
    return null;
  }, [probeJob.status, probeJob.error, probeJob.log, probeJob.result]);

  const handleDeleteRun = async (runId: string) => {
    setDataError(null);
    const ok = await deleteRun(runId);
    if (!ok) {
      setDataError('Failed to delete run.');
      return;
    }
    if (selectedRunId === runId) setSelectedRunId(null);
  };

  const sliceQuery = useImageSlice(source, kind, currentSlice, renderOpts, serverUri);
  const baseImageUrl = sliceQuery.data ?? null;

  const handleImportPredictions = (runClasses: RunClass[], slices: Record<string, unknown>) => {
    if (!sourceKey) return;
    const remapped = remapPredictedShapes(runClasses, classes, slices as Record<string, Shape[]>);
    setClasses(remapped.classes);
    for (const [sliceKey, shapes] of Object.entries(remapped.slices)) {
      // Predicted shape ids are deterministic per run+slice (see infer_jobs.py),
      // so importing the same run onto the same slice twice would otherwise
      // collide with the previous import's ids — corrupting the shape list
      // (duplicate React/Konva keys) and making the canvas unresponsive.
      const freshIds = shapes.map((shape) => ({ ...shape, id: uuidv4() }));
      addShapes(sourceKey, Number(sliceKey), freshIds);
    }
  };

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-4 overflow-y-auto p-6 text-slate-100">
      <div className="flex items-center gap-2">
        <Brain size={22} className="text-sky-400" />
        <h1 className="text-lg font-semibold">Train</h1>
      </div>

      <CapabilityBanner capability={capability} />

      <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4 space-y-4">
        <TrainingDataPanel candidates={candidates} selected={selectedKeys} onToggle={toggleKey} />

        <HyperparamsPanel
          values={hyperparams} onChange={handleHyperparamsChange}
          runName={runName} onRunNameChange={setRunName}
          onEstimateBatch={handleEstimateBatch}
          onCancelEstimate={handleCancelProbe}
          estimatingBatch={probeJob.status === 'running'}
          batchEstimateNote={batchEstimateNote}
          estimateDisabledReason={estimateDisabledReason}
        />

        <TrainDenoiseToggle
          checked={trainOnDenoised}
          onChange={setTrainOnDenoised}
          disabled={trainJob.status === 'running'}
        />

        {dataError && <p className="text-sm text-red-400">{dataError}</p>}

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleStartTraining}
            disabled={startDisabled}
            className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white hover:bg-sky-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {trainJob.status === 'running' ? 'Training…' : 'Start training'}
          </button>
          {trainJob.status === 'running' && (
            <button
              type="button" onClick={handleCancelTrain}
              className="px-3 py-2 text-sm rounded-md border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
            >
              Cancel
            </button>
          )}
        </div>

        <JobProgressBar job={trainJob} unit="batches" />
        {trainJob.status === 'done' && (
          <p className="text-sm text-emerald-300">
            Saved run <span className="font-mono">{String(trainJob.result?.run_id ?? '')}</span>
            {typeof trainJob.result?.val_miou === 'number' && ` — val mIoU ${trainJob.result.val_miou.toFixed(3)}`}.
          </p>
        )}
      </div>

      <div ref={inferenceSectionRef} className="rounded-lg border border-slate-700 bg-slate-800/60 p-4 space-y-4">
        <RunsPanel runs={runs} selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} onDeleteRun={handleDeleteRun} />
        <InferencePanel
          selectedRunId={selectedRunId}
          hasOpenSample={!!source && !!meta}
          isTiledSource={isTiledSource}
          source={source}
          serverUri={serverUri}
          currentSlice={currentSlice}
          nSlices={meta?.nSlices ?? 1}
          baseImageUrl={baseImageUrl}
          onImportPredictions={handleImportPredictions}
        />
      </div>
    </div>
  );
}
