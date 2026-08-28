/**
 * AnnotatePage — react-konva canvas workspace with sidebar tools.
 */
import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router';
import { DownloadSimple, FloppyDisk, ClockCounterClockwise, CircleDashed, ChartBar } from '@phosphor-icons/react';
import { cn } from '@/lib/utils';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { useDraftSync } from '@/hooks/useDraftSync';
import { clearHistory } from '@/hooks/editHistory';
import { useGuideLoad } from '@/hooks/useGuideSync';
import { useSave, type VersionPayload } from '@/hooks/useSave';
import { buildSourceKey } from '@/lib/sourceKey';
import { API_BASE } from '@/config';
import { useKeybinds } from '@/hooks/useKeybinds';
import type { ColormapName } from '@/lib/colormaps';
import Toolbar from '@/components/annotate/Toolbar';
import ClassManager from '@/components/annotate/ClassManager';
import LayersPanel from '@/components/annotate/LayersPanel';
import DisplayControls from '@/components/annotate/DisplayControls';
import DenoisePanel from '@/components/annotate/DenoisePanel';
import DenoiseBakeModal from '@/components/annotate/DenoiseBakeModal';
import SliceNavigator from '@/components/annotate/SliceNavigator';
import MaskToolsPanel from '@/components/annotate/MaskToolsPanel';
import MeasurementPanel from '@/components/annotate/MeasurementPanel';
import FeatureChannelsPanel from '@/components/annotate/FeatureChannelsPanel';
import PixelClassifierPanel from '@/components/annotate/PixelClassifierPanel';
import AnnotationCanvas from '@/components/annotate/AnnotationCanvas';
import { useFeatureChannels } from '@/hooks/useFeatureChannels';
import { usePixelClassifier } from '@/hooks/usePixelClassifier';
import { useFeatureManifold } from '@/hooks/useFeatureManifold';
import { loadLabelPng, labelMapToPolygonShapes } from '@/lib/pixelClf';
import { ipredRunCommitUrl } from '@/lib/ipredApi';
import type { Shape } from '@/stores/annotationStore';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import DownloadModal from '@/components/annotate/DownloadModal';
import InsightsModal from '@/components/annotate/InsightsModal';
import VersionHistoryModal from '@/components/annotate/VersionHistoryModal';
import VersionPreviewBar from '@/components/annotate/VersionPreviewBar';
import PerfOverlay from '@/components/annotate/PerfOverlay';
import type { SamplerFit } from '@/components/annotate/AnnotationCanvas';
import { initPerf } from '@/lib/perf';
import SaveModal from '@/components/annotate/SaveModal';
import type { SaveDraftPayload } from '@/hooks/useSave';

type AnnotateStage = 'draw' | 'assist' | 'predict';

const STAGES: { id: AnnotateStage; label: string }[] = [
  { id: 'draw', label: 'Draw' },
  { id: 'assist', label: 'Assist' },
  { id: 'predict', label: 'Predict' },
];

function StageSwitcher({ stage, onChange }: { stage: AnnotateStage; onChange: (s: AnnotateStage) => void }) {
  return (
    <div className="flex rounded-md border border-gray-200 bg-gray-50 p-0.5" role="tablist" aria-label="Annotate stage">
      {STAGES.map((s) => (
        <button
          key={s.id}
          type="button"
          role="tab"
          aria-selected={stage === s.id}
          onClick={() => onChange(s.id)}
          className={cn(
            'flex-1 rounded px-2 py-1 text-xs font-medium transition-colors',
            stage === s.id ? 'bg-white text-sky-800 shadow-sm' : 'text-gray-500 hover:text-gray-800',
          )}
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}

/** Renders the annotation workspace: tool sidebar, canvas, and save/version/export flows. */
export default function AnnotatePage() {
  const navigate = useNavigate();
  const { source, kind, serverUri, meta, denoise, setDenoise } = useDatasetStore();
  const { removeShapes, addShapes, addShapesAcrossSlices, byImage } = useAnnotationStore();
  const { selectedShapeIds, setSelectedShapeId, fillOpacity, setFillOpacity } = useToolStore();
  const { classes } = useClassStore();

  const [activeClassId, setActiveClassId] = useState<number | null>(null);
  const [activeBrushShapeId, setActiveBrushShapeId] = useState<string | null>(null);

  /** Sets the active class and clears any in-progress brush instance. */
  const handleActivateClass = useCallback((classId: number) => {
    setActiveClassId(classId);
    setActiveBrushShapeId(null);
  }, []);

  /** Resets the active brush instance after a class is deleted. */
  const handleClassDeleted = useCallback((_deletedClassId: number) => {
    setActiveBrushShapeId(null);
  }, []);

  useEffect(() => {
    if (activeClassId !== null) return;
    if (classes.length > 0) setActiveClassId(classes[0].classId);
  }, [classes, activeClassId]);

  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  // Min/max levels window (0–255) + histogram of the current slice (client-side).
  const [levelsLo, setLevelsLo] = useState(0);
  const [levelsHi, setLevelsHi] = useState(255);
  const [histogramBins, setHistogramBins] = useState<number[] | null>(null);
  // Display-only false-color map + gamma.
  const [colormap, setColormap] = useState<ColormapName>('gray');
  const [gamma, setGamma] = useState(1);
  // Display-only nonlinear preprocessors (Gaussian blur / adaptive CLAHE / Sharpen).
  const [clahe, setClahe] = useState(false);
  const [sharpen, setSharpen] = useState(false);
  const [blur, setBlur] = useState(0);
  const [bakeOpen, setBakeOpen] = useState(false);
  // Working resolution for the drawing tools (1x, 2x, 4x). Annotation coordinates
  // stay native; upscaling only buys sub-pixel precision on small features.
  const [upscale, setUpscale] = useState(1);
  // Each level costs 4x the pixels in the base canvas AND in every tool field, so
  // cap the offer at what this slice can afford (the canvas enforces the same limit).
  // Bundled for the Toolbar's threshold band picker, which remaps the (base-space)
  // histogram into displayed space so the plot matches the image and the band.
  const thresholdDisplay = useMemo(
    () => ({ brightness, contrast, levelsLo, levelsHi, gamma }),
    [brightness, contrast, levelsLo, levelsHi, gamma],
  );
  const maxUpscale = useMemo(() => {
    if (!meta) return 4;
    const px = meta.width * meta.height;
    return px * 16 <= 64e6 ? 4 : px * 4 <= 64e6 ? 2 : 1;
  }, [meta]);
  const [showDownload, setShowDownload] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  // Region to zoom to + highlight on the canvas (from an Insights QA flag).
  const [focusRegion, setFocusRegion] = useState<{ x: number; y: number; w: number; h: number; nonce: number } | null>(null);
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveModalPayload, setSaveModalPayload] = useState<SaveDraftPayload | null>(null);

  // Version preview ("time-travel") state — non-destructive.
  const [previewVersion, setPreviewVersion] = useState<number | null>(null);
  const [previewPayload, setPreviewPayload] = useState<VersionPayload | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;

  // Dev-only timing HUD (?perf=1). Read once — the flag is fixed for the session.
  const [perfOn] = useState(() => initPerf());
  const [stage, setStage] = useState<AnnotateStage>('draw');

  // Latest Sampler lasso fit — transient UI state, not persisted: only the band
  // and blur it applies are durable.
  const [samplerFit, setSamplerFit] = useState<SamplerFit | null>(null);
  /** Put back the band and blur that were in force before the last fit. */
  const revertSamplerFit = useCallback(() => {
    if (!samplerFit) return;
    useToolStore.getState().setThresholdBand(samplerFit.previousBand[0], samplerFit.previousBand[1]);
    setBlur(samplerFit.previousBlur);
    setSamplerFit(null);
  }, [samplerFit]);

  // Crash-recovery autosave (local draft only, no Tiled sync)
  useDraftSync(sourceKey);
  // Undo/redo is per-sample: reset the region history + class-delete journal on switch.
  useEffect(() => { clearHistory(); }, [sourceKey]);
  // Load the dataset's annotation guide (read-only) for class suggestions/examples.
  useGuideLoad(sourceKey);

  // Explicit versioned save
  const { isDirty, isSaving, lastSavedAt, save, buildSavePayload, saveSummary, versions, fetchVersionPayload, restoreVersion } = useSave(sourceKey);

  const { currentSlice, setSlice } = useDatasetStore();

  // Shapes on the current slice — shown in the perf HUD, since it is the variable
  // that drives most of the costs it reports. Only subscribed when the HUD is on.
  const currentShapeCount = useAnnotationStore((s) =>
    perfOn && sourceKey ? (s.byImage[sourceKey]?.[String(currentSlice)]?.length ?? 0) : 0,
  );

  /** From an Insights QA flag: jump to its slice and zoom/highlight its region. */
  const handleInsightFocus = useCallback((slice: number, bbox?: { x: number; y: number; w: number; h: number }) => {
    setSlice(slice);
    setFocusRegion(bbox ? { ...bbox, nonce: Date.now() } : null);
  }, [setSlice]);

  const sliceShapes = sourceKey ? (byImage[sourceKey]?.[String(currentSlice)] ?? []) : [];

  const features = useFeatureChannels({
    source,
    kind,
    sliceIndex: currentSlice,
    serverUri,
  });

  const clf = usePixelClassifier({
    featureJobId: features.job?.jobId ?? null,
    // Sample/composition identity only — NOT the slice, so a trained model
    // survives a slice change and can be applied to whichever slice is on
    // screen (see usePixelClassifier's two reset effects for the split).
    resetKey: sourceKey,
    source,
    kind,
    serverUri,
    sliceIndex: currentSlice,
    onFeatureJobExpired: features.invalidateJob,
    onFeatureReady: (info) => features.adoptFeatureBank(info),
  });

  const manifold = useFeatureManifold({
    featureJobId: features.job?.jobId ?? null,
    onFeatureJobExpired: features.invalidateJob,
  });

  // ---- Batch iPred: multi-slice train + whole-volume apply ----
  const [trainAcrossSlices, setTrainAcrossSlices] = useState(false);
  const [commitClassIds, setCommitClassIds] = useState<number[]>([]);
  const [volumeApplyCommitting, setVolumeApplyCommitting] = useState(false);

  // Every non-empty slice of this sample — the pool for "train across all
  // annotated slices" and the source of truth for `annotatedSliceCount`.
  const annotatedSlices = useMemo(() => {
    const slices = sourceKey ? (byImage[sourceKey] ?? {}) : {};
    const out: Record<number, Shape[]> = {};
    for (const [key, shapes] of Object.entries(slices)) {
      if (shapes.length > 0) out[Number(key)] = shapes;
    }
    return out;
  }, [sourceKey, byImage]);
  const annotatedSliceCount = Object.keys(annotatedSlices).length;
  const totalSliceCount = meta?.nSlices ?? 1;

  // Keep the commit class filter in sync with whichever model is current —
  // default to "all classes" each time a (re)train finishes.
  useEffect(() => {
    setCommitClassIds(clf.model?.classIds ?? []);
  }, [clf.model]);

  const toggleCommitClassId = useCallback((classId: number) => {
    setCommitClassIds((prev) =>
      prev.includes(classId) ? prev.filter((c) => c !== classId) : [...prev, classId],
    );
  }, []);

  const handleClfTrain = useCallback(() => {
    if (trainAcrossSlices) {
      void clf.trainAcrossSlices(annotatedSlices);
    } else {
      void clf.train(sliceShapes);
    }
  }, [clf, sliceShapes, trainAcrossSlices, annotatedSlices]);

  const handleClfPredict = useCallback(() => {
    void clf.predict(sliceShapes);
  }, [clf, sliceShapes]);

  const handleClfCommit = useCallback(async () => {
    if (!sourceKey || !clf.commitUrl || !clf.model) return;
    try {
      const { data, width, height } = await loadLabelPng(clf.commitUrl);
      const shapes = labelMapToPolygonShapes(data, width, height, commitClassIds, {
        minRegion: 64,
        preserveShapes: sliceShapes,
        origin: 'predicted',
      });
      if (shapes.length) addShapes(sourceKey, currentSlice, shapes);
      clf.dismiss();
    } catch {
      /* keep overlay; error surfaces via clf.error if needed */
    }
  }, [sourceKey, clf, addShapes, currentSlice, sliceShapes, commitClassIds]);

  const handleApplyToVolume = useCallback(() => {
    const sliceIndices = Array.from({ length: totalSliceCount }, (_, i) => i);
    void clf.applyAcrossVolume(sliceIndices);
  }, [clf, totalSliceCount]);

  const handleCancelVolumeApply = useCallback(async () => {
    const jobId = clf.volumeApplyJob.jobId;
    if (!jobId) return;
    // Cooperative: the job stops at its next slice boundary (see DenoiseBakeModal
    // for the same pattern against the same /api/export/cancel/{id} route).
    await fetch(`${API_BASE}/api/export/cancel/${jobId}`, { method: 'POST' });
  }, [clf.volumeApplyJob.jobId]);

  // Fetch + vectorize each completed slice's commit PNG client-side, reusing the
  // same tracer as the single-slice commit — see Phase 4.5's plan for why this
  // stays client-side rather than porting polygon tracing to the ipred service.
  const handleCommitVolumeApply = useCallback(async () => {
    if (!sourceKey) return;
    const result = clf.volumeApplyJob.result as { runs?: Record<string, string> } | null;
    const runs = result?.runs;
    if (!runs || commitClassIds.length === 0) return;
    setVolumeApplyCommitting(true);
    try {
      const bySlice: Record<number, Shape[]> = {};
      let processed = 0;
      for (const [sliceKey, runId] of Object.entries(runs)) {
        const sliceIdx = Number(sliceKey);
        const { data, width, height } = await loadLabelPng(ipredRunCommitUrl(runId));
        const existing = byImage[sourceKey]?.[sliceKey] ?? [];
        const shapes = labelMapToPolygonShapes(data, width, height, commitClassIds, {
          minRegion: 64,
          preserveShapes: existing,
          origin: 'predicted',
        });
        if (shapes.length) bySlice[sliceIdx] = shapes;
        processed += 1;
        // Yield every few slices so vectorizing a large volume doesn't freeze the tab.
        if (processed % 4 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (Object.keys(bySlice).length) addShapesAcrossSlices(sourceKey, bySlice);
      clf.resetVolumeApplyJob();
    } finally {
      setVolumeApplyCommitting(false);
    }
  }, [sourceKey, clf, byImage, commitClassIds, addShapesAcrossSlices]);

  const classLabelForId = useCallback(
    (classId: number) => {
      const c = classes.find((x) => x.classId === classId);
      return c?.label?.trim() || `class ${classId}`;
    },
    [classes],
  );

  const predictionClassColorById = useMemo(() => {
    const m = new Map<number, string>();
    for (const c of classes) m.set(c.classId, c.color);
    return m;
  }, [classes]);

  // Load the previewed version's payload (cached) whenever the slider moves.
  useEffect(() => {
    if (previewVersion === null) {
      setPreviewPayload(null);
      return;
    }
    let cancelled = false;
    setPreviewLoading(true);
    fetchVersionPayload(previewVersion).then((payload) => {
      if (cancelled) return;
      setPreviewPayload(payload);
      setPreviewLoading(false);
    });
    return () => { cancelled = true; };
  }, [previewVersion, fetchVersionPayload]);

  // Exit preview if the source changes out from under us.
  useEffect(() => {
    setPreviewVersion(null);
  }, [sourceKey]);

  /** Restores the given version into the editor and exits preview mode. */
  const handleRestoreFromPreview = useCallback((version: number) => {
    restoreVersion(version);
    setPreviewVersion(null);
  }, [restoreVersion]);

  // Shapes for the current slice from the previewed version (read-only on canvas).
  const previewShapes = previewPayload
    ? (previewPayload.slices[String(currentSlice)] ?? [])
    : null;

  /** Removes the currently selected shapes from the active slice and clears the selection. */
  const handleDeleteSelected = () => {
    if (!sourceKey || selectedShapeIds.length === 0) return;
    removeShapes(sourceKey, currentSlice, selectedShapeIds);
    setSelectedShapeId(null);
  };

  /** Builds the save payload and opens the save modal (no-op if nothing to save). */
  const handleOpenSaveModal = () => {
    const payload = buildSavePayload();
    if (!payload) return;
    setSaveModalPayload(payload);
    setShowSaveModal(true);
  };

  /** Saves the version and closes the modal on success. */
  const handleConfirmSave = async (opts: { annotatedBy: string; notes: string; thumbnailBase64?: string }) => {
    const ok = await save(opts);
    if (ok) {
      setShowSaveModal(false);
      setSaveModalPayload(null);
    }
  };

  /** Cancels the in-progress draft by clearing the active brush instance. */
  const handleCancelDraft = () => {
    setActiveBrushShapeId(null);
  };

  useKeybinds(
    activeClassId,
    handleActivateClass,
    () => setActiveBrushShapeId(null),
    handleDeleteSelected,
    handleCancelDraft
  );

  if (!meta) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-sky-200">
        <p>No sample loaded. Pick a sample to annotate.</p>
        <button
          type="button"
          onClick={() => navigate('/browse')}
          className="px-4 py-2 rounded-md bg-sky-600 text-white text-sm font-medium hover:bg-sky-700 transition-colors"
        >
          Go to Browse
        </button>
      </div>
    );
  }

  const savedLabel = lastSavedAt
    ? `Saved ${new Date(lastSavedAt).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`
    : versions.length > 0
      ? `v${versions[versions.length - 1].version} saved`
      : null;

  return (
    <>
      <div className="flex h-full overflow-hidden">
        {/* Sidebar */}
        <div className="w-72 flex-shrink-0 border-r border-gray-200 bg-white overflow-y-auto overflow-x-hidden p-3 flex flex-col gap-4">
          <ClassManager
            activeClassId={activeClassId}
            onActivate={handleActivateClass}
            onClassDeleted={handleClassDeleted}
          />
          <DebouncedSlider
            label="Annotation opacity"
            format={(v) => `${v}%`}
            min={0}
            max={100}
            value={Math.round(fillOpacity * 100)}
            onChange={(v) => setFillOpacity(v / 100)}
          />
          <LayersPanel
            hasFeatures={!!features.job && features.channelIndex !== null}
            hasProba={!!clf.probaUrl}
            hasPredictions={!!clf.commitUrl}
            predictionClassIds={clf.model?.classIds ?? []}
            hasManifold={manifold.hasSample || !!manifold.heatmapUrl}
          />
          <StageSwitcher stage={stage} onChange={setStage} />
          <SliceNavigator />

          {stage === 'draw' && (
            <>
              <Toolbar
                disabled={classes.length === 0}
                histogramBins={histogramBins}
                display={thresholdDisplay}
                upscale={upscale}
                samplerFit={samplerFit}
                onRevertSamplerFit={revertSamplerFit}
              />
              <hr />
              <DisplayControls
                brightness={brightness}
                contrast={contrast}
                onBrightnessChange={setBrightness}
                onContrastChange={setContrast}
                onReset={() => { setBrightness(0); setContrast(0); setLevelsLo(0); setLevelsHi(255); setColormap('gray'); setGamma(1); setClahe(false); setSharpen(false); setBlur(0); setUpscale(1); }}
                histogramBins={histogramBins}
                levelsLo={levelsLo}
                levelsHi={levelsHi}
                onLevelsChange={(lo, hi) => { setLevelsLo(lo); setLevelsHi(hi); }}
                onLevelsReset={() => { setLevelsLo(0); setLevelsHi(255); }}
                colormap={colormap}
                gamma={gamma}
                onColormapChange={setColormap}
                onGammaChange={setGamma}
                clahe={clahe}
                sharpen={sharpen}
                onClaheChange={setClahe}
                onSharpenChange={setSharpen}
                blur={blur}
                onBlurChange={setBlur}
                upscale={upscale}
                onUpscaleChange={setUpscale}
                maxUpscale={maxUpscale}
                denoiseSlot={
                  <DenoisePanel
                    denoise={denoise}
                    onChange={setDenoise}
                    source={source}
                    kind={kind}
                    serverUri={serverUri}
                    sliceIndex={currentSlice}
                    onBake={source ? () => setBakeOpen(true) : undefined}
                  />
                }
              />
              <MaskToolsPanel sourceKey={sourceKey} activeClassId={activeClassId} />
              <MeasurementPanel sourceKey={sourceKey} />
            </>
          )}

          {stage === 'assist' && (
            <FeatureChannelsPanel
              job={features.job}
              channelIndex={features.channelIndex}
              computing={features.computing}
              error={features.error}
              onCompute={features.compute}
              onSelectChannel={features.selectChannel}
              onCycle={features.cycleChannel}
              onOriginal={features.clearSelection}
            />
          )}

          {stage === 'predict' && (
              <PixelClassifierPanel
                hasFeatureJob={!!features.job}
                canAutoPreprocess={clf.canTrainWithoutJob}
                hasShapes={sliceShapes.length > 0}
                training={clf.training}
                predicting={clf.predicting}
                params={clf.params}
                onParamsChange={clf.setParams}
                model={clf.model}
                hasPrediction={!!clf.commitUrl}
                predictCounts={clf.predictCounts}
                probaClassIndex={clf.probaClassIndex}
                activeProbaClassId={clf.activeProbaClassId}
                activeProbaThreshold={clf.activeProbaThreshold}
                classLabelForId={classLabelForId}
                onCycleProbaClass={clf.cycleProbaClass}
                onProbaThresholdChange={clf.setProbaThreshold}
                error={clf.error}
                onTrain={handleClfTrain}
                onPredict={handleClfPredict}
                onCommit={() => { void handleClfCommit(); }}
                onDismiss={clf.dismiss}
                commitLabel="Commit singletons"
                featureSetupId={features.job?.setupId ?? null}
                trainerId={clf.trainerId}
                annotatedSliceCount={annotatedSliceCount}
                trainAcrossSlices={trainAcrossSlices}
                onTrainAcrossSlicesChange={setTrainAcrossSlices}
                multiTraining={clf.multiTraining}
                multiTrainProgress={{ done: clf.multiTrainJob.done, total: clf.multiTrainJob.total }}
                totalSliceCount={totalSliceCount}
                commitClassIds={commitClassIds}
                onToggleCommitClassId={toggleCommitClassId}
                volumeApplying={clf.volumeApplying}
                volumeApplyProgress={{ done: clf.volumeApplyJob.done, total: clf.volumeApplyJob.total }}
                volumeApplyResult={
                  clf.volumeApplyJob.result
                    ? (() => {
                        const r = clf.volumeApplyJob.result as {
                          runs?: Record<string, string>;
                          errors?: unknown[];
                          cancelled?: boolean;
                        };
                        return {
                          runCount: Object.keys(r.runs ?? {}).length,
                          errorCount: (r.errors ?? []).length,
                          cancelled: !!r.cancelled,
                        };
                      })()
                    : null
                }
                volumeApplyCommitting={volumeApplyCommitting}
                onApplyToVolume={handleApplyToVolume}
                onCommitVolumeApply={() => { void handleCommitVolumeApply(); }}
                onCancelVolumeApply={() => { void handleCancelVolumeApply(); }}
                onDismissVolumeApply={clf.resetVolumeApplyJob}
                manifoldParams={manifold.params}
                onManifoldParamsChange={manifold.setParams}
                manifoldSampling={manifold.sampling}
                manifoldHasSample={manifold.hasSample}
                manifoldShowHeatmap={manifold.showHeatmap}
                onManifoldShowHeatmapChange={manifold.setShowHeatmap}
                manifoldShowMarkers={manifold.showMarkers}
                onManifoldShowMarkersChange={manifold.setShowMarkers}
                manifoldHeatmapOpacity={manifold.heatmapOpacity}
                onManifoldHeatmapOpacityChange={manifold.setHeatmapOpacity}
                manifoldMeta={manifold.meta}
                manifoldError={manifold.error}
                onManifoldSample={() => { void manifold.sample(); }}
                onManifoldDismiss={manifold.dismiss}
                manifoldRoiShapeCount={manifold.placementMask?.length ?? 0}
                canCaptureManifoldRoi={selectedShapeIds.length > 0}
                onCaptureManifoldRoi={() => {
                  const selected = sliceShapes.filter((s) => selectedShapeIds.includes(s.id));
                  manifold.setPlacementMaskFromShapes(selected);
                }}
                onClearManifoldRoi={manifold.clearPlacementMask}
              />
          )}

          {/* Save button + status */}
          <div className="flex flex-col gap-1.5">
            <button
              type="button"
              onClick={handleOpenSaveModal}
              disabled={isSaving}
              className={[
                'flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isDirty
                  ? 'bg-sky-600 text-white hover:bg-sky-700'
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200',
                isSaving ? 'opacity-60 cursor-not-allowed' : '',
              ].join(' ')}
            >
              {isSaving ? (
                <CircleDashed size={16} className="animate-spin" />
              ) : (
                <FloppyDisk size={16} />
              )}
              {isSaving ? 'Saving…' : isDirty ? 'Save' : 'Saved'}
            </button>

            {/* Dirty / saved status line */}
            <div className="flex items-center justify-between text-xs text-gray-500 px-0.5">
              <span>{isDirty ? 'Unsaved changes' : (savedLabel ?? 'No saves yet')}</span>
              {versions.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowVersionHistory(true)}
                  className="flex items-center gap-1 hover:text-sky-600 transition-colors"
                  title="Version history"
                >
                  <ClockCounterClockwise size={13} />
                  {versions.length}
                </button>
              )}
            </div>
          </div>

          {/* Insights */}
          <button
            type="button"
            onClick={() => setShowInsights(true)}
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-gray-100 text-gray-700 text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            <ChartBar size={16} />
            Insights
          </button>

          {/* Download / export */}
          <button
            type="button"
            onClick={() => setShowDownload(true)}
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-gray-100 text-gray-700 text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            <DownloadSimple size={16} />
            Export
          </button>
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-hidden relative">
          <AnnotationCanvas
            brightness={brightness}
            contrast={contrast}
            levelsLo={levelsLo}
            levelsHi={levelsHi}
            colormap={colormap}
            gamma={gamma}
            clahe={clahe}
            sharpen={sharpen}
            blur={blur}
            upscale={upscale}
            onHistogram={setHistogramBins}
            onSamplerFit={setSamplerFit}
            onBlurChange={setBlur}
            activeClassId={activeClassId}
            activeBrushShapeId={activeBrushShapeId}
            onNewBrushInstance={setActiveBrushShapeId}
            previewShapes={previewShapes}
            previewClasses={previewPayload?.classes ?? null}
            focusRegion={focusRegion}
            featureChannelUrl={features.channelUrl}
            probaOverlayUrl={clf.probaUrl}
            clfCommitUrl={clf.commitUrl}
            clfStatusUrl={clf.statusUrl}
            predictionClassColorById={predictionClassColorById}
            manifoldHeatmapUrl={manifold.heatmapUrl}
            manifoldHeatmapOpacity={manifold.heatmapOpacity}
            manifoldShowHeatmap={manifold.showHeatmap}
            manifoldMarkers={manifold.points}
            manifoldShowMarkers={manifold.showMarkers}
            manifoldBoxSize={manifold.meta?.boxSize ?? 64}
          />
          {previewVersion !== null && (
            <VersionPreviewBar
              versions={versions}
              current={previewVersion}
              loading={previewLoading}
              onChange={setPreviewVersion}
              onExit={() => setPreviewVersion(null)}
              onRestore={handleRestoreFromPreview}
            />
          )}
          {perfOn && <PerfOverlay shapeCount={currentShapeCount} />}
        </div>
      </div>

      {showDownload && <DownloadModal onClose={() => setShowDownload(false)} />}
      {showInsights && (
        <InsightsModal
          sourceKey={sourceKey}
          onClose={() => setShowInsights(false)}
          onFocus={handleInsightFocus}
        />
      )}
      {showSaveModal && saveModalPayload && sourceKey && (
        <SaveModal
          sourceKey={sourceKey}
          payload={saveModalPayload}
          shapeCount={saveSummary.shapeCount}
          classCount={saveSummary.classCount}
          isSaving={isSaving}
          onSave={handleConfirmSave}
          onClose={() => {
            if (!isSaving) {
              setShowSaveModal(false);
              setSaveModalPayload(null);
            }
          }}
        />
      )}
      {showVersionHistory && (
        <VersionHistoryModal
          versions={versions}
          sourceKey={sourceKey}
          onPreview={(v) => { setPreviewVersion(v); setShowVersionHistory(false); }}
          onRestore={restoreVersion}
          onClose={() => setShowVersionHistory(false)}
        />
      )}
      {source && (
        <DenoiseBakeModal
          open={bakeOpen}
          onClose={() => setBakeOpen(false)}
          source={source}
          serverUri={serverUri}
          denoise={denoise}
          nSlices={meta?.nSlices ?? 1}
          methodLabel={denoise.method}
        />
      )}
    </>
  );
}
