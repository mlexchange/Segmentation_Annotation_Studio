/**
 * AnnotateWorkspace — shared canvas + Hub-stage-driven sidebar.
 */
import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router';
import { DownloadSimple, FloppyDisk, ClockCounterClockwise, CircleDashed } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { useMaskSetStore } from '@/stores/maskSetStore';
import { useDraftSync } from '@/hooks/useDraftSync';
import { useSave, type VersionPayload } from '@/hooks/useSave';
import { buildSourceKey } from '@/lib/sourceKey';
import { stageFromPath } from '@/lib/annotateStage';
import { useKeybinds } from '@/hooks/useKeybinds';
import Toolbar from '@/components/annotate/Toolbar';
import ClassManager from '@/components/annotate/ClassManager';
import DisplayControls from '@/components/annotate/DisplayControls';
import FeatureChannelsPanel from '@/components/annotate/FeatureChannelsPanel';
import PixelClassifierPanel from '@/components/annotate/PixelClassifierPanel';
import ManifoldSuggestPanel from '@/components/annotate/ManifoldSuggestPanel';
import SliceNavigator from '@/components/annotate/SliceNavigator';
import MaskToolsPanel from '@/components/annotate/MaskToolsPanel';
import ModelShelfPanel from '@/components/annotate/ModelShelfPanel';
import LayersPanel from '@/components/annotate/LayersPanel';
import AnnotationCanvas from '@/components/annotate/AnnotationCanvas';
import DownloadModal from '@/components/annotate/DownloadModal';
import VersionHistoryModal from '@/components/annotate/VersionHistoryModal';
import VersionPreviewBar from '@/components/annotate/VersionPreviewBar';
import SaveModal from '@/components/annotate/SaveModal';
import type { SaveDraftPayload } from '@/hooks/useSave';
import { useFeatureChannels } from '@/hooks/useFeatureChannels';
import { usePixelClassifier } from '@/hooks/usePixelClassifier';
import { useFeatureManifold } from '@/hooks/useFeatureManifold';
import { classProbaMaskSetName, labelMapToPolygonShapes, loadLabelPng } from '@/lib/pixelClf';
import { labelMapFromBase64 } from '@/lib/labelMap';
import { featureParamsFromRecipe } from '@/lib/featureRecipe';
import { useConnectionStore } from '@/stores/connectionStore';
import type { Tool } from '@/stores/toolStore';

const PREPROCESS_TOOLS: Tool[] = ['pan', 'select', 'polygon', 'rectangle', 'ellipse', 'brush'];

/** Shared annotate shell used by Preprocess / Draw / Train Hub stages. */
export default function AnnotateWorkspace() {
  const navigate = useNavigate();
  const location = useLocation();
  const stage = stageFromPath(location.pathname);
  const { source, kind, serverUri, meta } = useDatasetStore();
  const { removeShapes, byImage, addShapes } = useAnnotationStore();
  const { selectedShapeIds, setSelectedShapeId } = useToolStore();
  const { classes } = useClassStore();
  const addMaskSet = useMaskSetStore((s) => s.addSet);

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
  const [clahe, setClahe] = useState(false);
  const [sharpen, setSharpen] = useState(false);
  // Min/max levels window (0–255) + histogram of the current slice (client-side).
  const [levelsLo, setLevelsLo] = useState(0);
  const [levelsHi, setLevelsHi] = useState(255);
  const [histogramBins, setHistogramBins] = useState<number[] | null>(null);
  const [showDownload, setShowDownload] = useState(false);
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

  // Crash-recovery autosave (local draft only, no Tiled sync)
  useDraftSync(sourceKey);

  // Explicit versioned save
  const { isDirty, isSaving, lastSavedAt, save, buildSavePayload, saveSummary, versions, fetchVersionPayload, restoreVersion } = useSave(sourceKey);

  const { currentSlice } = useDatasetStore();

  const features = useFeatureChannels({
    source,
    kind,
    sliceIndex: currentSlice,
    serverUri,
  });

  // Prefill Preprocess feature params from the session's preferred model recipe.
  const preferredFeatureRecipe = useConnectionStore((s) => s.preferredFeatureRecipe);
  useEffect(() => {
    if (!preferredFeatureRecipe) return;
    features.setParams(featureParamsFromRecipe(preferredFeatureRecipe));
    // Intentionally only when the preferred recipe identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preferredFeatureRecipe]);

  const sliceShapes = sourceKey
    ? (byImage[sourceKey]?.[String(currentSlice)] ?? [])
    : [];

  const clf = usePixelClassifier({
    featureJobId: features.job?.jobId ?? null,
    resetKey: sourceKey ? `${sourceKey}:${currentSlice}` : null,
    source,
    kind,
    serverUri,
    sliceIndex: currentSlice,
    onFeatureJobExpired: features.invalidateJob,
    onFeatureReady: (info) => {
      features.adoptFeatureBank(info);
    },
  });

  const manifold = useFeatureManifold({
    featureJobId: features.job?.jobId ?? null,
    onFeatureJobExpired: features.invalidateJob,
  });

  const handleClfTrain = useCallback(() => {
    void clf.train(sliceShapes);
  }, [clf, sliceShapes]);

  const handleClfPredict = useCallback(() => {
    void clf.predict(sliceShapes);
  }, [clf, sliceShapes]);

  const handleClfCommit = useCallback(async () => {
    if (!sourceKey || !clf.commitUrl || !clf.model) return;
    try {
      const { data, width, height } = await loadLabelPng(clf.commitUrl);
      const shapes = labelMapToPolygonShapes(data, width, height, clf.model.classIds, {
        minRegion: 64,
        preserveShapes: sliceShapes,
      });
      if (shapes.length) addShapes(sourceKey, currentSlice, shapes);
      clf.dismiss();
    } catch {
      /* keep overlay; error surface via clf if needed */
    }
  }, [sourceKey, clf, addShapes, currentSlice, sliceShapes]);

  const classLabelForId = useCallback(
    (classId: number) => {
      const c = classes.find((x) => x.classId === classId);
      return c?.label?.trim() || `class ${classId}`;
    },
    [classes],
  );

  const handleSaveClassMap = useCallback(async () => {
    if (!sourceKey || clf.activeProbaClassId === null) return;
    const result = await clf.saveThresholdedClass();
    if (!result) return;
    const labelMap = labelMapFromBase64(result.label_map_b64, result.width * result.height);
    addMaskSet({
      name: classProbaMaskSetName(
        result.class_id,
        result.threshold,
        classLabelForId(result.class_id),
      ),
      sourceKey,
      slice: currentSlice,
      origin: 'clf',
      labelMap,
      width: result.width,
      height: result.height,
      shapes: [],
    });
  }, [sourceKey, clf, addMaskSet, currentSlice, classLabelForId]);

  const maskSetOverlayShapes = null;
  const maskSetLabelOverlay = null;

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
        <div className="w-56 flex-shrink-0 border-r border-gray-200 bg-white overflow-y-auto p-3 flex flex-col gap-4">
          {stage === 'draw' && (
            <>
              <ClassManager
                activeClassId={activeClassId}
                onActivate={handleActivateClass}
                onClassDeleted={handleClassDeleted}
              />
              <hr />
              <Toolbar disabled={classes.length === 0} />
              <hr />
              <MaskToolsPanel sourceKey={sourceKey} activeClassId={activeClassId} />
              <hr />
            </>
          )}

          {stage === 'preprocess' && (
            <>
              <Toolbar allowedTools={[...PREPROCESS_TOOLS]} disabled={false} />
              <hr />
              <DisplayControls
                brightness={brightness}
                contrast={contrast}
                onBrightnessChange={setBrightness}
                onContrastChange={setContrast}
                onReset={() => {
                  setBrightness(0);
                  setContrast(0);
                  setClahe(false);
                  setSharpen(false);
                  setLevelsLo(0);
                  setLevelsHi(255);
                }}
                histogramBins={histogramBins}
                levelsLo={levelsLo}
                levelsHi={levelsHi}
                onLevelsChange={(lo, hi) => { setLevelsLo(lo); setLevelsHi(hi); }}
                onLevelsReset={() => { setLevelsLo(0); setLevelsHi(255); }}
                clahe={clahe}
                onClaheChange={setClahe}
                sharpen={sharpen}
                onSharpenChange={setSharpen}
              />
              <hr />
              <FeatureChannelsPanel
                params={features.params}
                onParamsChange={features.setParams}
                job={features.job}
                channelIndex={features.channelIndex}
                computing={features.computing}
                error={features.error}
                samAvailable={features.samAvailable}
                featureSetupId={features.preferredFeatureSetupId}
                onCompute={features.compute}
                onSelectChannel={features.selectChannel}
                onCycle={features.cycleChannel}
                onOriginal={features.clearSelection}
                manifoldSlot={(
                  <ManifoldSuggestPanel
                    hasFeatureJob={!!features.job}
                    params={manifold.params}
                    onParamsChange={manifold.setParams}
                    sampling={manifold.sampling}
                    hasSample={manifold.hasSample}
                    showHeatmap={manifold.showHeatmap}
                    onShowHeatmapChange={manifold.setShowHeatmap}
                    showMarkers={manifold.showMarkers}
                    onShowMarkersChange={manifold.setShowMarkers}
                    heatmapOpacity={manifold.heatmapOpacity}
                    onHeatmapOpacityChange={manifold.setHeatmapOpacity}
                    meta={manifold.meta}
                    error={manifold.error}
                    onSample={() => { void manifold.sample(); }}
                    onDismiss={manifold.dismiss}
                    roiShapeCount={manifold.placementMask?.length ?? 0}
                    canCaptureRoi={selectedShapeIds.length > 0}
                    onCaptureRoi={() => {
                      const selected = sliceShapes.filter((s) =>
                        selectedShapeIds.includes(s.id),
                      );
                      manifold.setPlacementMaskFromShapes(selected);
                    }}
                    onClearRoi={manifold.clearPlacementMask}
                  />
                )}
              />
              <hr />
            </>
          )}

          {stage === 'train' && (
            <>
              <PixelClassifierPanel
                hasFeatureJob={!!features.job}
                canAutoPreprocess={clf.canTrainWithoutJob}
                hasShapes={sliceShapes.length > 0}
                training={clf.training}
                predicting={clf.predicting}
                params={clf.params}
                onParamsChange={clf.setParams}
                model={clf.model}
                treeView={clf.treeView}
                treeIndex={clf.treeIndex}
                onTreeIndexChange={(i) => { void clf.loadTree(i); }}
                hasPrediction={!!clf.commitUrl}
                predictCounts={clf.predictCounts}
                probaClassIndex={clf.probaClassIndex}
                activeProbaClassId={clf.activeProbaClassId}
                activeProbaThreshold={clf.activeProbaThreshold}
                classLabelForId={classLabelForId}
                onCycleProbaClass={clf.cycleProbaClass}
                onProbaThresholdChange={clf.setProbaThreshold}
                onSaveClassMap={() => { void handleSaveClassMap(); }}
                savingClass={clf.savingClass}
                error={clf.error}
                onTrain={handleClfTrain}
                onPredict={handleClfPredict}
                onCommit={() => { void handleClfCommit(); }}
                onDismiss={clf.dismiss}
                commitLabel="Commit singletons"
                featureSetupId={clf.featureSetupId}
                trainerId={clf.trainerId}
              />
              <hr />
              <ModelShelfPanel
                sourceKey={sourceKey}
                trainedModelId={clf.model?.modelId ?? null}
                alpha={clf.params.alpha}
                featureParams={features.params}
              />
              <hr />
            </>
          )}

          <LayersPanel
            hasFeatures={!!features.job && features.channelIndex !== null}
            hasProba={!!clf.probaUrl}
            hasPredictions={!!clf.commitUrl}
            predictionClassIds={clf.model?.classIds ?? []}
            hasManifold={manifold.hasSample || !!manifold.heatmapUrl}
          />
          <hr />

          <SliceNavigator />
          <hr />

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
            <div className="flex items-center justify-between text-xs text-gray-400 px-0.5">
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

          {/* Download / export */}
          <button
            type="button"
            onClick={() => setShowDownload(true)}
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-gray-100 text-gray-700 text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            <DownloadSimple size={16} />
            Export COCO
          </button>
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-hidden relative">
          <AnnotationCanvas
            brightness={brightness}
            contrast={contrast}
            clahe={clahe}
            sharpen={sharpen}
            featureChannelUrl={
              features.channelUrl && features.channelIndex !== null
                ? features.channelUrl
                : null
            }
            featureCacheKey={
              features.job && features.channelIndex !== null
                ? `${features.job.jobId}:${features.channelIndex}`
                : null
            }
            probaOverlayUrl={clf.probaUrl}
            clfCommitUrl={clf.commitUrl}
            clfStatusUrl={clf.statusUrl}
            maskSetOverlayShapes={maskSetOverlayShapes}
            maskSetLabelOverlay={maskSetLabelOverlay}
            manifoldHeatmapUrl={manifold.heatmapUrl}
            manifoldHeatmapOpacity={manifold.heatmapOpacity}
            manifoldShowHeatmap={manifold.showHeatmap}
            manifoldMarkers={manifold.points}
            manifoldBoxSize={manifold.meta?.boxSize}
            manifoldShowMarkers={manifold.showMarkers}
            placementMaskShapes={manifold.placementMask}
            levelsLo={levelsLo}
            levelsHi={levelsHi}
            onHistogram={setHistogramBins}
            activeClassId={activeClassId}
            activeBrushShapeId={activeBrushShapeId}
            onNewBrushInstance={setActiveBrushShapeId}
            previewShapes={previewShapes}
            previewClasses={previewPayload?.classes ?? null}
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
        </div>
      </div>

      {showDownload && <DownloadModal onClose={() => setShowDownload(false)} />}
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
    </>
  );
}
