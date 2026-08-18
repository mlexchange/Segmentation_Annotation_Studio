/**
 * AnnotatePage — react-konva canvas workspace with sidebar tools.
 */
import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router';
import { DownloadSimple, FloppyDisk, ClockCounterClockwise, CircleDashed, ChartBar } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { useDraftSync } from '@/hooks/useDraftSync';
import { clearHistory } from '@/hooks/editHistory';
import { useGuideLoad } from '@/hooks/useGuideSync';
import { useSave, type VersionPayload } from '@/hooks/useSave';
import { buildSourceKey } from '@/lib/sourceKey';
import { useKeybinds } from '@/hooks/useKeybinds';
import type { ColormapName } from '@/lib/colormaps';
import Toolbar from '@/components/annotate/Toolbar';
import ClassManager from '@/components/annotate/ClassManager';
import DisplayControls from '@/components/annotate/DisplayControls';
import SliceNavigator from '@/components/annotate/SliceNavigator';
import MaskToolsPanel from '@/components/annotate/MaskToolsPanel';
import MeasurementPanel from '@/components/annotate/MeasurementPanel';
import AnnotationCanvas from '@/components/annotate/AnnotationCanvas';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import DownloadModal from '@/components/annotate/DownloadModal';
import InsightsModal from '@/components/annotate/InsightsModal';
import VersionHistoryModal from '@/components/annotate/VersionHistoryModal';
import VersionPreviewBar from '@/components/annotate/VersionPreviewBar';
import SaveModal from '@/components/annotate/SaveModal';
import type { SaveDraftPayload } from '@/hooks/useSave';

/** Renders the annotation workspace: tool sidebar, canvas, and save/version/export flows. */
export default function AnnotatePage() {
  const navigate = useNavigate();
  const { source, kind, serverUri, meta } = useDatasetStore();
  const { removeShapes } = useAnnotationStore();
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

  // Crash-recovery autosave (local draft only, no Tiled sync)
  useDraftSync(sourceKey);
  // Undo/redo is per-sample: reset the region history + class-delete journal on switch.
  useEffect(() => { clearHistory(); }, [sourceKey]);
  // Load the dataset's annotation guide (read-only) for class suggestions/examples.
  useGuideLoad(sourceKey);

  // Explicit versioned save
  const { isDirty, isSaving, lastSavedAt, save, buildSavePayload, saveSummary, versions, fetchVersionPayload, restoreVersion } = useSave(sourceKey);

  const { currentSlice, setSlice } = useDatasetStore();

  /** From an Insights QA flag: jump to its slice and zoom/highlight its region. */
  const handleInsightFocus = useCallback((slice: number, bbox?: { x: number; y: number; w: number; h: number }) => {
    setSlice(slice);
    setFocusRegion(bbox ? { ...bbox, nonce: Date.now() } : null);
  }, [setSlice]);

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
          <hr />
          <Toolbar
            disabled={classes.length === 0}
            histogramBins={histogramBins}
            display={thresholdDisplay}
            upscale={upscale}
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
          />
          <hr />
          <SliceNavigator />
          <hr />
          <MaskToolsPanel sourceKey={sourceKey} activeClassId={activeClassId} />
          <hr />
          <MeasurementPanel sourceKey={sourceKey} />
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
            activeClassId={activeClassId}
            activeBrushShapeId={activeBrushShapeId}
            onNewBrushInstance={setActiveBrushShapeId}
            previewShapes={previewShapes}
            previewClasses={previewPayload?.classes ?? null}
            focusRegion={focusRegion}
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
    </>
  );
}
