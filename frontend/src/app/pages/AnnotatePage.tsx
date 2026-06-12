/**
 * AnnotatePage — react-konva canvas workspace with sidebar tools.
 */
import { useState, useEffect, useCallback } from 'react';
import { DownloadSimple, FloppyDisk, ClockCounterClockwise, CircleDashed } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { useDraftSync } from '@/hooks/useDraftSync';
import { useSave, type VersionPayload } from '@/hooks/useSave';
import { buildSourceKey } from '@/lib/sourceKey';
import { useKeybinds } from '@/hooks/useKeybinds';
import Toolbar from '@/components/annotate/Toolbar';
import ClassManager from '@/components/annotate/ClassManager';
import DisplayControls from '@/components/annotate/DisplayControls';
import SliceNavigator from '@/components/annotate/SliceNavigator';
import AnnotationCanvas from '@/components/annotate/AnnotationCanvas';
import DownloadModal from '@/components/annotate/DownloadModal';
import VersionHistoryModal from '@/components/annotate/VersionHistoryModal';
import VersionPreviewBar from '@/components/annotate/VersionPreviewBar';
import SaveModal from '@/components/annotate/SaveModal';
import type { SaveDraftPayload } from '@/hooks/useSave';

export default function AnnotatePage() {
  const { source, kind, serverUri, meta } = useDatasetStore();
  const { removeShape } = useAnnotationStore();
  const { selectedShapeId, setSelectedShapeId } = useToolStore();
  const { classes } = useClassStore();

  const [activeClassId, setActiveClassId] = useState<number | null>(null);
  const [activeBrushShapeId, setActiveBrushShapeId] = useState<string | null>(null);

  const handleActivateClass = useCallback((classId: number) => {
    setActiveClassId(classId);
    setActiveBrushShapeId(null);
  }, []);

  const handleClassDeleted = useCallback((_deletedClassId: number) => {
    setActiveBrushShapeId(null);
  }, []);

  useEffect(() => {
    if (activeClassId !== null) return;
    if (classes.length > 0) setActiveClassId(classes[0].classId);
  }, [classes, activeClassId]);

  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
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

  const handleRestoreFromPreview = useCallback((version: number) => {
    restoreVersion(version);
    setPreviewVersion(null);
  }, [restoreVersion]);

  // Shapes for the current slice from the previewed version (read-only on canvas).
  const previewShapes = previewPayload
    ? (previewPayload.slices[String(currentSlice)] ?? [])
    : null;

  const handleDeleteSelected = () => {
    if (!sourceKey || !selectedShapeId) return;
    removeShape(sourceKey, currentSlice, selectedShapeId);
    setSelectedShapeId(null);
  };

  const handleOpenSaveModal = () => {
    const payload = buildSavePayload();
    if (!payload) return;
    setSaveModalPayload(payload);
    setShowSaveModal(true);
  };

  const handleConfirmSave = async (opts: { annotatedBy: string; notes: string; thumbnailBase64?: string }) => {
    const ok = await save(opts);
    if (ok) {
      setShowSaveModal(false);
      setSaveModalPayload(null);
    }
  };

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
      <div className="flex h-full items-center justify-center text-sky-200">
        <p>No sample loaded. Go to Browse to pick a sample.</p>
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
          <Toolbar />
          <hr />
          <DisplayControls
            brightness={brightness}
            contrast={contrast}
            onBrightnessChange={setBrightness}
            onContrastChange={setContrast}
            onReset={() => { setBrightness(0); setContrast(0); }}
          />
          <hr />
          <ClassManager
            activeClassId={activeClassId}
            onActivate={handleActivateClass}
            onClassDeleted={handleClassDeleted}
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
