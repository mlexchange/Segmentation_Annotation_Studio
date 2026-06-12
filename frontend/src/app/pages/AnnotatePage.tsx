/**
 * AnnotatePage — react-konva canvas workspace with sidebar tools.
 */
import { useState, useEffect, useCallback } from 'react';
import { DownloadSimple } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { useDraftSync } from '@/hooks/useDraftSync';
import { buildSourceKey } from '@/lib/sourceKey';
import { useKeybinds } from '@/hooks/useKeybinds';
import Toolbar from '@/components/annotate/Toolbar';
import ClassManager from '@/components/annotate/ClassManager';
import DisplayControls from '@/components/annotate/DisplayControls';
import SliceNavigator from '@/components/annotate/SliceNavigator';
import AnnotationCanvas from '@/components/annotate/AnnotationCanvas';
import DownloadModal from '@/components/annotate/DownloadModal';

export default function AnnotatePage() {
  const { source, kind, serverUri, meta } = useDatasetStore();
  const { removeShape } = useAnnotationStore();
  const { selectedShapeId, setSelectedShapeId } = useToolStore();
  const { classes } = useClassStore();

  const [activeClassId, setActiveClassId] = useState<number | null>(null);
  const [activeBrushShapeId, setActiveBrushShapeId] = useState<string | null>(null);

  const handleActivateClass = useCallback((classId: number) => {
    setActiveClassId(classId);
    // New class → new brush instance so paint uses the selected class color.
    setActiveBrushShapeId(null);
  }, []);

  // Auto-select the first class when classes exist but none is active.
  useEffect(() => {
    if (activeClassId !== null) return;
    if (classes.length > 0) setActiveClassId(classes[0].classId);
  }, [classes, activeClassId]);
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [showDownload, setShowDownload] = useState(false);

  // Build canonical sourceKey for autosave
  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;
  useDraftSync(sourceKey);

  const { currentSlice } = useDatasetStore();

  const handleDeleteSelected = () => {
    if (!sourceKey || !selectedShapeId) return;
    removeShape(sourceKey, currentSlice, selectedShapeId);
    setSelectedShapeId(null);
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
          <ClassManager activeClassId={activeClassId} onActivate={handleActivateClass} />
          <hr />
          <SliceNavigator />
          <hr />
          {/* Download button at the bottom of the sidebar */}
          <button
            type="button"
            onClick={() => setShowDownload(true)}
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-sky-600 text-white text-sm font-medium hover:bg-sky-700 transition-colors"
          >
            <DownloadSimple size={16} />
            Download
          </button>
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-hidden">
          <AnnotationCanvas
            brightness={brightness}
            contrast={contrast}
            activeClassId={activeClassId}
            activeBrushShapeId={activeBrushShapeId}
            onNewBrushInstance={setActiveBrushShapeId}
          />
        </div>
      </div>

      {showDownload && <DownloadModal onClose={() => setShowDownload(false)} />}
    </>
  );
}
