/**
 * AnnotatePage — react-konva canvas workspace with sidebar tools.
 */
import { useState } from 'react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useDraftSync } from '@/hooks/useDraftSync';
import { useKeybinds } from '@/hooks/useKeybinds';
import Toolbar from '@/components/annotate/Toolbar';
import ClassManager from '@/components/annotate/ClassManager';
import DisplayControls from '@/components/annotate/DisplayControls';
import SliceNavigator from '@/components/annotate/SliceNavigator';
import AnnotationCanvas from '@/components/annotate/AnnotationCanvas';

export default function AnnotatePage() {
  const { source, meta } = useDatasetStore();
  const { removeShape } = useAnnotationStore();
  const { selectedShapeId, setSelectedShapeId } = useToolStore();

  const [activeClassId, setActiveClassId] = useState<number | null>(null);
  const [activeBrushShapeId, setActiveBrushShapeId] = useState<string | null>(null);
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);

  // Autosave
  useDraftSync(source);

  const { currentSlice } = useDatasetStore();

  const handleDeleteSelected = () => {
    if (!source || !selectedShapeId) return;
    removeShape(source, currentSlice, selectedShapeId);
    setSelectedShapeId(null);
  };

  const handleCancelDraft = () => {
    setActiveBrushShapeId(null);
  };

  useKeybinds(
    activeClassId,
    setActiveClassId,
    () => setActiveBrushShapeId(null),
    handleDeleteSelected,
    handleCancelDraft
  );

  if (!meta) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        <p>No dataset loaded. Go to Connect to pick a dataset.</p>
      </div>
    );
  }

  return (
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
        <ClassManager activeClassId={activeClassId} onActivate={setActiveClassId} />
        <hr />
        <SliceNavigator />
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
  );
}
