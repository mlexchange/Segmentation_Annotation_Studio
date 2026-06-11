/**
 * useKeybinds — keyboard shortcuts matching mlex bindings + extensions.
 *
 * q=polygon  w=ellipse  e=rectangle  a=pan
 * b=brush    x=eraser   s=select     n=new-brush-instance
 * 1-9=class  arrows=slice  Del/Back=delete  Ctrl+Z/Ctrl+Shift+Z=undo/redo  Esc=cancel
 */
import { useEffect } from 'react';
import { useToolStore, type Tool } from '@/stores/toolStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { useTemporalStore } from 'zundo';

const TOOL_KEYBINDS: Record<string, Tool> = {
  a: 'pan',
  s: 'select',
  q: 'polygon',
  e: 'rectangle',
  w: 'ellipse',
  b: 'brush',
  x: 'eraser',
};

export function useKeybinds(
  activeClassId: number | null,
  onActivateClass: (id: number) => void,
  onNewBrushInstance: () => void,
  onDeleteSelected: () => void,
  onCancelDraft: () => void
) {
  const { setTool } = useToolStore();
  const { meta, currentSlice, setSlice } = useDatasetStore();
  const { classes } = useClassStore();
  const temporalStore = useTemporalStore(useAnnotationStore);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Skip when typing in an input/textarea
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      const key = e.key;

      // Undo / redo
      if (e.ctrlKey || e.metaKey) {
        if (key === 'z' && !e.shiftKey) { e.preventDefault(); temporalStore.undo(); return; }
        if ((key === 'z' && e.shiftKey) || key === 'y') { e.preventDefault(); temporalStore.redo(); return; }
      }

      // Tool select
      if (TOOL_KEYBINDS[key]) { setTool(TOOL_KEYBINDS[key]); return; }

      // Class select 1-9
      const digit = parseInt(key);
      if (digit >= 1 && digit <= 9) {
        const cls = classes[digit - 1];
        if (cls) onActivateClass(cls.classId);
        return;
      }

      // Slice navigation
      if (key === 'ArrowLeft' || key === 'ArrowRight') {
        e.preventDefault();
        if (!meta) return;
        const n = meta.nSlices;
        setSlice(key === 'ArrowLeft' ? Math.max(0, currentSlice - 1) : Math.min(n - 1, currentSlice + 1));
        return;
      }

      if (key === 'n') { onNewBrushInstance(); return; }
      if (key === 'Delete' || key === 'Backspace') { onDeleteSelected(); return; }
      if (key === 'Escape') { onCancelDraft(); return; }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [setTool, meta, currentSlice, setSlice, classes, onActivateClass, onNewBrushInstance, onDeleteSelected, onCancelDraft, temporalStore]);
}
