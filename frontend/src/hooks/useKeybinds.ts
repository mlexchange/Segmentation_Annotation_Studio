/**
 * useKeybinds — keyboard shortcuts for the annotation workspace.
 *
 * Tools:   a=polygon  w=ellipse  e=rectangle  q=eraser  b=brush  s=select
 * Pan:     hold Space (reverts to the previous tool on release)
 * View:    x=next slice (forward)  f=fit image to screen  arrows=prev/next slice
 * Edit:    z (or Ctrl/Cmd+Z)=undo   Ctrl/Cmd+Shift+Z / Ctrl+Y=redo
 * Other:   1-9=class  n=new brush instance  Del/Back=delete  Esc=cancel
 */
import { useEffect, useRef } from 'react';
import { useStore } from 'zustand';
import { useToolStore, type Tool } from '@/stores/toolStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

const TOOL_KEYBINDS: Record<string, Tool> = {
  s: 'select',
  a: 'polygon',
  e: 'rectangle',
  w: 'ellipse',
  b: 'brush',
  q: 'eraser',
};

export function useKeybinds(
  activeClassId: number | null,
  onActivateClass: (id: number) => void,
  onNewBrushInstance: () => void,
  onDeleteSelected: () => void,
  onCancelDraft: () => void
) {
  const { setTool, requestFit } = useToolStore();
  const { meta, currentSlice, setSlice } = useDatasetStore();
  const { classes } = useClassStore();
  const temporalStore = useStore(useAnnotationStore.temporal);

  // Tracks the tool to revert to after a hold-Space pan ends.
  const prevToolRef = useRef<Tool | null>(null);

  useEffect(() => {
    const isEditableTarget = (target: EventTarget | null) => {
      const tag = (target as HTMLElement | null)?.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    };

    const goToSlice = (next: number) => {
      if (!meta) return;
      setSlice(Math.max(0, Math.min(meta.nSlices - 1, next)));
    };

    const handler = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;
      const key = e.key;

      // Hold Space = pan; restored on keyup.
      if (key === ' ' || key === 'Spacebar') {
        e.preventDefault();
        if (e.repeat) return;
        const st = useToolStore.getState();
        if (st.tool !== 'pan') {
          prevToolRef.current = st.tool;
          st.setTool('pan');
        }
        return;
      }

      // Undo / redo
      if ((e.ctrlKey || e.metaKey) && key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        temporalStore.undo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && ((key.toLowerCase() === 'z' && e.shiftKey) || key.toLowerCase() === 'y')) {
        e.preventDefault();
        temporalStore.redo();
        return;
      }
      // Plain Z = undo
      if (key === 'z' && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
        e.preventDefault();
        temporalStore.undo();
        return;
      }

      // Fit image to screen
      if (key === 'f') {
        requestFit();
        return;
      }

      // Forward = next slice
      if (key === 'x') {
        e.preventDefault();
        goToSlice(currentSlice + 1);
        return;
      }

      // Tool select
      if (TOOL_KEYBINDS[key]) {
        setTool(TOOL_KEYBINDS[key]);
        return;
      }

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
        goToSlice(currentSlice + (key === 'ArrowLeft' ? -1 : 1));
        return;
      }

      if (key === 'n') { onNewBrushInstance(); return; }
      if (key === 'Delete' || key === 'Backspace') { onDeleteSelected(); return; }
      if (key === 'Escape') { onCancelDraft(); return; }
    };

    const upHandler = (e: KeyboardEvent) => {
      if ((e.key === ' ' || e.key === 'Spacebar') && prevToolRef.current) {
        e.preventDefault();
        useToolStore.getState().setTool(prevToolRef.current);
        prevToolRef.current = null;
      }
    };

    window.addEventListener('keydown', handler);
    window.addEventListener('keyup', upHandler);
    return () => {
      window.removeEventListener('keydown', handler);
      window.removeEventListener('keyup', upHandler);
    };
  }, [setTool, requestFit, meta, currentSlice, setSlice, classes, onActivateClass, onNewBrushInstance, onDeleteSelected, onCancelDraft, temporalStore]);
}
