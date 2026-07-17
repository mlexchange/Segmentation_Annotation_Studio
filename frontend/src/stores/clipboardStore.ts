/**
 * clipboardStore — in-memory shape clipboard for copy/paste within the editor.
 * Holds a deep copy of the shapes that were copied (ids stripped on paste).
 */
import { create } from 'zustand';
import type { Shape } from '@/stores/annotationStore';

interface ClipboardState {
  shapes: Shape[];
  /** Store a deep copy of the given shapes as the clipboard contents. */
  copy: (shapes: Shape[]) => void;
  clear: () => void;
}

export const useClipboardStore = create<ClipboardState>((set) => ({
  shapes: [],
  copy: (shapes) => set({ shapes: shapes.map((s) => structuredClone(s)) }),
  clear: () => set({ shapes: [] }),
}));
