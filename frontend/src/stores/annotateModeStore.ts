/**
 * AnnotateModeStore — thin compatibility shim; stage is driven by Hub routes.
 * Prefer {@link stageFromPath} from annotateStage for new code.
 */
import { create } from 'zustand';
import type { AnnotateStage } from '@/lib/annotateStage';

export type AnnotateMode = AnnotateStage;

interface AnnotateModeState {
  /** @deprecated Prefer route pathname; kept for any leftover consumers. */
  mode: AnnotateMode;
  setMode: (mode: AnnotateMode) => void;
}

export const useAnnotateModeStore = create<AnnotateModeState>((set) => ({
  mode: 'preprocess',
  setMode: (mode) => set({ mode }),
}));
