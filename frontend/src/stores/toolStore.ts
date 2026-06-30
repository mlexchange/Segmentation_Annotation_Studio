/**
 * Tool store — active tool, brush size, fill opacity, selection state.
 */
import { create } from 'zustand';

export type Tool = 'pan' | 'select' | 'polygon' | 'magnetic' | 'magic' | 'rectangle' | 'ellipse' | 'brush' | 'eraser';

export type MagicMode = 'contiguous' | 'global';
/** Magic-selection engine: SAM (learned object prior) or the classic wand. */
export type MagicEngine = 'sam' | 'classic';
/** SAM mask granularity — which of SAM's 3 multimask outputs to keep. */
export type SamDetail = 'auto' | 'fine' | 'medium' | 'coarse';

export interface ToolState {
  tool: Tool;
  brushSize: number;
  fillOpacity: number;
  /** Selected shape ids (multi-select via marquee/shift-click). */
  selectedShapeIds: string[];
  /** Magic-wand: similarity tolerance (0–1) and selection mode + denoise. */
  magicTolerance: number;
  magicMode: MagicMode;
  magicSigma: number;
  /** Edge barrier strength (0–1) for the contiguous magic wand. */
  magicEdgeStop: number;
  /** Active magic engine. Defaults to SAM; auto-falls back to 'classic'. */
  magicEngine: MagicEngine;
  /** SAM mask granularity (which multimask output to keep). */
  samDetail: SamDetail;
  /** SAM mask-logit threshold: >0 tightens the selection, <0 grows it. */
  samThreshold: number;
  /** Bumped to request the canvas re-fit the image to the viewport (F shortcut). */
  fitRequestId: number;
  setTool: (tool: Tool) => void;
  setBrushSize: (size: number) => void;
  setFillOpacity: (opacity: number) => void;
  /** Convenience single-select (clears to [] when null). */
  setSelectedShapeId: (id: string | null) => void;
  setSelectedShapeIds: (ids: string[]) => void;
  setMagicTolerance: (t: number) => void;
  setMagicMode: (m: MagicMode) => void;
  setMagicSigma: (s: number) => void;
  setMagicEdgeStop: (e: number) => void;
  setMagicEngine: (e: MagicEngine) => void;
  setSamDetail: (d: SamDetail) => void;
  setSamThreshold: (t: number) => void;
  requestFit: () => void;
}

export const useToolStore = create<ToolState>((set) => ({
  tool: 'pan',
  brushSize: 10,
  fillOpacity: 0.5,
  selectedShapeIds: [],
  magicTolerance: 0.08,
  magicMode: 'contiguous',
  magicSigma: 1,
  magicEdgeStop: 0.3,
  magicEngine: 'sam',
  samDetail: 'auto',
  samThreshold: 0,
  fitRequestId: 0,
  /** Switches the active tool and clears the current shape selection. */
  setTool: (tool) => set({ tool, selectedShapeIds: [] }),
  /** Sets the brush/eraser radius. */
  setBrushSize: (brushSize) => set({ brushSize }),
  /** Sets the shape fill opacity (0–1). */
  setFillOpacity: (fillOpacity) => set({ fillOpacity }),
  /** Single-selects a shape, or clears the selection when null. */
  setSelectedShapeId: (id) => set({ selectedShapeIds: id ? [id] : [] }),
  /** Replaces the multi-selection with the given shape ids. */
  setSelectedShapeIds: (selectedShapeIds) => set({ selectedShapeIds }),
  /** Sets the magic-wand similarity tolerance (0–1). */
  setMagicTolerance: (magicTolerance) => set({ magicTolerance }),
  /** Sets the magic-wand selection mode (contiguous/global). */
  setMagicMode: (magicMode) => set({ magicMode }),
  /** Sets the magic-wand pre-blur denoise sigma. */
  setMagicSigma: (magicSigma) => set({ magicSigma }),
  /** Sets the contiguous magic-wand edge-barrier strength (0–1). */
  setMagicEdgeStop: (magicEdgeStop) => set({ magicEdgeStop }),
  /** Selects the magic engine (SAM or classic wand). */
  setMagicEngine: (magicEngine) => set({ magicEngine }),
  /** Sets the SAM multimask granularity to keep. */
  setSamDetail: (samDetail) => set({ samDetail }),
  /** Sets the SAM mask-logit threshold (>0 tightens, <0 grows). */
  setSamThreshold: (samThreshold) => set({ samThreshold }),
  /** Bumps fitRequestId to signal the canvas to re-fit the image to the viewport. */
  requestFit: () => set((s) => ({ fitRequestId: s.fitRequestId + 1 })),
}));
