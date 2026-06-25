/**
 * Tool store — active tool, brush size, fill opacity, selection state.
 */
import { create } from 'zustand';

export type Tool = 'pan' | 'select' | 'polygon' | 'magnetic' | 'magic' | 'rectangle' | 'ellipse' | 'brush' | 'eraser';

export type MagicMode = 'contiguous' | 'global';

export interface ToolState {
  tool: Tool;
  brushSize: number;
  fillOpacity: number;
  selectedShapeId: string | null;
  /** Magic-wand: similarity tolerance (0–1) and selection mode + denoise. */
  magicTolerance: number;
  magicMode: MagicMode;
  magicSigma: number;
  /** Bumped to request the canvas re-fit the image to the viewport (F shortcut). */
  fitRequestId: number;
  setTool: (tool: Tool) => void;
  setBrushSize: (size: number) => void;
  setFillOpacity: (opacity: number) => void;
  setSelectedShapeId: (id: string | null) => void;
  setMagicTolerance: (t: number) => void;
  setMagicMode: (m: MagicMode) => void;
  setMagicSigma: (s: number) => void;
  requestFit: () => void;
}

export const useToolStore = create<ToolState>((set) => ({
  tool: 'pan',
  brushSize: 10,
  fillOpacity: 0.5,
  selectedShapeId: null,
  magicTolerance: 0.08,
  magicMode: 'contiguous',
  magicSigma: 1,
  fitRequestId: 0,
  setTool: (tool) => set({ tool, selectedShapeId: null }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setFillOpacity: (fillOpacity) => set({ fillOpacity }),
  setSelectedShapeId: (selectedShapeId) => set({ selectedShapeId }),
  setMagicTolerance: (magicTolerance) => set({ magicTolerance }),
  setMagicMode: (magicMode) => set({ magicMode }),
  setMagicSigma: (magicSigma) => set({ magicSigma }),
  requestFit: () => set((s) => ({ fitRequestId: s.fitRequestId + 1 })),
}));
