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
  /** Selected shape ids (multi-select via marquee/shift-click). */
  selectedShapeIds: string[];
  /** Magic-wand: similarity tolerance (0–1) and selection mode + denoise. */
  magicTolerance: number;
  magicMode: MagicMode;
  magicSigma: number;
  /** Edge barrier strength (0–1) for the contiguous magic wand. */
  magicEdgeStop: number;
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
  fitRequestId: 0,
  setTool: (tool) => set({ tool, selectedShapeIds: [] }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setFillOpacity: (fillOpacity) => set({ fillOpacity }),
  setSelectedShapeId: (id) => set({ selectedShapeIds: id ? [id] : [] }),
  setSelectedShapeIds: (selectedShapeIds) => set({ selectedShapeIds }),
  setMagicTolerance: (magicTolerance) => set({ magicTolerance }),
  setMagicMode: (magicMode) => set({ magicMode }),
  setMagicSigma: (magicSigma) => set({ magicSigma }),
  setMagicEdgeStop: (magicEdgeStop) => set({ magicEdgeStop }),
  requestFit: () => set((s) => ({ fitRequestId: s.fitRequestId + 1 })),
}));
