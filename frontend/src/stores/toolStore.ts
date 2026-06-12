/**
 * Tool store — active tool, brush size, fill opacity, selection state.
 */
import { create } from 'zustand';

export type Tool = 'pan' | 'select' | 'polygon' | 'rectangle' | 'ellipse' | 'brush' | 'eraser';

export interface ToolState {
  tool: Tool;
  brushSize: number;
  fillOpacity: number;
  selectedShapeId: string | null;
  /** Bumped to request the canvas re-fit the image to the viewport (F shortcut). */
  fitRequestId: number;
  setTool: (tool: Tool) => void;
  setBrushSize: (size: number) => void;
  setFillOpacity: (opacity: number) => void;
  setSelectedShapeId: (id: string | null) => void;
  requestFit: () => void;
}

export const useToolStore = create<ToolState>((set) => ({
  tool: 'pan',
  brushSize: 10,
  fillOpacity: 0.5,
  selectedShapeId: null,
  fitRequestId: 0,
  setTool: (tool) => set({ tool, selectedShapeId: null }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setFillOpacity: (fillOpacity) => set({ fillOpacity }),
  setSelectedShapeId: (selectedShapeId) => set({ selectedShapeId }),
  requestFit: () => set((s) => ({ fitRequestId: s.fitRequestId + 1 })),
}));
