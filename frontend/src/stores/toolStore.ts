/**
 * Tool store — active tool, brush size, fill opacity, selection state.
 */
import { create } from 'zustand';

export type Tool = 'pan' | 'select' | 'polygon' | 'magnetic' | 'magic' | 'rectangle' | 'ellipse' | 'brush' | 'fill' | 'eraser';

export type MagicMode = 'contiguous' | 'global';
/** Magic-selection engine: SAM (learned object prior) or the classic wand. */
export type MagicEngine = 'sam' | 'classic';
/** SAM mask granularity — which of SAM's 3 multimask outputs to keep. */
export type SamDetail = 'auto' | 'fine' | 'medium' | 'coarse';
/** Select-tool scope: only the active class, or every class. */
export type SelectScope = 'class' | 'all';

export interface ToolState {
  tool: Tool;
  brushSize: number;
  fillOpacity: number;
  /** Fill (paint-bucket) similarity threshold (0–1): higher fills more. */
  fillThreshold: number;
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
  /** When true, feed interior points of other-class regions to SAM as negative
   *  ("not") prompts so a new selection won't bleed into already-labeled areas. */
  samAvoidLabeled: boolean;
  /** When true, keep only the SAM component(s) containing a positive prompt (else
   *  the largest) — drops detached speckle regions. */
  samConnectedOnly: boolean;
  /** When true, new annotations are clipped so they can't overlap other classes'
   *  regions on the current slice (neighbor classes act as a hard boundary). */
  clipToOtherClasses: boolean;
  /** When true, a new annotation that overlaps existing shapes of the SAME class
   *  is unioned with them into one merged shape on commit. */
  mergeOverlappingSameClass: boolean;
  /** Eraser scope: when true the eraser carves any visible class under the cursor;
   *  when false (default) it only carves shapes of the active class. */
  eraseAllClasses: boolean;
  /** Select-tool scope for "Select all" (Cmd/Ctrl+A): active class only, or all. */
  selectScope: SelectScope;
  /** Tool to revert to after a hold-Space pan; also lets the canvas keep the
   *  brush/eraser cursor visible while `tool==='pan'`. Null when not space-panning. */
  panReturnTool: Tool | null;
  /** Bumped to request the canvas re-fit the image to the viewport (F shortcut). */
  fitRequestId: number;
  setTool: (tool: Tool) => void;
  setBrushSize: (size: number) => void;
  setFillOpacity: (opacity: number) => void;
  setFillThreshold: (t: number) => void;
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
  setSamAvoidLabeled: (v: boolean) => void;
  setSamConnectedOnly: (v: boolean) => void;
  setClipToOtherClasses: (v: boolean) => void;
  setMergeOverlappingSameClass: (v: boolean) => void;
  setEraseAllClasses: (v: boolean) => void;
  setSelectScope: (v: SelectScope) => void;
  setPanReturnTool: (t: Tool | null) => void;
  requestFit: () => void;
}

export const useToolStore = create<ToolState>((set) => ({
  tool: 'pan',
  brushSize: 10,
  fillOpacity: 0.5,
  fillThreshold: 0.1,
  selectedShapeIds: [],
  magicTolerance: 0.08,
  magicMode: 'contiguous',
  magicSigma: 1,
  magicEdgeStop: 0.3,
  magicEngine: 'sam',
  samDetail: 'auto',
  samThreshold: 0,
  samAvoidLabeled: true,
  samConnectedOnly: true,
  clipToOtherClasses: true,
  mergeOverlappingSameClass: false,
  eraseAllClasses: false,
  selectScope: 'all',
  panReturnTool: null,
  fitRequestId: 0,
  /** Switches the active tool and clears the current shape selection. */
  setTool: (tool) => set({ tool, selectedShapeIds: [] }),
  /** Sets the brush/eraser radius. */
  setBrushSize: (brushSize) => set({ brushSize }),
  /** Sets the shape fill opacity (0–1). */
  setFillOpacity: (fillOpacity) => set({ fillOpacity }),
  /** Sets the Fill (paint-bucket) similarity threshold (0–1). */
  setFillThreshold: (fillThreshold) => set({ fillThreshold }),
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
  /** Toggles using other-class regions as SAM negative ("not") prompts. */
  setSamAvoidLabeled: (samAvoidLabeled) => set({ samAvoidLabeled }),
  /** Toggles keeping only SAM component(s) at a positive prompt (drops speckle). */
  setSamConnectedOnly: (samConnectedOnly) => set({ samConnectedOnly }),
  /** Toggles clipping new annotations against other classes' regions. */
  setClipToOtherClasses: (clipToOtherClasses) => set({ clipToOtherClasses }),
  /** Toggles auto-merging new annotations with overlapping same-class shapes. */
  setMergeOverlappingSameClass: (mergeOverlappingSameClass) => set({ mergeOverlappingSameClass }),
  /** Toggles whether the eraser carves any visible class or just the active one. */
  setEraseAllClasses: (eraseAllClasses) => set({ eraseAllClasses }),
  /** Sets the select-tool "Select all" scope (active class vs all classes). */
  setSelectScope: (selectScope) => set({ selectScope }),
  /** Records the tool to revert to after a hold-Space pan (null clears it). */
  setPanReturnTool: (panReturnTool) => set({ panReturnTool }),
  /** Bumps fitRequestId to signal the canvas to re-fit the image to the viewport. */
  requestFit: () => set((s) => ({ fitRequestId: s.fitRequestId + 1 })),
}));
