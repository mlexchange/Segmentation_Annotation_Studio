/**
 * Dataset store — active image source, shape metadata, current slice, render options.
 */
import { create } from 'zustand';

export interface ImageMeta {
  nSlices: number;
  height: number;
  width: number;
  dtype: string;
  isRgb: boolean;
  valueRange: [number, number];
}

export interface RenderOpts {
  norm: 'slice' | 'global';
  scale: 'linear' | 'log' | 'symlog';
  vminPct: number;
  vmaxPct: number;
  cmap: 'gray' | 'viridis';
}

export interface DatasetState {
  /** 'tiled' or 'local' */
  kind: string | null;
  /** Tiled path or local relative path */
  source: string | null;
  serverUri: string | null;
  meta: ImageMeta | null;
  currentSlice: number;
  renderOpts: RenderOpts;
  setDataset: (kind: string, source: string, serverUri: string | null, meta: ImageMeta) => void;
  setSlice: (idx: number) => void;
  setRenderOpts: (opts: Partial<RenderOpts>) => void;
  reset: () => void;
}

const DEFAULT_RENDER: RenderOpts = {
  norm: 'global',
  scale: 'linear',
  vminPct: 1,
  vmaxPct: 99,
  cmap: 'gray',
};

export const useDatasetStore = create<DatasetState>((set) => ({
  kind: null,
  source: null,
  serverUri: null,
  meta: null,
  currentSlice: 0,
  renderOpts: { ...DEFAULT_RENDER },
  setDataset: (kind, source, serverUri, meta) =>
    set({ kind, source, serverUri, meta, currentSlice: 0 }),
  setSlice: (idx) => set({ currentSlice: idx }),
  setRenderOpts: (opts) =>
    set((s) => ({ renderOpts: { ...s.renderOpts, ...opts } })),
  reset: () =>
    set({ kind: null, source: null, serverUri: null, meta: null, currentSlice: 0, renderOpts: { ...DEFAULT_RENDER } }),
}));
