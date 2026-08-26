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
  /** Multiscale (Zarr) volumes only — which pyramid level is being displayed.
   *  `width`/`height`/`nSlices` above always describe the FINEST level, because
   *  annotations are stored in full-resolution coordinates whichever level is
   *  open; these describe the image actually drawn underneath them. */
  levelKey?: string | null;
  levelIndex?: number | null;
  levelCount?: number | null;
  levelWidth?: number | null;
  levelHeight?: number | null;
  levelNSlices?: number | null;
  /** Finest-z / level-z. When > 1, only every f-th full-res slice is addressable. */
  zDownsample?: number | null;
}

export interface RenderOpts {
  norm: 'slice' | 'global';
  scale: 'linear' | 'log' | 'symlog';
  vminPct: number;
  vmaxPct: number;
  cmap: 'gray' | 'viridis';
}

/**
 * Denoising applied to the slice before it is normalized for display.
 *
 * Deliberately NOT part of `RenderOpts`: render options travel into export
 * payloads, and a denoise preview must not silently change exported pixels. It
 * changes what you see (and what the intensity tools therefore act on) — turning
 * it into data is what the "Save denoised copy" bake is for.
 */
export interface DenoiseOpts {
  /** A `denoise.ALL_METHODS` entry; 'none' disables it. */
  method: string;
  /** 0..1, mapped by the backend onto each method's native parameter. */
  strength: number;
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
  denoise: DenoiseOpts;
  setDataset: (kind: string, source: string, serverUri: string | null, meta: ImageMeta) => void;
  setSlice: (idx: number) => void;
  setRenderOpts: (opts: Partial<RenderOpts>) => void;
  setDenoise: (opts: Partial<DenoiseOpts>) => void;
  reset: () => void;
}

const DEFAULT_RENDER: RenderOpts = {
  norm: 'global',
  scale: 'linear',
  vminPct: 1,
  vmaxPct: 99,
  cmap: 'gray',
};

const DEFAULT_DENOISE: DenoiseOpts = { method: 'none', strength: 0.5 };

export const useDatasetStore = create<DatasetState>((set) => ({
  kind: null,
  source: null,
  serverUri: null,
  meta: null,
  currentSlice: 0,
  renderOpts: { ...DEFAULT_RENDER },
  denoise: { ...DEFAULT_DENOISE },
  /** Activates a new image source and resets the current slice to 0.
   *  Denoising resets too: its strength is tuned to one volume's noise level and
   *  carrying it to the next would silently mis-filter it. */
  setDataset: (kind, source, serverUri, meta) =>
    set({ kind, source, serverUri, meta, currentSlice: 0, denoise: { ...DEFAULT_DENOISE } }),
  /** Sets the active slice index. */
  setSlice: (idx) => set({ currentSlice: idx }),
  /** Merges partial render options (normalization/scale/percentiles/cmap). */
  setRenderOpts: (opts) =>
    set((s) => ({ renderOpts: { ...s.renderOpts, ...opts } })),
  /** Merges partial denoise options (method/strength). */
  setDenoise: (opts) => set((s) => ({ denoise: { ...s.denoise, ...opts } })),
  /** Clears the active dataset and restores default render options. */
  reset: () =>
    set({
      kind: null, source: null, serverUri: null, meta: null, currentSlice: 0,
      renderOpts: { ...DEFAULT_RENDER }, denoise: { ...DEFAULT_DENOISE },
    }),
}));
