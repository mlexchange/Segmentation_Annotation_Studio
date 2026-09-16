/**
 * predictedRasterStore — lightweight pointers to un-vectorized predicted
 * regions from an iPred "Apply across volume" run, keyed by sample + slice.
 *
 * This is the direct fix for the 297MB-draft incident (139,004 shapes from
 * one volume-apply commit, 98.8% predicted-origin): "Commit predicted
 * shapes" used to eagerly fetch + vectorize every slice's commit.png into
 * real `Shape[]` objects added to `annotationStore` — for a 690-slice
 * volume that's hundreds of thousands of polygon objects landing straight
 * in the autosaved draft, most of which nobody ever looks at let alone
 * edits.
 *
 * Deliberately a SEPARATE store from `annotationStore`/`useDraftSync`: a
 * pointer here is just `{runId, classIds}` (tens of bytes), and — critically
 * — is never included in the autosaved draft payload, so committing however
 * many slices costs nothing until a specific slice is actually vectorized
 * (see `usePixelClassifier.ts`'s live-preview effect, which renders a
 * pointer's commit.png directly via the existing Predictions layer, and
 * AnnotatePage's "Make this slice editable" action, the only path that ever
 * turns a pointer into real `Shape[]`).
 *
 * Not persisted across page reloads on purpose — same rationale as
 * `annotationStore`'s draft being the autosave surface: an un-vectorized
 * pointer is recoverable at any time by re-running "Apply across volume" or
 * committing again, unlike hand-drawn shapes.
 */
import { create } from 'zustand';

export interface PredictedRegionPointer {
  /** ipred run id — `ipredRunCommitUrl(runId)` fetches its raw label-map PNG. */
  runId: string;
  /** Frontend class ids this run's commit.png pixel values are drawn from
   *  (matches `labelMapToPolygonShapes`'s `classIds` param exactly — pixel
   *  value IS the class id, not a 1-based sequential index). */
  classIds: number[];
}

export interface PredictedRasterState {
  /** sourceKey -> sliceIndex (as string) -> pointer. */
  bySource: Record<string, Record<string, PredictedRegionPointer>>;
  /** Replace/merge pointers for many slices of one sample at once (a
   *  "Commit predicted shapes" click). */
  setPointers: (sourceKey: string, pointers: Record<string, PredictedRegionPointer>) => void;
  /** Drop one slice's pointer — called once it's been vectorized into real
   *  `Shape[]` (or the user wants to discard the prediction for that slice). */
  clearSlice: (sourceKey: string, sliceKey: string) => void;
  /** Drop every pointer for a sample — a fresh volume-apply run supersedes
   *  whatever was committed before. */
  clearSource: (sourceKey: string) => void;
}

export const usePredictedRasterStore = create<PredictedRasterState>((set) => ({
  bySource: {},

  setPointers: (sourceKey, pointers) =>
    set((s) => ({
      bySource: {
        ...s.bySource,
        [sourceKey]: { ...(s.bySource[sourceKey] ?? {}), ...pointers },
      },
    })),

  clearSlice: (sourceKey, sliceKey) =>
    set((s) => {
      const existing = s.bySource[sourceKey];
      if (!existing || !(sliceKey in existing)) return s;
      const next = { ...existing };
      delete next[sliceKey];
      return { bySource: { ...s.bySource, [sourceKey]: next } };
    }),

  clearSource: (sourceKey) =>
    set((s) => {
      if (!(sourceKey in s.bySource)) return s;
      const next = { ...s.bySource };
      delete next[sourceKey];
      return { bySource: next };
    }),
}));
