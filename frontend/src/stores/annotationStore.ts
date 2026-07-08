/**
 * Annotation store — per-(image,slice) shapes, split map, negative slices.
 * Wrapped in zundo temporal middleware for undo/redo (limit 200).
 */
import { create } from 'zustand';
import { temporal } from 'zundo';
import { v4 as uuidv4 } from 'uuid';

// ---- Shape types ----

export interface BrushStroke {
  points: number[];
  radius: number;
  mode: 'paint' | 'erase';
}

/** An erase carve-out applied to any vector shape (polygon/rect/ellipse). */
export interface EraseStroke {
  points: number[];
  radius: number;
}

export interface BaseShape {
  id: string;
  classId: number;
  kind: Shape['kind'];
  /** Optional erase carve-outs (rendered destination-out, subtracted on export). */
  erased?: EraseStroke[];
}

export interface PolygonShape extends BaseShape {
  kind: 'polygon';
  points: number[];
  /** Optional inner rings (flat [x,y,…] image coords) carved out of the outer
   *  polygon — e.g. produced by "invert shape". Rendered/rasterized even-odd. */
  holes?: number[][];
}

export interface RectShape extends BaseShape {
  kind: 'rectangle';
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EllipseShape extends BaseShape {
  kind: 'ellipse';
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

export interface BrushShape extends BaseShape {
  kind: 'brush';
  strokes: BrushStroke[];
}

export type Shape = PolygonShape | RectShape | EllipseShape | BrushShape;
export type Split = 'train' | 'valid' | 'test';

/** Deep-clone a shape with a fresh id (used when copying across slices). */
function cloneShapeWithNewId(shape: Shape): Shape {
  return { ...structuredClone(shape), id: uuidv4() };
}

export interface AnnotationState {
  byImage: Record<string, Record<string, Shape[]>>;
  splitBySlice: Record<string, Record<string, Split | 'auto'>>;
  negativeSlices: Record<string, string[]>;

  addShape: (sourceKey: string, sliceIdx: number, shape: Shape) => void;
  /** Append several shapes in one update (one undo step) — used by magic-wand. */
  addShapes: (sourceKey: string, sliceIdx: number, shapes: Shape[]) => void;
  removeShape: (sourceKey: string, sliceIdx: number, shapeId: string) => void;
  /** Remove several shapes in one update (one undo step). */
  removeShapes: (sourceKey: string, sliceIdx: number, shapeIds: string[]) => void;
  /** Replace a single shape via an updater (used for move / vertex editing). */
  updateShape: (sourceKey: string, sliceIdx: number, shapeId: string, updater: (shape: Shape) => Shape) => void;
  /** Reassign several shapes to a class in one update (one undo step). */
  setClassForShapes: (sourceKey: string, sliceIdx: number, shapeIds: string[], classId: number) => void;
  /** Replace every shape of *classId* on a slice with *shapes* (one undo step).
   *  Used by threshold/cleanup/interpolate to write a recomputed region back. */
  replaceClassShapesOnSlice: (sourceKey: string, sliceIdx: number, classId: number, shapes: Shape[]) => void;
  /** Clone shapes from *fromSlice* into each *toSlices* index (fresh ids, merged
   *  with existing), optionally limited to one class. One undo step. */
  copySliceShapes: (sourceKey: string, fromSlice: number, toSlices: number[], classId?: number | null) => void;
  /** Remove every shape with *classId* across all loaded samples (all slices). */
  removeShapesByClassId: (classId: number) => void;
  appendBrushStroke: (sourceKey: string, sliceIdx: number, shapeId: string, stroke: BrushStroke) => void;
  /** Append an erase carve-out to any shape (brush → erase stroke; vector → `erased`). */
  appendEraseStroke: (sourceKey: string, sliceIdx: number, shapeId: string, stroke: EraseStroke) => void;
  setShapes: (sourceKey: string, sliceIdx: number, shapes: Shape[]) => void;
  setSplitForSlice: (sourceKey: string, sliceIdx: number, split: Split | 'auto') => void;
  toggleNegativeSlice: (sourceKey: string, sliceIdx: number) => void;
  /** Replaces ALL annotation data (used only at initial session restore). */
  loadFromDraft: (draft: Pick<AnnotationState, 'byImage' | 'splitBySlice' | 'negativeSlices'>) => void;
  /** Merges a single sample's annotation data into the store without clearing other samples. */
  mergeSourceDraft: (sourceKey: string, slices: Record<string, Shape[]>, splitBySlice: Record<string, string>, negativeSlices: string[]) => void;
  reset: () => void;
}

export const useAnnotationStore = create<AnnotationState>()(
  temporal(
    (set) => ({
      byImage: {},
      splitBySlice: {},
      negativeSlices: {},

      /** Appends one shape to the given (sourceKey, slice); one undo step. */
      addShape: (sourceKey, sliceIdx, shape) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: [...prev, shape],
              },
            },
          };
        }),

      /** Appends several shapes in one update (one undo step); used by magic-wand. */
      addShapes: (sourceKey, sliceIdx, shapes) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: [...prev, ...shapes],
              },
            },
          };
        }),

      /** Removes the shape with the given id from the (sourceKey, slice). */
      removeShape: (sourceKey, sliceIdx, shapeId) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: prev.filter((sh) => sh.id !== shapeId),
              },
            },
          };
        }),

      /** Removes several shapes by id in one update (one undo step). */
      removeShapes: (sourceKey, sliceIdx, shapeIds) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          const drop = new Set(shapeIds);
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: prev.filter((sh) => !drop.has(sh.id)),
              },
            },
          };
        }),

      /** Replaces a single shape via the updater fn (move / vertex edit). */
      updateShape: (sourceKey, sliceIdx, shapeId, updater) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: prev.map((sh) => (sh.id === shapeId ? updater(sh) : sh)),
              },
            },
          };
        }),

      /** Reassigns every shape in *shapeIds* to *classId* (one undo step). */
      setClassForShapes: (sourceKey, sliceIdx, shapeIds, classId) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          const ids = new Set(shapeIds);
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: prev.map((sh) => (ids.has(sh.id) ? { ...sh, classId } : sh)),
              },
            },
          };
        }),

      /** Replaces all shapes of *classId* on one slice with *shapes* (one undo step). */
      replaceClassShapesOnSlice: (sourceKey, sliceIdx, classId, shapes) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.byImage[sourceKey]?.[sliceKey] ?? [];
          const kept = prev.filter((sh) => sh.classId !== classId);
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: [...kept, ...shapes],
              },
            },
          };
        }),

      /** Clones shapes (fresh ids) from *fromSlice* into each *toSlices* index. */
      copySliceShapes: (sourceKey, fromSlice, toSlices, classId = null) =>
        set((s) => {
          const src = s.byImage[sourceKey]?.[String(fromSlice)] ?? [];
          const picked = classId == null ? src : src.filter((sh) => sh.classId === classId);
          if (picked.length === 0) return {};
          const slices = { ...(s.byImage[sourceKey] ?? {}) };
          for (const t of toSlices) {
            if (t === fromSlice) continue;
            const key = String(t);
            slices[key] = [...(slices[key] ?? []), ...picked.map(cloneShapeWithNewId)];
          }
          return { byImage: { ...s.byImage, [sourceKey]: slices } };
        }),

      /** Removes every shape of *classId* across all samples/slices, pruning emptied slices and sources. */
      removeShapesByClassId: (classId) =>
        set((s) => {
          const nextByImage: Record<string, Record<string, Shape[]>> = {};
          for (const [sourceKey, slices] of Object.entries(s.byImage)) {
            const nextSlices: Record<string, Shape[]> = {};
            for (const [sliceKey, shapes] of Object.entries(slices)) {
              const filtered = shapes.filter((sh) => sh.classId !== classId);
              if (filtered.length > 0) {
                nextSlices[sliceKey] = filtered;
              }
            }
            if (Object.keys(nextSlices).length > 0) {
              nextByImage[sourceKey] = nextSlices;
            }
          }
          return { byImage: nextByImage };
        }),

      /** Appends a brush stroke to the named brush shape; no-op for non-brush shapes. */
      appendBrushStroke: (sourceKey, sliceIdx, shapeId, stroke) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const shapes = s.byImage[sourceKey]?.[sliceKey] ?? [];
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: shapes.map((sh) =>
                  sh.id === shapeId && sh.kind === 'brush'
                    ? { ...sh, strokes: [...sh.strokes, stroke] }
                    : sh
                ),
              },
            },
          };
        }),

      /** Appends an erase carve-out: brush shapes get an erase stroke, vector shapes get an `erased` entry. */
      appendEraseStroke: (sourceKey, sliceIdx, shapeId, stroke) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const shapes = s.byImage[sourceKey]?.[sliceKey] ?? [];
          return {
            byImage: {
              ...s.byImage,
              [sourceKey]: {
                ...(s.byImage[sourceKey] ?? {}),
                [sliceKey]: shapes.map((sh) => {
                  if (sh.id !== shapeId) return sh;
                  if (sh.kind === 'brush') {
                    return { ...sh, strokes: [...sh.strokes, { ...stroke, mode: 'erase' as const }] };
                  }
                  return { ...sh, erased: [...(sh.erased ?? []), stroke] };
                }),
              },
            },
          };
        }),

      /** Replaces all shapes for the given (sourceKey, slice). */
      setShapes: (sourceKey, sliceIdx, shapes) =>
        set((s) => ({
          byImage: {
            ...s.byImage,
            [sourceKey]: {
              ...(s.byImage[sourceKey] ?? {}),
              [String(sliceIdx)]: shapes,
            },
          },
        })),

      /** Sets the train/valid/test split (or 'auto') for one slice. */
      setSplitForSlice: (sourceKey, sliceIdx, split) =>
        set((s) => ({
          splitBySlice: {
            ...s.splitBySlice,
            [sourceKey]: {
              ...(s.splitBySlice[sourceKey] ?? {}),
              [String(sliceIdx)]: split,
            },
          },
        })),

      /** Toggles whether a slice is flagged as a negative (background-only) example. */
      toggleNegativeSlice: (sourceKey, sliceIdx) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.negativeSlices[sourceKey] ?? [];
          const next = prev.includes(sliceKey)
            ? prev.filter((k) => k !== sliceKey)
            : [...prev, sliceKey];
          return { negativeSlices: { ...s.negativeSlices, [sourceKey]: next } };
        }),

      /** Replaces ALL annotation data wholesale; used only on initial session restore. */
      loadFromDraft: (draft) =>
        set({ byImage: draft.byImage, splitBySlice: draft.splitBySlice, negativeSlices: draft.negativeSlices }),

      /** Merges one sample's data into the store, leaving other samples untouched.
       *  The draft's split map arrives loosely typed (JSON); its values are always
       *  'train' | 'valid' | 'test' | 'auto', so we narrow it here. */
      mergeSourceDraft: (sourceKey, slices, splitMap, negSlices) =>
        set((s) => ({
          byImage: { ...s.byImage, [sourceKey]: slices },
          splitBySlice: { ...s.splitBySlice, [sourceKey]: splitMap as Record<string, Split | 'auto'> },
          negativeSlices: { ...s.negativeSlices, [sourceKey]: negSlices },
        })),

      /** Clears all shapes, splits, and negative-slice flags. */
      reset: () => set({ byImage: {}, splitBySlice: {}, negativeSlices: {} }),
    }),
    { limit: 200, partialize: (s) => ({ byImage: s.byImage }) }
  )
);
