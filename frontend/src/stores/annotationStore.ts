/**
 * Annotation store — per-(image,slice) shapes, split map, negative slices.
 * Wrapped in zundo temporal middleware for undo/redo (limit 200).
 */
import { create } from 'zustand';
import { temporal } from 'zundo';

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

export interface AnnotationState {
  byImage: Record<string, Record<string, Shape[]>>;
  splitBySlice: Record<string, Record<string, Split | 'auto'>>;
  negativeSlices: Record<string, string[]>;

  addShape: (sourceKey: string, sliceIdx: number, shape: Shape) => void;
  removeShape: (sourceKey: string, sliceIdx: number, shapeId: string) => void;
  /** Replace a single shape via an updater (used for move / vertex editing). */
  updateShape: (sourceKey: string, sliceIdx: number, shapeId: string, updater: (shape: Shape) => Shape) => void;
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

      toggleNegativeSlice: (sourceKey, sliceIdx) =>
        set((s) => {
          const sliceKey = String(sliceIdx);
          const prev = s.negativeSlices[sourceKey] ?? [];
          const next = prev.includes(sliceKey)
            ? prev.filter((k) => k !== sliceKey)
            : [...prev, sliceKey];
          return { negativeSlices: { ...s.negativeSlices, [sourceKey]: next } };
        }),

      loadFromDraft: (draft) =>
        set({ byImage: draft.byImage, splitBySlice: draft.splitBySlice, negativeSlices: draft.negativeSlices }),

      mergeSourceDraft: (sourceKey, slices, splitMap, negSlices) =>
        set((s) => ({
          byImage: { ...s.byImage, [sourceKey]: slices },
          splitBySlice: { ...s.splitBySlice, [sourceKey]: splitMap },
          negativeSlices: { ...s.negativeSlices, [sourceKey]: negSlices },
        })),

      reset: () => set({ byImage: {}, splitBySlice: {}, negativeSlices: {} }),
    }),
    { limit: 200, partialize: (s) => ({ byImage: s.byImage }) }
  )
);
