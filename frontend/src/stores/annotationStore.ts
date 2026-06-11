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

export interface BaseShape {
  id: string;
  classId: number;
  kind: Shape['kind'];
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
  appendBrushStroke: (sourceKey: string, sliceIdx: number, shapeId: string, stroke: BrushStroke) => void;
  setShapes: (sourceKey: string, sliceIdx: number, shapes: Shape[]) => void;
  setSplitForSlice: (sourceKey: string, sliceIdx: number, split: Split | 'auto') => void;
  toggleNegativeSlice: (sourceKey: string, sliceIdx: number) => void;
  loadFromDraft: (draft: Pick<AnnotationState, 'byImage' | 'splitBySlice' | 'negativeSlices'>) => void;
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

      reset: () => set({ byImage: {}, splitBySlice: {}, negativeSlices: {} }),
    }),
    { limit: 200, partialize: (s) => ({ byImage: s.byImage }) }
  )
);
