/**
 * Canvas layer visibility — function groups + per-class toggles.
 *
 * Groups control image / denoise / features / probability / predictions /
 * annotations / manifold. Per-class keys under predictions and annotations
 * (annotations also sync with classStore.isVisible).
 */
import { create } from 'zustand';

export type LayerGroupId =
  | 'image'
  | 'denoise'
  | 'features'
  | 'proba'
  | 'predictions'
  | 'annotations'
  | 'manifold';

export const LAYER_GROUP_META: Record<
  LayerGroupId,
  { label: string; hint: string }
> = {
  image: { label: 'Image', hint: 'Base slice (brightness/levels apply here)' },
  denoise: { label: 'Denoise', hint: 'Server-side denoising of the base slice' },
  features: { label: 'Features', hint: 'Preprocess channel as display base' },
  proba: { label: 'Probability', hint: 'Softmax class heatmap overlay' },
  predictions: { label: 'Predictions', hint: 'Conformal singleton / multi / abstain' },
  annotations: { label: 'Annotations', hint: 'Drawn shapes / scribbles' },
  manifold: { label: 'Suggest', hint: 'Manifold heatmap + markers' },
};

export interface LayerVisibilityState {
  groups: Record<LayerGroupId, boolean>;
  /** Opacity of the probability overlay (0–1). */
  probaOpacity: number;
  /** Opacity of the prediction overlay (0–1). */
  predictionsOpacity: number;
  /** Per-class visibility for prediction singletons. Missing → visible. */
  predictionClassVisible: Record<number, boolean>;
  /** Show conformal multi-set hatch. */
  showPredictionMulti: boolean;
  /** Show conformal abstain. */
  showPredictionAbstain: boolean;
  /** Annotations-layer sub-toggle: show human-drawn vs. iPred-predicted shapes
   *  independently (see `ShapeOrigin` in annotationStore). Both default visible. */
  annotationOriginVisible: { human: boolean; predicted: boolean };
  setAnnotationOriginVisible: (origin: 'human' | 'predicted', visible: boolean) => void;
  setGroup: (id: LayerGroupId, visible: boolean) => void;
  toggleGroup: (id: LayerGroupId) => void;
  setProbaOpacity: (v: number) => void;
  setPredictionsOpacity: (v: number) => void;
  setPredictionClassVisible: (classId: number, visible: boolean) => void;
  togglePredictionClass: (classId: number) => void;
  setShowPredictionMulti: (v: boolean) => void;
  setShowPredictionAbstain: (v: boolean) => void;
  ensurePredictionClasses: (classIds: number[]) => void;
}

const DEFAULT_GROUPS: Record<LayerGroupId, boolean> = {
  image: true,
  denoise: true,
  features: true,
  proba: true,
  predictions: true,
  annotations: true,
  manifold: true,
};

export const useLayerVisibilityStore = create<LayerVisibilityState>((set, get) => ({
  groups: { ...DEFAULT_GROUPS },
  probaOpacity: 0.55,
  predictionsOpacity: 0.85,
  predictionClassVisible: {},
  showPredictionMulti: true,
  showPredictionAbstain: true,
  annotationOriginVisible: { human: true, predicted: true },

  setAnnotationOriginVisible: (origin, visible) =>
    set((s) => ({
      annotationOriginVisible: { ...s.annotationOriginVisible, [origin]: visible },
    })),

  setGroup: (id, visible) =>
    set((s) => ({ groups: { ...s.groups, [id]: visible } })),

  toggleGroup: (id) => {
    const cur = get().groups[id];
    set((s) => ({ groups: { ...s.groups, [id]: !cur } }));
  },

  setProbaOpacity: (v) =>
    set({ probaOpacity: Math.min(1, Math.max(0, v)) }),

  setPredictionsOpacity: (v) =>
    set({ predictionsOpacity: Math.min(1, Math.max(0, v)) }),

  setPredictionClassVisible: (classId, visible) =>
    set((s) => ({
      predictionClassVisible: { ...s.predictionClassVisible, [classId]: visible },
    })),

  togglePredictionClass: (classId) => {
    const cur = get().predictionClassVisible[classId];
    const next = cur === undefined ? false : !cur;
    set((s) => ({
      predictionClassVisible: { ...s.predictionClassVisible, [classId]: next },
    }));
  },

  setShowPredictionMulti: (v) => set({ showPredictionMulti: v }),
  setShowPredictionAbstain: (v) => set({ showPredictionAbstain: v }),

  ensurePredictionClasses: (classIds) =>
    set((s) => {
      const next = { ...s.predictionClassVisible };
      let changed = false;
      for (const id of classIds) {
        if (next[id] === undefined) {
          next[id] = true;
          changed = true;
        }
      }
      return changed ? { predictionClassVisible: next } : s;
    }),
}));

/** True when a prediction class should draw (default visible). */
export function isPredictionClassVisible(
  map: Record<number, boolean>,
  classId: number,
): boolean {
  return map[classId] !== false;
}

/** True when a shape's origin (human/predicted; undefined = human) should draw. */
export function isShapeOriginVisible(
  visible: { human: boolean; predicted: boolean },
  origin: 'human' | 'predicted' | undefined,
): boolean {
  return origin === 'predicted' ? visible.predicted : visible.human;
}
