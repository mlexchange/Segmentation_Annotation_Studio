/**
 * Named mask sets for Cleanup Studio — dense pixel label maps (+ optional shapes).
 */
import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import type { Shape } from '@/stores/annotationStore';
import { labelMapFromBase64, labelMapToBase64 } from '@/lib/labelMap';

export type MaskSetOrigin = 'clf' | 'manual' | 'merge' | 'import' | 'shelf';

export interface MaskSet {
  id: string;
  name: string;
  sourceKey: string;
  slice: number;
  origin: MaskSetOrigin;
  /** Legacy / cached polygons — prefer labelMap for cleanup edits. */
  shapes: Shape[];
  /** Full-res classId map (HxW). Primary Cleanup buffer. */
  labelMap: Uint8Array | null;
  width: number;
  height: number;
  createdAt: number;
  visible: boolean;
}

/** JSON-friendly draft form of a mask set. */
export interface MaskSetSerialized {
  id: string;
  name: string;
  sourceKey: string;
  slice: number;
  origin: MaskSetOrigin;
  shapes: Shape[];
  label_map_b64?: string | null;
  width: number;
  height: number;
  createdAt: number;
  visible: boolean;
}

export function serializeMaskSet(m: MaskSet): MaskSetSerialized {
  return {
    id: m.id,
    name: m.name,
    sourceKey: m.sourceKey,
    slice: m.slice,
    origin: m.origin,
    shapes: m.shapes,
    label_map_b64: m.labelMap ? labelMapToBase64(m.labelMap) : null,
    width: m.width,
    height: m.height,
    createdAt: m.createdAt,
    visible: m.visible,
  };
}

export function deserializeMaskSet(raw: MaskSetSerialized | MaskSet): MaskSet {
  const r = raw as MaskSetSerialized & Partial<MaskSet>;
  let labelMap: Uint8Array | null = null;
  if (r.labelMap instanceof Uint8Array) {
    labelMap = r.labelMap;
  } else if (typeof r.label_map_b64 === 'string' && r.label_map_b64 && r.width && r.height) {
    try {
      labelMap = labelMapFromBase64(r.label_map_b64, r.width * r.height);
    } catch {
      labelMap = null;
    }
  }
  return {
    id: r.id,
    name: r.name,
    sourceKey: r.sourceKey,
    slice: r.slice,
    origin: r.origin,
    shapes: Array.isArray(r.shapes) ? r.shapes : [],
    labelMap,
    width: Number(r.width) || 0,
    height: Number(r.height) || 0,
    createdAt: Number(r.createdAt) || Date.now(),
    visible: r.visible !== false,
  };
}

type AddSetInput = {
  name: string;
  sourceKey: string;
  slice: number;
  origin: MaskSetOrigin;
  shapes?: Shape[];
  labelMap?: Uint8Array | null;
  width?: number;
  height?: number;
  visible?: boolean;
};

interface MaskSetState {
  sets: MaskSet[];
  activeSetId: string | null;
  addSet: (input: AddSetInput) => string;
  updateSetShapes: (id: string, shapes: Shape[]) => void;
  updateSetLabelMap: (id: string, labelMap: Uint8Array, width: number, height: number) => void;
  renameSet: (id: string, name: string) => void;
  deleteSet: (id: string) => void;
  duplicateSet: (id: string) => string | null;
  setActiveSetId: (id: string | null) => void;
  setVisible: (id: string, visible: boolean) => void;
  replaceAll: (sets: MaskSet[]) => void;
  clearSource: (sourceKey: string) => void;
}

export const useMaskSetStore = create<MaskSetState>((set, get) => ({
  sets: [],
  activeSetId: null,

  addSet: (input) => {
    const id = uuidv4();
    const entry: MaskSet = {
      id,
      name: input.name,
      sourceKey: input.sourceKey,
      slice: input.slice,
      origin: input.origin,
      shapes: input.shapes ?? [],
      labelMap: input.labelMap ?? null,
      width: input.width ?? 0,
      height: input.height ?? 0,
      createdAt: Date.now(),
      visible: input.visible ?? true,
    };
    set((s) => ({ sets: [...s.sets, entry], activeSetId: id }));
    return id;
  },

  updateSetShapes: (id, shapes) =>
    set((s) => ({
      sets: s.sets.map((m) => (m.id === id ? { ...m, shapes } : m)),
    })),

  updateSetLabelMap: (id, labelMap, width, height) =>
    set((s) => ({
      sets: s.sets.map((m) =>
        m.id === id
          ? { ...m, labelMap: new Uint8Array(labelMap), width, height, shapes: [] }
          : m,
      ),
    })),

  renameSet: (id, name) =>
    set((s) => ({
      sets: s.sets.map((m) => (m.id === id ? { ...m, name } : m)),
    })),

  deleteSet: (id) =>
    set((s) => ({
      sets: s.sets.filter((m) => m.id !== id),
      activeSetId: s.activeSetId === id ? null : s.activeSetId,
    })),

  duplicateSet: (id) => {
    const src = get().sets.find((m) => m.id === id);
    if (!src) return null;
    return get().addSet({
      name: `${src.name} copy`,
      sourceKey: src.sourceKey,
      slice: src.slice,
      origin: src.origin,
      shapes: src.shapes.map((sh) => ({ ...sh, id: uuidv4() })),
      labelMap: src.labelMap ? new Uint8Array(src.labelMap) : null,
      width: src.width,
      height: src.height,
    });
  },

  setActiveSetId: (id) => set({ activeSetId: id }),

  setVisible: (id, visible) =>
    set((s) => ({
      sets: s.sets.map((m) => (m.id === id ? { ...m, visible } : m)),
    })),

  replaceAll: (sets) =>
    set({
      sets,
      activeSetId: sets[0]?.id ?? null,
    }),

  clearSource: (sourceKey) =>
    set((s) => {
      const sets = s.sets.filter((m) => m.sourceKey !== sourceKey);
      const activeStill = sets.some((m) => m.id === s.activeSetId);
      return { sets, activeSetId: activeStill ? s.activeSetId : null };
    }),
}));

/** Sets for a source + slice. */
export function setsForSlice(sets: MaskSet[], sourceKey: string, slice: number): MaskSet[] {
  return sets.filter((m) => m.sourceKey === sourceKey && m.slice === slice);
}
