import { create } from 'zustand';
import type { AnnotationClass } from './classStore';

interface SourceTaxonomyState {
  classesBySource: Record<string, AnnotationClass[]>;
  recordTaxonomy: (sourceKey: string, classes: AnnotationClass[]) => void;
  reset: () => void;
}

/** Session-scoped taxonomy snapshots used to prevent ambiguous multi-sample exports. */
export const useSourceTaxonomyStore = create<SourceTaxonomyState>((set) => ({
  classesBySource: {},
  recordTaxonomy: (sourceKey, classes) => set((state) => ({
    classesBySource: { ...state.classesBySource, [sourceKey]: structuredClone(classes) },
  })),
  reset: () => set({ classesBySource: {} }),
}));
