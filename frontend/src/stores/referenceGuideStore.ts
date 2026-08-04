/**
 * Reference-guide store — the project lead's per-dataset annotation guide.
 *
 * Each entry describes one class (label, color, a written description of what it
 * is / how it looks, and example image crops). The guide is dataset-scoped and
 * persisted on the backend keyed by sourceKey (see useGuideSync). Its classes are
 * offered as one-click suggestions in the Annotate tab so annotators stay
 * consistent with the lead's intended labels and colors.
 */
import { create } from 'zustand';
import { sanitizeExampleCrops } from '@/lib/guideCrops';

export interface GuideClass {
  label: string;
  color: string;
  description: string;
  /** Base64 data-URL PNG crops illustrating the class. */
  exampleCrops: string[];
}

export type GuideLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

interface ReferenceGuideStore {
  entries: GuideClass[];
  notes: string;
  /** sourceKey the current guide was loaded for (null until a load resolves). */
  loadedFor: string | null;
  loadStatus: GuideLoadStatus;
  loadError: string | null;
  loadRevision: number;
  beginLoad: (sourceKey: string) => number;
  setLoadError: (sourceKey: string, revision: number, error: string) => void;
  setGuide: (entries: GuideClass[], notes: string, loadedFor: string) => void;
  addEntry: (entry: GuideClass) => void;
  updateEntry: (index: number, updates: Partial<GuideClass>) => void;
  removeEntry: (index: number) => void;
  setNotes: (notes: string) => void;
  clear: () => void;
  /**
   * Merge auto-generated entries in: match existing entries by label (keeping any
   * description the lead already wrote), refresh their color + example crops, and
   * append newly-discovered classes. Existing classes absent from the generation
   * are preserved.
   */
  applyGenerated: (generated: GuideClass[]) => void;
}

export const useReferenceGuideStore = create<ReferenceGuideStore>((set) => ({
  entries: [],
  notes: '',
  loadedFor: null,
  loadStatus: 'idle',
  loadError: null,
  loadRevision: 0,
  beginLoad: (_sourceKey) => {
    let revision = 0;
    set((state) => {
      revision = state.loadRevision + 1;
      return {
        entries: [], notes: '', loadedFor: null,
        loadStatus: 'loading', loadError: null, loadRevision: revision,
      };
    });
    return revision;
  },
  setLoadError: (_sourceKey, revision, error) => set((state) =>
    state.loadRevision === revision
      ? { loadStatus: 'error', loadError: error, loadedFor: null }
      : state,
  ),
  /** Replaces the whole guide (used when a dataset's guide loads from the server,
   *  or when a shared guide JSON file is imported — both are untrusted content
   *  as far as exampleCrops go, so they're sanitized before entering the store
   *  that feeds `<img src>` on the Reference/Class Manager pages). */
  setGuide: (entries, notes, loadedFor) => set({
    entries: entries.map((e) => ({ ...e, exampleCrops: sanitizeExampleCrops(e.exampleCrops) })),
    notes, loadedFor, loadStatus: 'ready', loadError: null,
  }),
  addEntry: (entry) => set((s) => ({
    entries: [...s.entries, { ...entry, exampleCrops: sanitizeExampleCrops(entry.exampleCrops) }],
  })),
  updateEntry: (index, updates) =>
    set((s) => ({
      entries: s.entries.map((e, i) => (i === index ? {
        ...e,
        ...updates,
        ...(updates.exampleCrops ? { exampleCrops: sanitizeExampleCrops(updates.exampleCrops) } : {}),
      } : e)),
    })),
  removeEntry: (index) => set((s) => ({ entries: s.entries.filter((_, i) => i !== index) })),
  setNotes: (notes) => set({ notes }),
  /** Resets to an empty guide with no loaded source (e.g. when no dataset is open). */
  clear: () => set({
    entries: [], notes: '', loadedFor: null, loadStatus: 'idle', loadError: null,
  }),
  applyGenerated: (generated) =>
    set((s) => {
      const byLabel = new Map(s.entries.map((e) => [e.label.trim().toLowerCase(), e]));
      const merged: GuideClass[] = [];
      const consumed = new Set<string>();
      for (const g of generated) {
        const key = g.label.trim().toLowerCase();
        const existing = byLabel.get(key);
        consumed.add(key);
        const generatedCrops = sanitizeExampleCrops(g.exampleCrops);
        merged.push({
          label: g.label,
          color: g.color || existing?.color || '#1f77b4',
          // Keep the lead's written description; only take the generated one if none yet.
          description: existing?.description || g.description || '',
          exampleCrops: generatedCrops.length ? generatedCrops : existing?.exampleCrops ?? [],
        });
      }
      // Preserve existing entries that weren't part of this generation.
      for (const e of s.entries) {
        if (!consumed.has(e.label.trim().toLowerCase())) merged.push(e);
      }
      return { entries: merged };
    }),
}));
