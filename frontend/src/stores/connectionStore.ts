/**
 * connectionStore — active data-source connection + session prefs.
 *
 * Separate from datasetStore (active sample). Connect once; browse + annotate many.
 * Session prefs (label names/colors, preferred model + its feature recipe) apply when opening samples.
 */
import { create } from 'zustand';
import type { AnnotationClass } from '@/stores/classStore';

export interface TrainerConfig {
  iterations: number;
  depth: number;
  learning_rate: number;
}

export const DEFAULT_TRAINER_CONFIG: TrainerConfig = {
  iterations: 200,
  depth: 6,
  learning_rate: 0.1,
};

export interface ConnectionState {
  kind: 'tiled' | 'local' | null;
  /** Tiled server URI */
  serverUri: string | null;
  /** Tiled container to browse (e.g. "browse/myset"); null = auto-discover */
  browseContainerPath: string | null;
  /** Granted absolute browse root (local mode) */
  localRoot: string | null;
  /** Chosen subfolder relative to localRoot (local mode) */
  localRel: string | null;
  /** Human-readable display label */
  label: string | null;
  /** Total number of samples reported by /api/connect/summary */
  sampleCount: number | null;
  // ##########################################################################
  // # REMOVE THIS AND USE YOUR OWN STUFF
  // Session prefs for scaffold label-sets + shelf models (Connect/Browse).
  // ##########################################################################
  /** Selected reusable label set id (disk). REMOVE — use your taxonomy. */
  preferredLabelSetId: string | null;
  /** Label name + color defs from preferred set. REMOVE — use your taxonomy. */
  preferredClasses: AnnotationClass[] | null;
  /** Optional CatBoost shelf model to prefer on Train. REMOVE — use your model UX. */
  preferredShelfModelId: string | null;
  preferredShelfModelName: string | null;
  /** Feature recipe required by the preferred model (for Preprocess compute). */
  preferredFeatureRecipe: Record<string, unknown> | null;
  /** Preferred ipred Feature Setup id (legacy procedure|weights|composition). */
  preferredFeatureSetupId: string | null;
  /** Preferred composition document id (modular feature graph). */
  preferredCompositionId: string | null;
  /** Preferred trainer plugin id (ipred). */
  preferredTrainerId: string;
  preferredTrainerConfig: TrainerConfig;
  /** Active ipred session for the opened sample (project). */
  ipredSessionId: string | null;
  ipredProjectId: string | null;
  setConnection: (payload: {
    kind: 'tiled' | 'local';
    serverUri?: string | null;
    browseContainerPath?: string | null;
    localRoot?: string | null;
    localRel?: string | null;
    label: string;
    sampleCount: number;
  }) => void;
  setPreferredLabelSet: (payload: {
    id: string | null;
    classes: AnnotationClass[] | null;
  }) => void;
  /** # REMOVE THIS AND USE YOUR OWN STUFF — scaffold shelf model picker. */
  setPreferredShelfModel: (payload: {
    id: string | null;
    name?: string | null;
    featureRecipe?: Record<string, unknown> | null;
  }) => void;
  setPreferredFeatureSetupId: (id: string | null) => void;
  setPreferredCompositionId: (id: string | null) => void;
  setPreferredTrainer: (payload: {
    id?: string;
    config?: Partial<TrainerConfig>;
  }) => void;
  setIpredSession: (payload: {
    sessionId: string | null;
    projectId?: string | null;
  }) => void;
  clearConnection: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  kind: null,
  serverUri: null,
  browseContainerPath: null,
  localRoot: null,
  localRel: null,
  label: null,
  sampleCount: null,
  preferredLabelSetId: null,
  preferredClasses: null,
  preferredShelfModelId: null,
  preferredShelfModelName: null,
  preferredFeatureRecipe: null,
  preferredFeatureSetupId: 'default-skimage-slimsam',
  preferredCompositionId: 'comp-skimage-slimsam',
  preferredTrainerId: 'catboost',
  preferredTrainerConfig: { ...DEFAULT_TRAINER_CONFIG },
  ipredSessionId: null,
  ipredProjectId: null,

  setConnection: ({
    kind,
    serverUri = null,
    browseContainerPath = null,
    localRoot = null,
    localRel = null,
    label,
    sampleCount,
  }) =>
    set({ kind, serverUri, browseContainerPath, localRoot, localRel, label, sampleCount }),

  setPreferredLabelSet: ({ id, classes }) =>
    set({ preferredLabelSetId: id, preferredClasses: classes }),

  setPreferredShelfModel: ({ id, name = null, featureRecipe = null }) =>
    set({
      preferredShelfModelId: id,
      preferredShelfModelName: name,
      preferredFeatureRecipe: featureRecipe,
    }),

  setPreferredFeatureSetupId: (id) => set({ preferredFeatureSetupId: id }),

  setPreferredCompositionId: (id) =>
    set({
      preferredCompositionId: id,
      // Keep legacy key in sync when selecting a composition
      preferredFeatureSetupId: id,
    }),

  setPreferredTrainer: ({ id, config }) =>
    set((s) => ({
      preferredTrainerId: id ?? s.preferredTrainerId,
      preferredTrainerConfig: config
        ? { ...s.preferredTrainerConfig, ...config }
        : s.preferredTrainerConfig,
    })),

  setIpredSession: ({ sessionId, projectId = null }) =>
    set({ ipredSessionId: sessionId, ipredProjectId: projectId }),

  clearConnection: () =>
    set({
      kind: null,
      serverUri: null,
      browseContainerPath: null,
      localRoot: null,
      localRel: null,
      label: null,
      sampleCount: null,
      preferredLabelSetId: null,
      preferredClasses: null,
      preferredShelfModelId: null,
      preferredShelfModelName: null,
      preferredFeatureRecipe: null,
      preferredFeatureSetupId: 'default-skimage-slimsam',
      preferredCompositionId: 'comp-skimage-slimsam',
      preferredTrainerId: 'catboost',
      preferredTrainerConfig: { ...DEFAULT_TRAINER_CONFIG },
      ipredSessionId: null,
      ipredProjectId: null,
    }),
}));
