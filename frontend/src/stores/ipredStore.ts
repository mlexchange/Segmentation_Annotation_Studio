/**
 * ipredStore — preferences + session state for the iPred (interactive
 * segmentation) service. Separate from connectionStore (data-source
 * connection) and datasetStore (active sample), per this codebase's existing
 * store-separation convention.
 */
import { create } from 'zustand';

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

export const DEFAULT_COMPOSITION_ID = 'comp-skimage-slimsam';
export const DEFAULT_TRAINER_ID = 'catboost';

export interface IpredState {
  /** Preferred composition document id (modular feature graph). */
  preferredCompositionId: string;
  /** Preferred trainer plugin id. */
  preferredTrainerId: string;
  preferredTrainerConfig: TrainerConfig;
  /** Active ipred session for the currently opened sample. */
  ipredSessionId: string | null;
  ipredProjectId: string | null;
  setPreferredCompositionId: (id: string) => void;
  setPreferredTrainer: (payload: { id?: string; config?: Partial<TrainerConfig> }) => void;
  setIpredSession: (payload: { sessionId: string | null; projectId?: string | null }) => void;
  reset: () => void;
}

export const useIpredStore = create<IpredState>((set) => ({
  preferredCompositionId: DEFAULT_COMPOSITION_ID,
  preferredTrainerId: DEFAULT_TRAINER_ID,
  preferredTrainerConfig: { ...DEFAULT_TRAINER_CONFIG },
  ipredSessionId: null,
  ipredProjectId: null,

  setPreferredCompositionId: (id) => set({ preferredCompositionId: id }),

  setPreferredTrainer: ({ id, config }) =>
    set((s) => ({
      preferredTrainerId: id ?? s.preferredTrainerId,
      preferredTrainerConfig: config
        ? { ...s.preferredTrainerConfig, ...config }
        : s.preferredTrainerConfig,
    })),

  setIpredSession: ({ sessionId, projectId = null }) =>
    set({ ipredSessionId: sessionId, ipredProjectId: projectId }),

  reset: () =>
    set({
      preferredCompositionId: DEFAULT_COMPOSITION_ID,
      preferredTrainerId: DEFAULT_TRAINER_ID,
      preferredTrainerConfig: { ...DEFAULT_TRAINER_CONFIG },
      ipredSessionId: null,
      ipredProjectId: null,
    }),
}));
