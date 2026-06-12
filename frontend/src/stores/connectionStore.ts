/**
 * connectionStore — active data-source connection (server or local folder).
 *
 * Separate from datasetStore (which holds the *active sample* being annotated).
 * Connect once; browse + annotate many samples.
 */
import { create } from 'zustand';

export interface ConnectionState {
  kind: 'tiled' | 'local' | null;
  /** Tiled server URI */
  serverUri: string | null;
  /** Relative folder path under LOCAL_DATA_ROOT (local mode) */
  localRoot: string | null;
  /** Human-readable display label */
  label: string | null;
  /** Total number of samples reported by /api/connect/summary */
  sampleCount: number | null;
  setConnection: (payload: {
    kind: 'tiled' | 'local';
    serverUri?: string | null;
    localRoot?: string | null;
    label: string;
    sampleCount: number;
  }) => void;
  clearConnection: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  kind: null,
  serverUri: null,
  localRoot: null,
  label: null,
  sampleCount: null,

  setConnection: ({ kind, serverUri = null, localRoot = null, label, sampleCount }) =>
    set({ kind, serverUri, localRoot, label, sampleCount }),

  clearConnection: () =>
    set({ kind: null, serverUri: null, localRoot: null, label: null, sampleCount: null }),
}));
