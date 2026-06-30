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
  setConnection: (payload: {
    kind: 'tiled' | 'local';
    serverUri?: string | null;
    browseContainerPath?: string | null;
    localRoot?: string | null;
    localRel?: string | null;
    label: string;
    sampleCount: number;
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

  /** Records the active data-source connection; unspecified fields default to null. */
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

  /** Resets all connection fields to null (disconnect). */
  clearConnection: () =>
    set({
      kind: null,
      serverUri: null,
      browseContainerPath: null,
      localRoot: null,
      localRel: null,
      label: null,
      sampleCount: null,
    }),
}));
