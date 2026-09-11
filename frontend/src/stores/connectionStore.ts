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
  /**
   * Full path of a sample to auto-select on arriving in Browse (e.g.
   * "browse/myset"). Set when jumping in from ingest so the user lands on the
   * dataset they just touched, while Browse still lists from the root — where a
   * sample is the whole volume and its annotations resolve.
   */
  browseFocusPath: string | null;
  /** Granted absolute browse root (local mode) */
  localRoot: string | null;
  /** Chosen subfolder relative to localRoot (local mode) */
  localRel: string | null;
  /** Human-readable display label */
  label: string | null;
  /** Total number of samples reported by /api/connect/summary */
  sampleCount: number | null;
  /**
   * Live Tiled reachability, driven by a periodic health check (see
   * useConnectionHealth). 'unknown' until the first check resolves, and
   * always 'unknown' for local connections (no network dependency to check).
   */
  status: 'unknown' | 'ok' | 'error';
  setStatus: (status: 'unknown' | 'ok' | 'error') => void;
  setConnection: (payload: {
    kind: 'tiled' | 'local';
    serverUri?: string | null;
    browseContainerPath?: string | null;
    browseFocusPath?: string | null;
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
  browseFocusPath: null,
  localRoot: null,
  localRel: null,
  label: null,
  sampleCount: null,
  status: 'unknown',

  setStatus: (status) => set({ status }),

  /** Records the active data-source connection; unspecified fields default to null. */
  setConnection: ({
    kind,
    serverUri = null,
    browseContainerPath = null,
    browseFocusPath = null,
    localRoot = null,
    localRel = null,
    label,
    sampleCount,
  }) =>
    set({
      kind,
      serverUri,
      browseContainerPath,
      browseFocusPath,
      localRoot,
      localRel,
      label,
      sampleCount,
      status: 'unknown',
    }),

  /** Resets all connection fields to null (disconnect). */
  clearConnection: () =>
    set({
      kind: null,
      serverUri: null,
      browseContainerPath: null,
      browseFocusPath: null,
      localRoot: null,
      localRel: null,
      label: null,
      status: 'unknown',
      sampleCount: null,
    }),
}));
