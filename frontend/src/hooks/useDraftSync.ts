/** Draft load + crash-recovery autosave with explicit load/error/dirty states. */
import { useEffect, useRef } from 'react';
import { create } from 'zustand';
import { API_BASE } from '@/config';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

const DEBOUNCE_MS = 1500;

export type DraftDocument = Record<string, unknown>;
export type DraftLoadResult =
  | { status: 'found'; draft: DraftDocument }
  | { status: 'not-found' }
  | { status: 'error'; error: Error };

interface DraftLoadEntry {
  status: 'loading' | DraftLoadResult['status'];
  revision: number;
  error?: Error;
}

interface DraftLoadState {
  bySource: Record<string, DraftLoadEntry>;
  begin: (sourceKey: string) => number;
  finish: (sourceKey: string, revision: number, result: DraftLoadResult) => void;
}

const useDraftLoadState = create<DraftLoadState>((set, get) => ({
  bySource: {},
  begin: (sourceKey) => {
    const revision = (get().bySource[sourceKey]?.revision ?? 0) + 1;
    set((state) => ({
      bySource: { ...state.bySource, [sourceKey]: { status: 'loading', revision } },
    }));
    return revision;
  },
  finish: (sourceKey, revision, result) => set((state) => {
    if (state.bySource[sourceKey]?.revision !== revision) return state;
    return {
      bySource: {
        ...state.bySource,
        [sourceKey]: {
          status: result.status,
          revision,
          ...(result.status === 'error' ? { error: result.error } : {}),
        },
      },
    };
  }),
}));

/** PUT a draft and reject on both network and HTTP failures. */
export async function putDraft(
  sourceKey: string,
  payload: unknown,
  options: { keepalive?: boolean } = {},
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: options.keepalive ?? false,
    },
  );
  if (!res.ok) throw new Error(`Draft save failed: ${res.status}`);
}

interface DraftPayload {
  classes: ReturnType<typeof useClassStore.getState>['classes'];
  slices: ReturnType<typeof useAnnotationStore.getState>['byImage'][string];
  split_by_slice: ReturnType<typeof useAnnotationStore.getState>['splitBySlice'][string];
  negative_slices: string[];
}

interface SaveSnapshot {
  sourceKey: string;
  identity: string;
  payload: DraftPayload;
  fingerprint: string;
  keepalive: boolean;
}

/** Autosave only after a successful load (or confirmed 404) and a real local edit. */
export function useDraftSync(sourceKey: string | null) {
  const byImage = useAnnotationStore((s) => s.byImage);
  const splitBySlice = useAnnotationStore((s) => s.splitBySlice);
  const negativeSlices = useAnnotationStore((s) => s.negativeSlices);
  const classes = useClassStore((s) => s.classes);
  const loadEntry = useDraftLoadState((s) => sourceKey ? s.bySource[sourceKey] : undefined);

  const ready = loadEntry?.status === 'found' || loadEntry?.status === 'not-found';
  const identity = sourceKey && ready ? `${sourceKey}:${loadEntry.revision}` : null;
  const payload: DraftPayload | null = sourceKey ? {
    classes,
    slices: byImage[sourceKey] ?? {},
    split_by_slice: splitBySlice[sourceKey] ?? {},
    negative_slices: negativeSlices[sourceKey] ?? [],
  } : null;
  const fingerprint = payload ? JSON.stringify(payload) : '';

  const baselineRef = useRef<{ identity: string; fingerprint: string } | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef<Promise<void> | null>(null);
  const queuedRef = useRef<SaveSnapshot | null>(null);
  const latestBySourceRef = useRef(new Map<string, SaveSnapshot>());

  if (sourceKey && identity && payload) {
    latestBySourceRef.current.set(sourceKey, {
      sourceKey, identity, payload, fingerprint, keepalive: false,
    });
  }

  const startSave = (snapshot: SaveSnapshot) => {
    const operation = putDraft(snapshot.sourceKey, snapshot.payload, {
      keepalive: snapshot.keepalive,
    })
      .then(() => {
        if (baselineRef.current?.identity === snapshot.identity) {
          baselineRef.current.fingerprint = snapshot.fingerprint;
        }
      })
      .catch((error: unknown) => {
        console.warn('Autosave failed:', error);
      })
      .finally(() => {
        if (inFlightRef.current !== operation) return;
        inFlightRef.current = null;
        const queued = queuedRef.current;
        queuedRef.current = null;
        if (queued) startSave(queued);
      });
    inFlightRef.current = operation;
  };

  const enqueue = (snapshot: SaveSnapshot) => {
    if (inFlightRef.current) {
      // Latest-wins queue: saves are serialized, while intermediate debounce states
      // that were never visible on the server need not be written one by one.
      queuedRef.current = snapshot;
      return;
    }
    startSave(snapshot);
  };

  const isDirtySnapshot = (snapshot: SaveSnapshot) =>
    baselineRef.current?.identity === snapshot.identity
    && baselineRef.current.fingerprint !== snapshot.fingerprint;

  // Establish a clean baseline once the load result is authoritative. This effect is
  // declared before the change effect so the loaded payload itself is never autosaved.
  useEffect(() => {
    if (!identity) return;
    baselineRef.current = { identity, fingerprint };
  // The fingerprint is intentionally captured only when a new load revision becomes ready.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity]);

  useEffect(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (!sourceKey || !identity || !payload) return;
    const snapshot = latestBySourceRef.current.get(sourceKey);
    if (!snapshot || !isDirtySnapshot(snapshot)) return;
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      enqueue(snapshot);
    }, DEBOUNCE_MS);
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, identity, fingerprint]);

  // Flush a pending dirty snapshot when leaving the source. Load failures never acquire
  // a ready identity/baseline, so they cannot write empty in-memory data on cleanup.
  useEffect(() => {
    if (!sourceKey || !identity) return;
    const sourceAtSetup = sourceKey;
    return () => {
      const snapshot = latestBySourceRef.current.get(sourceAtSetup);
      if (snapshot && isDirtySnapshot(snapshot)) enqueue(snapshot);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, identity]);

  useEffect(() => {
    if (!sourceKey || !identity) return;
    const flush = () => {
      const snapshot = latestBySourceRef.current.get(sourceKey);
      if (snapshot && isDirtySnapshot(snapshot)) enqueue({ ...snapshot, keepalive: true });
    };
    window.addEventListener('beforeunload', flush);
    return () => window.removeEventListener('beforeunload', flush);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, identity]);
}

/** Load a draft while preserving the difference between 404 and operational failure. */
export async function loadDraft(sourceKey: string): Promise<DraftLoadResult> {
  const revision = useDraftLoadState.getState().begin(sourceKey);
  let result: DraftLoadResult;
  try {
    const res = await fetch(
      `${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`,
    );
    if (res.status === 404) {
      result = { status: 'not-found' };
    } else if (!res.ok) {
      result = { status: 'error', error: new Error(`Draft load failed: ${res.status}`) };
    } else {
      result = { status: 'found', draft: await res.json() as DraftDocument };
    }
  } catch (error) {
    result = {
      status: 'error',
      error: error instanceof Error ? error : new Error('Draft load failed.'),
    };
  }
  useDraftLoadState.getState().finish(sourceKey, revision, result);
  return result;
}
