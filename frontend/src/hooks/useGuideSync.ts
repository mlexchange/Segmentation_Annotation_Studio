/** Dataset-scoped annotation-guide loading, generation, and dirty-only autosave. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE } from '@/config';
import { useReferenceGuideStore, type GuideClass } from '@/stores/referenceGuideStore';

const DEBOUNCE_MS = 1000;

export type GuideLoadResult =
  | { status: 'found'; guide: { classes: GuideClass[]; notes: string } }
  | { status: 'not-found' }
  | { status: 'error'; error: Error };

/** Fetch a guide while preserving the difference between 404 and operational failure. */
export async function loadGuide(sourceKey: string): Promise<GuideLoadResult> {
  try {
    const res = await fetch(`${API_BASE}/api/guide?source_key=${encodeURIComponent(sourceKey)}`);
    if (res.status === 404) return { status: 'not-found' };
    if (!res.ok) return { status: 'error', error: new Error(`Guide load failed: ${res.status}`) };
    const doc = await res.json();
    const guide = doc.guide ?? {};
    return {
      status: 'found',
      guide: { classes: guide.classes ?? [], notes: guide.notes ?? '' },
    };
  } catch (error) {
    return {
      status: 'error',
      error: error instanceof Error ? error : new Error('Guide load failed.'),
    };
  }
}

/** Payload shape accepted by the guide generator (a draft or version snapshot). */
export interface GuideGenPayload {
  classes: { classId: number; label: string; color: string; isVisible?: boolean }[];
  slices: Record<string, unknown[]>;
  split_by_slice?: Record<string, string>;
  negative_slices?: string[];
}

/** Ask the backend to build a guide skeleton from an annotation snapshot. */
export async function generateGuide(
  sourceKey: string,
  payload: GuideGenPayload,
): Promise<{ classes: GuideClass[]; notes: string }> {
  const res = await fetch(
    `${API_BASE}/api/guide/generate?source_key=${encodeURIComponent(sourceKey)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  );
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Guide generation failed (HTTP ${res.status})${detail ? `: ${detail}` : ''}`);
  }
  const guide = await res.json();
  return { classes: guide.classes ?? [], notes: guide.notes ?? '' };
}

/** Persist a dataset's guide and report both network and HTTP failures. */
export async function saveGuide(
  sourceKey: string,
  entries: GuideClass[],
  notes: string,
  options: { keepalive?: boolean } = {},
): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/guide?source_key=${encodeURIComponent(sourceKey)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classes: entries, notes }),
      keepalive: options.keepalive ?? false,
    });
    return res.ok;
  } catch (error) {
    console.warn('Guide autosave failed:', error);
    return false;
  }
}

export interface GuideLoadController {
  status: ReturnType<typeof useReferenceGuideStore.getState>['loadStatus'];
  error: string | null;
  retry: () => void;
}

/** Load the current source guide without enabling writes. */
export function useGuideLoad(sourceKey: string | null): GuideLoadController {
  const beginLoad = useReferenceGuideStore((s) => s.beginLoad);
  const setGuide = useReferenceGuideStore((s) => s.setGuide);
  const setLoadError = useReferenceGuideStore((s) => s.setLoadError);
  const clear = useReferenceGuideStore((s) => s.clear);
  const status = useReferenceGuideStore((s) => s.loadStatus);
  const error = useReferenceGuideStore((s) => s.loadError);
  const [retryToken, setRetryToken] = useState(0);
  const retry = useCallback(() => setRetryToken((value) => value + 1), []);

  useEffect(() => {
    if (!sourceKey) {
      clear();
      return;
    }
    const revision = beginLoad(sourceKey);
    let cancelled = false;
    loadGuide(sourceKey).then((result) => {
      if (cancelled) return;
      if (result.status === 'error') {
        setLoadError(sourceKey, revision, result.error.message);
      } else if (result.status === 'not-found') {
        setGuide([], '', sourceKey);
      } else {
        setGuide(result.guide.classes, result.guide.notes, sourceKey);
      }
    });
    return () => { cancelled = true; };
  }, [sourceKey, retryToken, beginLoad, setGuide, setLoadError, clear]);

  return { status, error, retry };
}

interface GuideSnapshot {
  sourceKey: string;
  identity: string;
  entries: GuideClass[];
  notes: string;
  fingerprint: string;
  keepalive: boolean;
}

/** Load the guide and autosave only dirty edits made after an authoritative load. */
export function useGuideSync(sourceKey: string | null): GuideLoadController {
  const entries = useReferenceGuideStore((s) => s.entries);
  const notes = useReferenceGuideStore((s) => s.notes);
  const loadedFor = useReferenceGuideStore((s) => s.loadedFor);
  const loadStatus = useReferenceGuideStore((s) => s.loadStatus);
  const loadRevision = useReferenceGuideStore((s) => s.loadRevision);
  const controller = useGuideLoad(sourceKey);
  const identity = sourceKey && loadedFor === sourceKey && loadStatus === 'ready'
    ? `${sourceKey}:${loadRevision}`
    : null;
  const fingerprint = JSON.stringify({ entries, notes });

  const baselineRef = useRef<{ identity: string; fingerprint: string } | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef<Promise<void> | null>(null);
  const queuedRef = useRef<GuideSnapshot | null>(null);
  const latestBySourceRef = useRef(new Map<string, GuideSnapshot>());

  if (sourceKey && identity) {
    latestBySourceRef.current.set(sourceKey, {
      sourceKey, identity, entries, notes, fingerprint, keepalive: false,
    });
  }

  const startSave = (snapshot: GuideSnapshot) => {
    const operation = saveGuide(
      snapshot.sourceKey,
      snapshot.entries,
      snapshot.notes,
      { keepalive: snapshot.keepalive },
    )
      .then((ok) => {
        if (ok && baselineRef.current?.identity === snapshot.identity) {
          baselineRef.current.fingerprint = snapshot.fingerprint;
        }
        if (!ok) console.warn('Guide autosave failed: server rejected the request.');
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

  const enqueue = (snapshot: GuideSnapshot) => {
    if (inFlightRef.current) queuedRef.current = snapshot;
    else startSave(snapshot);
  };
  const isDirty = (snapshot: GuideSnapshot) =>
    baselineRef.current?.identity === snapshot.identity
    && baselineRef.current.fingerprint !== snapshot.fingerprint;

  useEffect(() => {
    if (identity) baselineRef.current = { identity, fingerprint };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity]);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!sourceKey || !identity) return;
    const snapshot = latestBySourceRef.current.get(sourceKey);
    if (!snapshot || !isDirty(snapshot)) return;
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

  useEffect(() => {
    if (!sourceKey || !identity) return;
    const sourceAtSetup = sourceKey;
    return () => {
      const snapshot = latestBySourceRef.current.get(sourceAtSetup);
      if (snapshot && isDirty(snapshot)) enqueue({ ...snapshot, keepalive: true });
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, identity]);

  return controller;
}
