/**
 * useGuideSync — loads a dataset's annotation guide on open and autosaves edits.
 *
 * Mirrors useDraftSync: the guide is dataset-scoped and keyed by sourceKey. On
 * mount (and whenever sourceKey changes) the guide is fetched into the store;
 * subsequent edits are debounced and PUT back. Saving is gated on `loadedFor`
 * matching the current sourceKey so we never clobber a dataset's guide before
 * its own guide has finished loading.
 */
import { useEffect, useRef } from 'react';
import { API_BASE } from '@/config';
import { useReferenceGuideStore, type GuideClass } from '@/stores/referenceGuideStore';

const DEBOUNCE_MS = 1000;

/** Fetch a dataset's guide. Returns null if none exists (404). */
export async function loadGuide(
  sourceKey: string,
): Promise<{ classes: GuideClass[]; notes: string } | null> {
  try {
    const res = await fetch(`${API_BASE}/api/guide?source_key=${encodeURIComponent(sourceKey)}`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`Guide load failed: ${res.status}`);
    const doc = await res.json();
    const guide = doc.guide ?? {};
    return { classes: guide.classes ?? [], notes: guide.notes ?? '' };
  } catch {
    return null;
  }
}

/** Payload shape accepted by the guide generator (a draft or version snapshot). */
export interface GuideGenPayload {
  classes: { classId: number; label: string; color: string; isVisible?: boolean }[];
  slices: Record<string, unknown[]>;
  split_by_slice?: Record<string, string>;
  negative_slices?: string[];
}

/**
 * Ask the backend to build a guide skeleton (per-class example crops) from an
 * annotation. Throws with a descriptive message on failure so callers can show it.
 */
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

/** Persist a dataset's guide. */
export async function saveGuide(
  sourceKey: string,
  entries: GuideClass[],
  notes: string,
): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/guide?source_key=${encodeURIComponent(sourceKey)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classes: entries, notes }),
    });
    return res.ok;
  } catch (err) {
    console.warn('Guide autosave failed:', err);
    return false;
  }
}

/**
 * Loads the guide for `sourceKey` into the store (read-only). Use in tabs that
 * consume the guide (e.g. Annotate) but must not autosave it.
 */
export function useGuideLoad(sourceKey: string | null) {
  const setGuide = useReferenceGuideStore((s) => s.setGuide);
  const clear = useReferenceGuideStore((s) => s.clear);
  useEffect(() => {
    if (!sourceKey) { clear(); return; }
    let cancelled = false;
    loadGuide(sourceKey).then((g) => {
      if (cancelled) return;
      setGuide(g?.classes ?? [], g?.notes ?? '', sourceKey);
    });
    return () => { cancelled = true; };
  }, [sourceKey, setGuide, clear]);
}

/** Loads the guide for `sourceKey` into the store and autosaves edits back. */
export function useGuideSync(sourceKey: string | null) {
  const entries = useReferenceGuideStore((s) => s.entries);
  const notes = useReferenceGuideStore((s) => s.notes);
  const loadedFor = useReferenceGuideStore((s) => s.loadedFor);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useGuideLoad(sourceKey);

  // Debounced autosave — only once the current source's guide has loaded.
  useEffect(() => {
    if (!sourceKey || loadedFor !== sourceKey) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => { saveGuide(sourceKey, entries, notes); }, DEBOUNCE_MS);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [sourceKey, loadedFor, entries, notes]);
}
