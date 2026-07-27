/**
 * useDraftSync — debounced autosave to /api/annotations/draft + restore on connect.
 */
import { useEffect, useRef } from 'react';
import { API_BASE } from '@/config';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

const DEBOUNCE_MS = 1500;

/** PUT the draft payload for a sourceKey. Fire-and-forget; a fetch already in flight
 *  completes even if the component unmounts, so a flush-on-leave isn't dropped. */
function putDraft(sourceKey: string, payload: unknown) {
  return fetch(`${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch((err) => console.warn('Autosave failed:', err));
}

/**
 * Autosaves the current annotation stores for `sourceKey` to the draft endpoint,
 * debounced 1.5s on change. Also flushes immediately when the sample changes or the
 * component unmounts (so a quick edit-then-navigate isn't lost to a cancelled debounce)
 * and via sendBeacon on page unload. No-op when sourceKey is null.
 */
export function useDraftSync(sourceKey: string | null) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const byImage = useAnnotationStore((s) => s.byImage);
  const splitBySlice = useAnnotationStore((s) => s.splitBySlice);
  const negativeSlices = useAnnotationStore((s) => s.negativeSlices);
  const classes = useClassStore((s) => s.classes);

  // Mirror the latest state in a ref so the flush-on-leave effect (keyed only on
  // sourceKey) can read current data without re-subscribing — otherwise its cleanup
  // would fire on every edit, not just when the sample actually changes.
  const latestRef = useRef({ byImage, splitBySlice, negativeSlices, classes });
  latestRef.current = { byImage, splitBySlice, negativeSlices, classes };

  const buildPayload = (sk: string) => {
    const state = latestRef.current;
    return {
      classes: state.classes,
      slices: state.byImage[sk] ?? {},
      split_by_slice: state.splitBySlice[sk] ?? {},
      negative_slices: state.negativeSlices[sk] ?? [],
    };
  };

  // Debounced autosave: re-arm on any change, save 1.5s after the last one.
  useEffect(() => {
    if (!sourceKey) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      putDraft(sourceKey, buildPayload(sourceKey));
      timerRef.current = null;
    }, DEBOUNCE_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // buildPayload reads via ref; re-arm only on data/sourceKey change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, byImage, splitBySlice, negativeSlices, classes]);

  // Flush any pending save when the sample changes or the component unmounts. Without
  // this, navigating away within the debounce window silently drops the last edits
  // (e.g. deleting classes then returning to Browse before the 1.5s timer fires).
  useEffect(() => {
    if (!sourceKey) return;
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      putDraft(sourceKey, buildPayload(sourceKey));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey]);

  // Flush on full page unload (browser close/refresh — no React unmount fires).
  useEffect(() => {
    if (!sourceKey) return;
    const flush = () => {
      navigator.sendBeacon(
        `${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`,
        JSON.stringify(buildPayload(sourceKey))
      );
    };
    window.addEventListener('beforeunload', flush);
    return () => window.removeEventListener('beforeunload', flush);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey]);
}

/** Load a draft from the server for sourceKey. Returns null if none exists. */
export async function loadDraft(sourceKey: string): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(
      `${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`
    );
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`Draft load failed: ${res.status}`);
    return await res.json();
  } catch {
    return null;
  }
}
