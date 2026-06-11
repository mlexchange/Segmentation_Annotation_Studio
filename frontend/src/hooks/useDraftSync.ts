/**
 * useDraftSync — debounced autosave to /api/annotations/draft + restore on connect.
 */
import { useEffect, useRef } from 'react';
import { API_BASE } from '@/config';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

const DEBOUNCE_MS = 1500;

export function useDraftSync(sourceKey: string | null) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const byImage = useAnnotationStore((s) => s.byImage);
  const splitBySlice = useAnnotationStore((s) => s.splitBySlice);
  const negativeSlices = useAnnotationStore((s) => s.negativeSlices);
  const classes = useClassStore((s) => s.classes);

  useEffect(() => {
    if (!sourceKey) return;

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      const payload = {
        classes,
        slices: byImage[sourceKey] ?? {},
        split_by_slice: splitBySlice[sourceKey] ?? {},
        negative_slices: negativeSlices[sourceKey] ?? [],
      };
      fetch(`${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).catch((err) => console.warn('Autosave failed:', err));
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [sourceKey, byImage, splitBySlice, negativeSlices, classes]);

  // Flush on page unload
  useEffect(() => {
    if (!sourceKey) return;
    const flush = () => {
      const payload = {
        classes,
        slices: byImage[sourceKey] ?? {},
        split_by_slice: splitBySlice[sourceKey] ?? {},
        negative_slices: negativeSlices[sourceKey] ?? [],
      };
      navigator.sendBeacon(
        `${API_BASE}/api/annotations/draft?source_key=${encodeURIComponent(sourceKey)}`,
        JSON.stringify(payload)
      );
    };
    window.addEventListener('beforeunload', flush);
    return () => window.removeEventListener('beforeunload', flush);
  }, [sourceKey, byImage, splitBySlice, negativeSlices, classes]);
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
