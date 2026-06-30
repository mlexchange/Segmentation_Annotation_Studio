/**
 * useAnnotatedSourceKeys — set of sourceKeys that have at least one annotation.
 * Combines in-session store state with persisted drafts from the backend.
 */
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import { useAnnotationStore } from '@/stores/annotationStore';

interface DraftSummary {
  source_key: string;
  has_annotations?: boolean;
}

/**
 * Returns a Set of sourceKeys that have at least one annotation, merging
 * in-session store state with backend drafts (queried, 30s stale time).
 */
export function useAnnotatedSourceKeys() {
  const byImage = useAnnotationStore((s) => s.byImage);

  const { data: drafts = [] } = useQuery<DraftSummary[]>({
    queryKey: ['annotationDrafts'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/annotations/drafts`);
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 30_000,
  });

  return useMemo(() => {
    const keys = new Set<string>();
    for (const [sk, slices] of Object.entries(byImage)) {
      if (Object.values(slices).some((shapes) => shapes.length > 0)) {
        keys.add(sk);
      }
    }
    for (const d of drafts) {
      if (d.source_key && d.has_annotations) keys.add(d.source_key);
    }
    return keys;
  }, [byImage, drafts]);
}
