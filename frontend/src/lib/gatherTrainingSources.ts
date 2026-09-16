/**
 * gatherTrainingSources — turn a set of selected, annotated sourceKeys into
 * the /api/train/start `sources` list.
 *
 * Mirrors DownloadModal.tsx's multi-source export scope-building: training
 * data is drawn from `annotationStore.byImage` (every sample touched this
 * session), not fetched from the backend for samples annotated in an earlier
 * session — same scoping DownloadModal itself uses for "All annotated samples".
 *
 * Unlike the fork this was ported from, this app has no per-source class
 * taxonomy to validate: `classStore` is a single global, flat class list
 * shared by every sample, so there is nothing to reconcile across sources —
 * the caller passes that one list straight to `/api/train/start` alongside
 * whatever this returns.
 */
import { parseSourceKey } from './sourceKey';
import type { Shape } from '@/stores/annotationStore';

export interface TrainingCandidate {
  sourceKey: string;
  shapeCount: number;
}

/** Every sourceKey in `byImage` with at least one shape, sorted for stable display. */
export function listAnnotatedSourceKeys(
  byImage: Record<string, Record<string, Shape[]>>,
): TrainingCandidate[] {
  return Object.keys(byImage)
    .map((sourceKey) => ({
      sourceKey,
      shapeCount: Object.values(byImage[sourceKey]).reduce((n, shapes) => n + shapes.length, 0),
    }))
    .filter((c) => c.shapeCount > 0)
    .sort((a, b) => a.sourceKey.localeCompare(b.sourceKey));
}

export interface TrainingSourceItem {
  kind: 'tiled' | 'local';
  source: string;
  server_uri: string | null;
  slices: Record<string, Shape[]>;
  split_by_slice: Record<string, string>;
  negative_slices: string[];
}

/**
 * Build the per-sample training payload for `selectedKeys`.
 *
 * @throws Error if `selectedKeys` is empty.
 */
export function gatherTrainingSources(
  selectedKeys: string[],
  byImage: Record<string, Record<string, Shape[]>>,
  splitBySlice: Record<string, Record<string, string>>,
  negativeSlices: Record<string, string[]>,
): TrainingSourceItem[] {
  if (selectedKeys.length === 0) {
    throw new Error('Select at least one annotated sample to train on.');
  }

  return selectedKeys.map((sk) => {
    const { kind, source, serverUri } = parseSourceKey(sk);
    return {
      kind,
      source,
      server_uri: serverUri,
      slices: byImage[sk] ?? {},
      split_by_slice: splitBySlice[sk] ?? {},
      negative_slices: negativeSlices[sk] ?? [],
    };
  });
}
