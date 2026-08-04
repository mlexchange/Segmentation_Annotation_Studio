/**
 * gatherTrainingSources — turn a set of selected, annotated sourceKeys into
 * the /api/train/start `sources` + a validated shared `classes` list.
 *
 * Mirrors DownloadModal.tsx's multi-source export scope-building: training
 * data is drawn from `annotationStore.byImage` (every sample touched this
 * session), not fetched from the backend for samples annotated in an earlier
 * session — same scoping DownloadModal itself uses for "All annotated samples".
 */
import { parseSourceKey } from './sourceKey';
import { validateSharedTaxonomy } from './exportValidation';
import type { AnnotationClass } from '@/stores/classStore';
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

export interface GatheredTrainingData {
  sources: TrainingSourceItem[];
  classes: AnnotationClass[];
}

/**
 * Build the per-sample training payload for `selectedKeys` and resolve one
 * shared class taxonomy across them (including the currently-open sample's
 * live class list, if it's among the selection).
 *
 * @throws Error if `selectedKeys` is empty, or the selected samples don't
 *   share one class taxonomy (message is written to be shown to the user
 *   directly, matching `validateSharedTaxonomy`'s convention).
 */
export function gatherTrainingSources(
  selectedKeys: string[],
  byImage: Record<string, Record<string, Shape[]>>,
  splitBySlice: Record<string, Record<string, string>>,
  negativeSlices: Record<string, string[]>,
  classesBySource: Record<string, AnnotationClass[]>,
  currentKey: string | null,
  currentClasses: AnnotationClass[],
): GatheredTrainingData {
  if (selectedKeys.length === 0) {
    throw new Error('Select at least one annotated sample to train on.');
  }

  const sources: TrainingSourceItem[] = selectedKeys.map((sk) => {
    const { kind, path, serverUri } = parseSourceKey(sk);
    return {
      kind,
      source: path,
      server_uri: serverUri,
      slices: byImage[sk] ?? {},
      split_by_slice: splitBySlice[sk] ?? {},
      negative_slices: negativeSlices[sk] ?? [],
    };
  });

  const taxonomies = currentKey ? { ...classesBySource, [currentKey]: currentClasses } : classesBySource;
  const classes = validateSharedTaxonomy(selectedKeys, taxonomies, byImage);

  return { sources, classes };
}
