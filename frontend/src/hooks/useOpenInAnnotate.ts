/**
 * Open any sample (Tiled or local) in the Annotate tab — load meta + restore draft.
 */
import { useCallback } from 'react';
import { useNavigate } from 'react-router';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { getClassPalette } from '@/lib/classColors';
import { useAnnotationStore } from '@/stores/annotationStore';
import { loadDraft, type DraftLoadResult } from '@/hooks/useDraftSync';
import { buildSourceKey } from '@/lib/sourceKey';
import type { Shape } from '@/stores/annotationStore';

/** Build annotation classes from a dataset's ingest keyword tags (one per tag). */
function classesFromKeywords(keywords: string[]): AnnotationClass[] {
  const palette = getClassPalette();
  const seen = new Set<string>();
  const classes: AnnotationClass[] = [];
  for (const raw of keywords) {
    const label = raw.trim();
    if (!label) continue;
    const key = label.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    classes.push({
      classId: classes.length + 1,
      label,
      color: palette[classes.length % palette.length],
      isVisible: true,
    });
  }
  return classes;
}

/** Merge a loaded draft into the stores for the given sourceKey.
 *  Returns true if the draft supplied a class list (so callers can skip seeding
 *  pre-created classes). Types are derived from each store's STATE (`getState`)
 *  rather than the hook's overloaded return type, which TS resolves to `unknown`. */
async function applyDraft(
  sourceKey: string,
  setClasses: ReturnType<typeof useClassStore.getState>['setClasses'],
  mergeSourceDraft: ReturnType<typeof useAnnotationStore.getState>['mergeSourceDraft'],
): Promise<boolean> {
  const result: DraftLoadResult = await loadDraft(sourceKey);
  if (result.status === 'error') throw result.error;
  if (result.status === 'not-found') {
    mergeSourceDraft(sourceKey, {}, {}, []);
    return false;
  }
  const payload = result.draft.payload as Record<string, unknown> | undefined;
  if (!payload) {
    mergeSourceDraft(sourceKey, {}, {}, []);
    return false;
  }
  let hadClasses = false;
  if (Array.isArray(payload.classes) && payload.classes.length > 0) {
    setClasses(payload.classes as Parameters<typeof setClasses>[0]);
    hadClasses = true;
  }
  const slices = (payload.slices ?? {}) as Record<string, Shape[]>;
  const splitMap = (payload.split_by_slice ?? {}) as Record<string, string>;
  const negSlices = (payload.negative_slices ?? []) as string[];
  mergeSourceDraft(sourceKey, slices, splitMap, negSlices);
  return hadClasses;
}

/**
 * Returns openers that load a sample's metadata, seed the dataset/class stores,
 * restore its saved draft, and navigate to the Annotate tab.
 */
export function useOpenInAnnotate() {
  const navigate = useNavigate();
  const { setDataset, setSlice } = useDatasetStore();
  const { setClasses } = useClassStore();
  const { mergeSourceDraft } = useAnnotationStore();

  /** Open a Tiled array: fetch meta, set the dataset, apply its draft, then navigate.
   *  Pass `initialSlice` to jump to a slice of a volume (keeps one sourceKey for
   *  the whole stack, so per-slice annotations stay unified). Pass `destination`
   *  to land on the Train tab instead of Annotate (e.g. "Open in Train" for an
   *  already-annotated sample). */
  const openTiledArray = useCallback(
    async (tiledPath: string, serverUri: string, initialSlice = 0, destination: '/annotate' | '/train' = '/annotate') => {
      const params = new URLSearchParams({ source: tiledPath, kind: 'tiled' });
      if (serverUri) params.set('server_uri', serverUri);

      const res = await fetch(`${API_BASE}/api/image/meta?${params}`);
      if (!res.ok) throw new Error(await res.text());

      const meta = await res.json();
      setDataset('tiled', tiledPath, serverUri || null, {
        nSlices: meta.n_slices,
        height: meta.height,
        width: meta.width,
        dtype: meta.dtype,
        isRgb: meta.is_rgb,
        valueRange: meta.value_range,
      });
      // setDataset resets to slice 0; jump to the requested slice (clamped).
      if (initialSlice > 0) setSlice(Math.min(initialSlice, Math.max(0, meta.n_slices - 1)));

      const sourceKey = buildSourceKey('tiled', tiledPath, serverUri);
      const hadClasses = await applyDraft(sourceKey, setClasses, mergeSourceDraft);
      // Classes are per-sample. When the draft carried none, always reset to THIS
      // sample's own set — keyword-seeded if available, else empty — so a previously
      // opened sample's classes never linger (they would otherwise autosave into this
      // sample's draft and cross-pollinate).
      if (!hadClasses) {
        setClasses(Array.isArray(meta.keywords) ? classesFromKeywords(meta.keywords) : []);
      }
      navigate(destination);
    },
    [navigate, setDataset, setSlice, setClasses, mergeSourceDraft],
  );

  /** Open a local file by relative path: fetch meta, set the dataset, apply its draft, then navigate.
   *  Pass `destination` to land on the Train tab instead of Annotate. */
  const openLocalFile = useCallback(
    async (relPath: string, destination: '/annotate' | '/train' = '/annotate') => {
      const params = new URLSearchParams({ source: relPath, kind: 'local' });

      const res = await fetch(`${API_BASE}/api/image/meta?${params}`);
      if (!res.ok) throw new Error(await res.text());

      const meta = await res.json();
      setDataset('local', relPath, null, {
        nSlices: meta.n_slices,
        height: meta.height,
        width: meta.width,
        dtype: meta.dtype,
        isRgb: meta.is_rgb,
        valueRange: meta.value_range,
      });

      const sourceKey = buildSourceKey('local', relPath);
      const hadClasses = await applyDraft(sourceKey, setClasses, mergeSourceDraft);
      // Reset to this sample's own classes (empty if none) so a previously opened
      // sample's classes never linger. See openTiledArray for the full rationale.
      if (!hadClasses) {
        setClasses(Array.isArray(meta.keywords) ? classesFromKeywords(meta.keywords) : []);
      }
      navigate(destination);
    },
    [navigate, setDataset, setClasses, mergeSourceDraft],
  );

  return { openTiledArray, openLocalFile };
}
