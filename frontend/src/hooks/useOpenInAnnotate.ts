/**
 * Open any sample (Tiled or local) in the Annotate tab — load meta + restore draft.
 */
import { useCallback } from 'react';
import { useNavigate } from 'react-router';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { loadDraft } from '@/hooks/useDraftSync';
import { buildSourceKey } from '@/lib/sourceKey';
import type { Shape } from '@/stores/annotationStore';

/** Merge a loaded draft into the stores for the given sourceKey.
 *  Types are derived from each store's STATE (`getState`) rather than the
 *  hook's overloaded return type, which TS resolves to `unknown`. */
async function applyDraft(
  sourceKey: string,
  setClasses: ReturnType<typeof useClassStore.getState>['setClasses'],
  mergeSourceDraft: ReturnType<typeof useAnnotationStore.getState>['mergeSourceDraft'],
) {
  const draft = await loadDraft(sourceKey);
  if (!draft?.payload) return;
  const payload = draft.payload as Record<string, unknown>;
  if (Array.isArray(payload.classes)) {
    setClasses(payload.classes as Parameters<typeof setClasses>[0]);
  }
  const slices = (payload.slices ?? {}) as Record<string, Shape[]>;
  const splitMap = (payload.split_by_slice ?? {}) as Record<string, string>;
  const negSlices = (payload.negative_slices ?? []) as string[];
  mergeSourceDraft(sourceKey, slices, splitMap, negSlices);
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
   *  the whole stack, so per-slice annotations stay unified). */
  const openTiledArray = useCallback(
    async (tiledPath: string, serverUri: string, initialSlice = 0) => {
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
      await applyDraft(sourceKey, setClasses, mergeSourceDraft);
      navigate('/annotate');
    },
    [navigate, setDataset, setSlice, setClasses, mergeSourceDraft],
  );

  /** Open a local file by relative path: fetch meta, set the dataset, apply its draft, then navigate. */
  const openLocalFile = useCallback(
    async (relPath: string) => {
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
      await applyDraft(sourceKey, setClasses, mergeSourceDraft);
      navigate('/annotate');
    },
    [navigate, setDataset, setClasses, mergeSourceDraft],
  );

  return { openTiledArray, openLocalFile };
}
