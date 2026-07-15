/**
 * Open any sample (Tiled or local) — load meta + restore draft + session label prefs.
 */
import { useCallback } from 'react';
import { useNavigate } from 'react-router';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useConnectionStore } from '@/stores/connectionStore';
import {
  deserializeMaskSet,
  useMaskSetStore,
  type MaskSetSerialized,
} from '@/stores/maskSetStore';
import { loadDraft } from '@/hooks/useDraftSync';
import { openIpredSession } from '@/lib/ipredApi';
import { buildSourceKey } from '@/lib/sourceKey';
import type { Shape } from '@/stores/annotationStore';

/** Best-effort: bind opened sample to an ipred session (project). */
async function bindIpredSession(payload: {
  kind: string;
  source: string;
  server_uri?: string | null;
  root?: string | null;
}): Promise<void> {
  try {
    const session = await openIpredSession(payload);
    useConnectionStore.getState().setIpredSession({
      sessionId: session.session_id,
      projectId: session.project_id,
    });
  } catch {
    // ipred may be down in some dev setups; Preprocess can retry later.
    useConnectionStore.getState().setIpredSession({ sessionId: null, projectId: null });
  }
}

/** Merge a loaded draft into the stores. Returns true if draft supplied classes. */
async function applyDraft(
  sourceKey: string,
  setClasses: ReturnType<typeof useClassStore.getState>['setClasses'],
  mergeSourceDraft: ReturnType<typeof useAnnotationStore.getState>['mergeSourceDraft'],
): Promise<boolean> {
  const draft = await loadDraft(sourceKey);
  if (!draft?.payload) return false;
  const payload = draft.payload as Record<string, unknown>;
  let hadClasses = false;
  if (Array.isArray(payload.classes) && payload.classes.length > 0) {
    setClasses(payload.classes as Parameters<typeof setClasses>[0]);
    hadClasses = true;
  }
  const slices = (payload.slices ?? {}) as Record<string, Shape[]>;
  const splitMap = (payload.split_by_slice ?? {}) as Record<string, string>;
  const negSlices = (payload.negative_slices ?? []) as string[];
  mergeSourceDraft(sourceKey, slices, splitMap, negSlices);
  const maskSets = Array.isArray(payload.mask_sets)
    ? (payload.mask_sets as MaskSetSerialized[]).map(deserializeMaskSet)
    : [];
  const others = useMaskSetStore.getState().sets.filter((m) => m.sourceKey !== sourceKey);
  useMaskSetStore.getState().replaceAll([...others, ...maskSets]);
  return hadClasses;
}

//////////////////////////////////////////////////////////////////////////////
// # REMOVE THIS AND USE YOUR OWN STUFF
// Seeds classes from the Connect/Browse label-set picker when a sample has no
// draft classes. Replace with your own label bootstrap.
//////////////////////////////////////////////////////////////////////////////
/** Seed preferred session label set when the sample has no draft classes. */
function applyPreferredClassesIfNeeded(
  hadDraftClasses: boolean,
  setClasses: ReturnType<typeof useClassStore.getState>['setClasses'],
) {
  if (hadDraftClasses) return;
  const preferred = useConnectionStore.getState().preferredClasses;
  if (preferred && preferred.length > 0) {
    setClasses(preferred.map((c) => ({ ...c })));
  }
}

/** Returns openers that load meta, restore draft, apply session prefs, go to Preprocess. */
export function useOpenInAnnotate() {
  const navigate = useNavigate();
  const { setDataset, setSlice } = useDatasetStore();
  const { setClasses } = useClassStore();
  const { mergeSourceDraft } = useAnnotationStore();

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
      if (initialSlice > 0) setSlice(Math.min(initialSlice, Math.max(0, meta.n_slices - 1)));

      const sourceKey = buildSourceKey('tiled', tiledPath, serverUri);
      const hadClasses = await applyDraft(sourceKey, setClasses, mergeSourceDraft);
      applyPreferredClassesIfNeeded(hadClasses, setClasses);
      await bindIpredSession({
        kind: 'tiled',
        source: tiledPath,
        server_uri: serverUri || null,
      });
      navigate('/preprocess');
    },
    [navigate, setDataset, setSlice, setClasses, mergeSourceDraft],
  );

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
      const hadClasses = await applyDraft(sourceKey, setClasses, mergeSourceDraft);
      applyPreferredClassesIfNeeded(hadClasses, setClasses);
      const localRoot = useConnectionStore.getState().localRoot;
      await bindIpredSession({
        kind: 'local',
        source: relPath,
        root: localRoot,
      });
      navigate('/preprocess');
    },
    [navigate, setDataset, setClasses, mergeSourceDraft],
  );

  return { openTiledArray, openLocalFile };
}
