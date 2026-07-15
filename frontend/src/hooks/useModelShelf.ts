/**
 * # REMOVE THIS AND USE YOUR OWN STUFF
 *
 * useModelShelf — TEMPORARY scaffold to persist / list / apply CatBoost models.
 * Replace with your model registry client, then delete this hook (+ ModelShelfPanel).
 */
import { useCallback, useEffect, useState } from 'react';
import { API_BASE } from '@/config';
import type { FeatureParams } from '@/hooks/useFeatureChannels';
import { loadLabelPng } from '@/lib/pixelClf';
import { useMaskSetStore } from '@/stores/maskSetStore';
import { useDatasetStore } from '@/stores/datasetStore';
import type { Shape } from '@/stores/annotationStore';

export interface ShelfModelMeta {
  id: string;
  name: string;
  class_ids: number[];
  n_train: number;
  n_cal: number;
  train_accuracy: number;
  uses_sam: boolean;
  feature_recipe: Record<string, unknown>;
  created_at: string;
  n_features?: number;
}

function recipeFromParams(p: FeatureParams): Record<string, unknown> {
  return {
    sigma_min: p.sigmaMin,
    sigma_max: p.sigmaMax,
    intensity: p.intensity,
    edges: p.edges,
    texture: p.texture,
    clahe: p.clahe,
    include_sam: p.includeSam,
  };
}

export function useModelShelf() {
  const [models, setModels] = useState<ShelfModelMeta[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const addSet = useMaskSetStore((s) => s.addSet);
  const { source, kind, serverUri, currentSlice } = useDatasetStore();

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/clf/models`);
      if (!res.ok) throw new Error(`List failed: ${res.status}`);
      const data = (await res.json()) as ShelfModelMeta[];
      setModels(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const saveModel = useCallback(
    async (modelId: string, name: string, featureParams: FeatureParams) => {
      setBusy(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/clf/models/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_id: modelId,
            name,
            feature_recipe: recipeFromParams(featureParams),
          }),
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error(t || `Save failed: ${res.status}`);
        }
        await refresh();
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const deleteModel = useCallback(
    async (shelfId: string) => {
      setBusy(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/clf/models/${shelfId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const applyToCurrent = useCallback(
    async (
      shelfId: string,
      {
        sourceKey,
        alpha,
        preserveShapes,
        name,
      }: {
        sourceKey: string;
        alpha: number;
        preserveShapes: Shape[];
        name?: string;
      },
    ) => {
      if (!source || !kind) {
        setError('No image open');
        return null;
      }
      setBusy(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/clf/models/${shelfId}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source,
            kind,
            slice_index: currentSlice,
            server_uri: serverUri,
            alpha,
            shapes: preserveShapes,
          }),
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error(t || `Predict failed: ${res.status}`);
        }
        const meta = (await res.json()) as {
          pred_id: string;
          class_ids: number[];
          alpha: number;
        };
        const commitRes = await fetch(
          `${API_BASE}/api/image/features/clf/predict/${meta.pred_id}/commit.png`,
        );
        if (!commitRes.ok) throw new Error('Failed to fetch commit PNG');
        const blob = await commitRes.blob();
        const url = URL.createObjectURL(blob);
        try {
          const { data, width, height } = await loadLabelPng(url);
          // Keep dense pixels for Cleanup; do not polygonize early.
          let any = false;
          for (let i = 0; i < data.length; i++) {
            if (data[i]) {
              any = true;
              break;
            }
          }
          if (!any) return null;
          const alphaPct = Math.round((meta.alpha ?? alpha) * 100);
          const id = addSet({
            name: name ?? `shelf α=${alphaPct}%`,
            sourceKey,
            slice: currentSlice,
            origin: 'shelf',
            labelMap: data,
            width,
            height,
            shapes: [],
          });
          return id;
        } finally {
          URL.revokeObjectURL(url);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return null;
      } finally {
        setBusy(false);
      }
    },
    [source, kind, serverUri, currentSlice, addSet],
  );

  return {
    models,
    busy,
    error,
    refresh,
    saveModel,
    deleteModel,
    applyToCurrent,
  };
}
