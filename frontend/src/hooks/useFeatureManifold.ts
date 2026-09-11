/**
 * useFeatureManifold — greedy variance-box suggestions + residual heatmap.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE } from '@/config';
import type { ManifoldPoint } from '@/lib/featureManifold';
import type { Shape } from '@/stores/annotationStore';

export interface ManifoldParams {
  k: number;
  /** Full square box side length in image pixels. */
  boxSize: number;
}

export const DEFAULT_MANIFOLD_PARAMS: ManifoldParams = {
  k: 24,
  boxSize: 64,
};

export interface UseFeatureManifoldArgs {
  featureJobId: string | null;
  onFeatureJobExpired?: () => void;
}

async function readErrorDetail(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === 'string') return parsed.detail;
  } catch {
    /* plain */
  }
  return text || `Request failed: ${res.status}`;
}

export function useFeatureManifold({
  featureJobId,
  onFeatureJobExpired,
}: UseFeatureManifoldArgs) {
  const [params, setParams] = useState<ManifoldParams>(DEFAULT_MANIFOLD_PARAMS);
  const [points, setPoints] = useState<ManifoldPoint[]>([]);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showMarkers, setShowMarkers] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.45);
  const [sampling, setSampling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [placementMask, setPlacementMask] = useState<Shape[] | null>(null);
  const [meta, setMeta] = useState<{
    k: number;
    nPicked: number;
    nSubsample: number;
    explainedVariance: number;
    radius: number;
    boxSize: number;
    hasMask?: boolean;
    maskPixels?: number;
  } | null>(null);
  const heatmapUrlRef = useRef<string | null>(null);
  const samplingRef = useRef(false);
  const hasSampleRef = useRef(false);
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const placementMaskRef = useRef(placementMask);
  placementMaskRef.current = placementMask;
  const onExpiredRef = useRef(onFeatureJobExpired);
  onExpiredRef.current = onFeatureJobExpired;

  const revoke = useCallback(() => {
    if (heatmapUrlRef.current) {
      URL.revokeObjectURL(heatmapUrlRef.current);
      heatmapUrlRef.current = null;
    }
    setHeatmapUrl(null);
    setPoints([]);
    setMeta(null);
    hasSampleRef.current = false;
  }, []);

  useEffect(() => {
    revoke();
    setError(null);
    setPlacementMask(null);
  }, [featureJobId, revoke]);

  useEffect(
    () => () => {
      if (heatmapUrlRef.current) URL.revokeObjectURL(heatmapUrlRef.current);
    },
    [],
  );

  const sample = useCallback(async () => {
    if (!featureJobId || samplingRef.current) return;
    samplingRef.current = true;
    setSampling(true);
    setError(null);
    const { k, boxSize } = paramsRef.current;
    const maskShapes = placementMaskRef.current;
    try {
      const res = await fetch(`${API_BASE}/api/ipred/manifold/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feature_id: featureJobId,
          k,
          box_size: boxSize,
          ...(maskShapes && maskShapes.length > 0 ? { shapes: maskShapes } : {}),
        }),
      });
      if (!res.ok) {
        const detail = await readErrorDetail(res);
        if (/not found|unknown feature/i.test(detail)) {
          onExpiredRef.current?.();
          throw new Error('Feature bank missing. Click Compute again, then Suggest labels.');
        }
        throw new Error(detail);
      }
      const body = (await res.json()) as {
        sample_id: string;
        points: ManifoldPoint[];
        k: number;
        n_picked?: number;
        n_subsample: number;
        explained_variance: number;
        radius?: number;
        box_size?: number;
        has_mask?: boolean;
        mask_pixels?: number;
      };
      const heatRes = await fetch(
        `${API_BASE}/api/ipred/manifold/${body.sample_id}/heatmap.png`,
      );
      if (!heatRes.ok) throw new Error('Failed to fetch manifold heatmap');
      const blob = await heatRes.blob();
      const url = URL.createObjectURL(blob);
      if (heatmapUrlRef.current) URL.revokeObjectURL(heatmapUrlRef.current);
      heatmapUrlRef.current = url;
      setHeatmapUrl(url);
      setPoints(body.points ?? []);
      hasSampleRef.current = (body.points?.length ?? 0) > 0 || !!url;
      setMeta({
        k: body.k,
        nPicked: body.n_picked ?? (body.points?.length ?? 0),
        nSubsample: body.n_subsample,
        explainedVariance: body.explained_variance,
        radius: body.radius ?? body.points?.[0]?.radius ?? 0,
        boxSize: body.box_size ?? boxSize,
        hasMask: body.has_mask,
        maskPixels: body.mask_pixels,
      });
    } catch (e) {
      revoke();
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      samplingRef.current = false;
      setSampling(false);
    }
  }, [featureJobId, revoke]);

  const sampleRef = useRef(sample);
  sampleRef.current = sample;

  // Re-place boxes when K / box size / mask change after an existing suggestion.
  useEffect(() => {
    if (!featureJobId || !hasSampleRef.current) return;
    const t = window.setTimeout(() => {
      void sampleRef.current();
    }, 350);
    return () => window.clearTimeout(t);
  }, [featureJobId, params.k, params.boxSize, placementMask]);

  const setPlacementMaskFromShapes = useCallback((shapes: Shape[]) => {
    setPlacementMask(shapes.length ? shapes.map((s) => ({ ...s })) : null);
  }, []);

  const clearPlacementMask = useCallback(() => {
    setPlacementMask(null);
  }, []);

  const dismiss = useCallback(() => {
    revoke();
    setError(null);
  }, [revoke]);

  return {
    params,
    setParams,
    points,
    heatmapUrl,
    showHeatmap,
    setShowHeatmap,
    showMarkers,
    setShowMarkers,
    heatmapOpacity,
    setHeatmapOpacity,
    sampling,
    error,
    meta,
    sample,
    dismiss,
    placementMask,
    setPlacementMaskFromShapes,
    clearPlacementMask,
    hasSample: points.length > 0 || !!heatmapUrl,
  };
}
