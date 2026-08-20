/**
 * useImageSlice — TanStack Query hook to fetch a PNG slice from the backend.
 */
import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import { RenderOpts } from '@/stores/datasetStore';

export interface SliceResult {
  /** Exact request URL that produced this blob. */
  requestKey: string;
  url: string;
}

export interface LoadedSliceImage {
  image: HTMLImageElement | null;
  requestKey: string | null;
  error: string | null;
}

/**
 * Server-side denoising of the raw slice, applied before normalization (see
 * the backend route's docstring for why it lives there rather than in
 * `render_slice`). Deliberately a separate parameter rather than a field on
 * `RenderOpts`: that store's `setRenderOpts` has no callers, Annotate always
 * fetches with its defaults, and denoise state belongs to the Annotate page.
 *
 * `crop` > 0 returns only a centred square of that size at 1:1 — the fast
 * tuning path for the expensive filters (measured at 3232²: bilateral 7.5s
 * full-slice vs 0.71s cropped). It crops rather than downscales because
 * downscaling is itself a denoiser, so a downscaled preview cannot be used to
 * judge denoising.
 *
 * `method: 'model'` previews a trained Noise2Noise/Noise2Void run (see
 * DenoisePanel's "Learned denoiser" mode) instead of a classical filter —
 * `runId` names which saved run to use and `strength` is ignored for it.
 * // TODO: the backend doesn't serve `denoise_method=model` yet (classical
 * // filters only, see backend/denoise.py's ALL_METHODS) — confirm the param
 * // name/shape against `/api/image/slice` once denoiser inference lands.
 */
export interface DenoiseOpts {
  method: string;
  strength: number;
  crop?: number;
  /** Saved run id for `method: 'model'`; unused otherwise. */
  runId?: string;
}

/** Stable "denoising off" value — a module constant so it can be used as a
 *  default prop / memo dependency without allocating a new object per render. */
export const NO_DENOISE: DenoiseOpts = { method: 'none', strength: 0.5 };

/** Build the /api/image/slice URL encoding source, slice index, and render options. */
export function buildSliceUrl(
  source: string,
  kind: string,
  sliceIndex: number,
  renderOpts: RenderOpts,
  serverUri: string | null,
  denoise: DenoiseOpts = NO_DENOISE
): string {
  const params = new URLSearchParams({
    source,
    kind,
    slice_index: String(sliceIndex),
    norm: renderOpts.norm,
    scale: renderOpts.scale,
    vmin_pct: String(renderOpts.vminPct),
    vmax_pct: String(renderOpts.vmaxPct),
    cmap: renderOpts.cmap,
  });
  if (serverUri) params.set('server_uri', serverUri);
  // Omitted entirely when off, so an un-denoised request stays byte-identical
  // to what this app sent before denoising existed (and keeps hitting the same
  // browser-cache entries).
  if (denoise.method !== 'none') {
    params.set('denoise_method', denoise.method);
    if (denoise.method === 'model') {
      // A learned denoiser has no strength knob — it's either applied or not.
      if (denoise.runId) params.set('denoise_run_id', denoise.runId);
    } else {
      params.set('denoise_strength', String(denoise.strength));
    }
    if (denoise.crop && denoise.crop > 0) params.set('denoise_crop', String(denoise.crop));
  }
  return `${API_BASE}/api/image/slice?${params.toString()}`;
}

/**
 * Fetches a PNG slice as a blob object URL (5min stale time). Disabled until
 * both source and kind are set. Use for displaying a rendered image slice.
 */
export function useImageSlice(
  source: string | null,
  kind: string | null,
  sliceIndex: number,
  renderOpts: RenderOpts,
  serverUri: string | null,
  denoise: DenoiseOpts = NO_DENOISE
) {
  const enabled = Boolean(source && kind);
  const url = enabled
    ? buildSliceUrl(source!, kind!, sliceIndex, renderOpts, serverUri, denoise)
    : null;

  const query = useQuery({
    // Denoise params are part of the key: a different filter or strength is a
    // genuinely different image, and the canvas's tool caches key off the
    // resulting requestKey, so they invalidate along with it for free.
    queryKey: [
      'imageSlice', source, kind, sliceIndex, renderOpts, serverUri,
      denoise.method, denoise.strength, denoise.crop ?? 0, denoise.runId ?? '',
    ],
    queryFn: async ({ signal }) => {
      const res = await fetch(url!, { signal });
      if (!res.ok) throw new Error(`Slice fetch failed: ${res.status}`);
      return res.blob();
    },
    enabled,
    staleTime: 1000 * 300,
  });

  const [sliceResult, setSliceResult] = useState<SliceResult | undefined>();
  useEffect(() => {
    if (!query.data || !url) {
      setSliceResult(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(query.data);
    setSliceResult({ requestKey: url, url: objectUrl });
    return () => URL.revokeObjectURL?.(objectUrl);
  }, [query.data, url]);

  return {
    ...query,
    requestKey: url,
    // Never expose the previous slice during the effect-cleanup gap after navigation.
    data: sliceResult?.requestKey === url ? sliceResult : undefined,
  };
}

/** Decode a fetched slice and retain it only while its exact request identity is current. */
export function useLoadedSliceImage(slice: SliceResult | undefined): LoadedSliceImage {
  const [loaded, setLoaded] = useState<{
    image: HTMLImageElement | null;
    requestKey: string | null;
    error: string | null;
  }>({ image: null, requestKey: null, error: null });

  useEffect(() => {
    if (!slice) {
      setLoaded({ image: null, requestKey: null, error: null });
      return;
    }
    let active = true;
    const image = new Image();
    image.onload = () => {
      if (active) setLoaded({ image, requestKey: slice.requestKey, error: null });
    };
    image.onerror = () => {
      if (active) {
        setLoaded({ image: null, requestKey: slice.requestKey, error: 'Image decode failed.' });
      }
    };
    image.src = slice.url;
    return () => {
      active = false;
      image.onload = null;
      image.onerror = null;
    };
  }, [slice]);

  if (!slice || loaded.requestKey !== slice.requestKey) {
    return { image: null, requestKey: null, error: null };
  }
  return loaded;
}
