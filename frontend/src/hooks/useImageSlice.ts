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

/** Build the /api/image/slice URL encoding source, slice index, and render options. */
export function buildSliceUrl(
  source: string,
  kind: string,
  sliceIndex: number,
  renderOpts: RenderOpts,
  serverUri: string | null
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
  serverUri: string | null
) {
  const enabled = Boolean(source && kind);
  const url = enabled
    ? buildSliceUrl(source!, kind!, sliceIndex, renderOpts, serverUri)
    : null;

  const query = useQuery({
    queryKey: ['imageSlice', source, kind, sliceIndex, renderOpts, serverUri],
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
