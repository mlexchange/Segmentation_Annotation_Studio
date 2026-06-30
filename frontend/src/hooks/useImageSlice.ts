/**
 * useImageSlice — TanStack Query hook to fetch a PNG slice from the backend.
 */
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import { RenderOpts } from '@/stores/datasetStore';

export interface SliceResult {
  url: string;
}

/** Build the /api/image/slice URL encoding source, slice index, and render options. */
function buildSliceUrl(
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

  return useQuery({
    queryKey: ['imageSlice', source, kind, sliceIndex, renderOpts, serverUri],
    queryFn: async () => {
      const res = await fetch(url!);
      if (!res.ok) throw new Error(`Slice fetch failed: ${res.status}`);
      const blob = await res.blob();
      return URL.createObjectURL(blob);
    },
    enabled,
    staleTime: 1000 * 300,
  });
}
