/**
 * VolumePage — 3D view of the open dataset, rendered straight from Tiled.
 *
 * The volume is streamed by the vendored WebGPU renderer directly off Tiled's
 * `/zarr/v2` router (see `lib/zarrUrl.ts`) — there is no export step and no
 * downsampled-volume endpoint in between, so whatever is open in Annotate is
 * what renders here. Rendering controls (transfer function, crop box, slice
 * planes, lighting) come from the renderer's own HUD, docked on the right.
 *
 * Which node actually holds the volume is asked of the backend rather than
 * guessed from the path: a registered Zarr volume is one already, a TIFF stack's
 * lives in a `__volume` sidecar, and a stack nobody has built one for has none.
 * See `backend/volume_nodes.py`.
 *
 * Annotation overlay is deliberately not here yet — it needs a second label
 * texture in the renderer, which lands upstream in the viewer repo.
 */
import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Cube } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { buildZarrUrl, describeUnavailable, type ZarrUnavailable } from '@/lib/zarrUrl';
import type { ServerInfo } from '@/types/server';
import VolumeViewer, { webGpuAvailability } from '@/components/volume/VolumeViewer';
import BuildVolumePanel from '@/components/volume/BuildVolumePanel';

/** Shape of `GET /api/volume/resolve`. */
interface VolumeNode {
  path: string | null;
  mode: 'self' | 'ancestor' | 'sidecar' | 'none';
  source_dir: string | null;
  message: string;
}

/** Centered message panel — every non-rendering state uses this shape. */
function Notice({ title, detail, hint }: { title: string; detail: string; hint?: string }) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-md text-center text-sky-200">
        <Cube size={40} className="mx-auto mb-3 opacity-60" />
        <p className="mb-1 font-medium">{title}</p>
        <p className="text-sm opacity-80">{detail}</p>
        {hint && <p className="mt-3 text-xs opacity-60">{hint}</p>}
      </div>
    </div>
  );
}

export default function VolumePage() {
  const { kind, source, serverUri } = useDatasetStore();
  const [bootError, setBootError] = useState<string | null>(null);

  // A new dataset deserves a fresh attempt — otherwise one bad volume leaves the
  // page stuck on its error for every dataset opened afterwards.
  useEffect(() => setBootError(null), [source, serverUri]);

  // Fallback only: a dataset opened against the default local Tiled carries a
  // null serverUri, and the resolved address (whatever port start_all.sh
  // actually bound) lives here. Never assume 8010.
  const { data: servers = [] } = useQuery<ServerInfo[]>({
    queryKey: ['servers'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/config/servers`);
      if (!res.ok) throw new Error('Failed to load servers');
      return res.json();
    },
    enabled: kind === 'tiled' && !serverUri,
  });

  const resolvedUri = serverUri ?? servers[0]?.uri ?? null;

  const { data: node, isLoading: resolving } = useQuery<VolumeNode>({
    queryKey: ['volume-node', resolvedUri, source],
    queryFn: async () => {
      const params = new URLSearchParams({ source: source! });
      if (resolvedUri) params.set('server_uri', resolvedUri);
      const res = await fetch(`${API_BASE}/api/volume/resolve?${params}`);
      if (!res.ok) throw new Error('Failed to resolve the volume node');
      return res.json();
    },
    enabled: kind === 'tiled' && !!source,
  });

  const availability = webGpuAvailability();
  if (!availability.ok) {
    return <Notice title="3D view unavailable" detail={availability.reason} />;
  }

  // Resolution only applies to Tiled sources; everything else already has a
  // reason of its own and should not sit on a spinner.
  let reason: ZarrUnavailable | null = null;
  if (kind === 'tiled' && source && (resolving || !node)) reason = 'resolving';
  else if (node && node.mode === 'none') reason = 'no-volume';

  const { url, reason: urlReason } = buildZarrUrl(kind, node?.path ?? null, resolvedUri);
  reason = reason ?? (url ? null : urlReason);

  // A stack with no volume is not an error state — it is one the user can
  // resolve in place, so offer the build rather than describing an API call.
  if (reason === 'no-volume' && source) {
    return <BuildVolumePanel source={source} serverUri={resolvedUri} message={node!.message} />;
  }

  if (!url || reason) {
    return <Notice title="Nothing to render" detail={describeUnavailable(reason ?? 'no-source')} />;
  }

  if (bootError) {
    return (
      <Notice
        title="This dataset could not be opened as a volume"
        detail={bootError}
        hint={`Tried ${node?.path} (${node?.mode}).`}
      />
    );
  }

  return (
    <div className="flex h-full w-full flex-col">
      <VolumeViewer
        // Remount on source change so the renderer tears its GPU device down
        // and rebuilds, rather than trying to swap a volume in place.
        key={url}
        zarrUrl={url}
        onError={(e) => setBootError(e instanceof Error ? e.message : String(e))}
      />
    </div>
  );
}
