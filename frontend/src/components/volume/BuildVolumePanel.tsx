/**
 * BuildVolumePanel — turn a per-slice dataset into a renderable 3-D volume.
 *
 * A dataset ingested as individual 2-D slices cannot be streamed as a volume;
 * the 3-D view needs a multiscale pyramid. Building one is a one-click job here
 * rather than a form, because the slices are already in Tiled — asking for a
 * source directory would mean asking for data the app already holds.
 *
 * Only the downsampled levels are written. Full resolution stays in the existing
 * per-slice nodes: the renderer only ever uploads a level that fits a GPU 3-D
 * texture, so a full-resolution copy would be cost with no benefit.
 */
import { useCallback, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Cube, Warning, CircleNotch } from '@phosphor-icons/react';
import { API_BASE } from '@/config';

interface PyramidLevel {
  path: string;
  factor: [number, number, number];
  shape: [number, number, number];
}

interface BuildInfo {
  full_shape: [number, number, number];
  dtype: string;
  pyramid_plan: PyramidLevel[];
  slices_to_read: number;
  already_small: boolean;
}

interface JobState {
  state: 'pending' | 'running' | 'done' | 'error';
  phase: string;
  done: number;
  total: number;
  error: string | null;
}

interface BuildVolumePanelProps {
  source: string;
  serverUri: string | null;
  /** Why no volume exists, from `GET /api/volume/resolve`. */
  message: string;
}

/** Pull the FastAPI `detail` message out of an error response. */
async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') return body.detail;
    return JSON.stringify(body);
  } catch {
    return `Request failed (${res.status})`;
  }
}

export default function BuildVolumePanel({ source, serverUri, message }: BuildVolumePanelProps) {
  const queryClient = useQueryClient();
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: info, isLoading, error: inspectError } = useQuery<BuildInfo>({
    queryKey: ['volume-build-inspect', serverUri, source],
    queryFn: async () => {
      const params = new URLSearchParams({ source, kind: 'tiled' });
      if (serverUri) params.set('server_uri', serverUri);
      const res = await fetch(`${API_BASE}/api/volume/build/inspect?${params}`);
      if (!res.ok) throw new Error(await readError(res));
      return res.json();
    },
    retry: false,
  });

  const build = useCallback(async () => {
    setError(null);
    setJob({ state: 'pending', phase: 'Starting', done: 0, total: 1, error: null });
    try {
      const res = await fetch(`${API_BASE}/api/volume/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, kind: 'tiled', server_uri: serverUri }),
      });
      if (!res.ok) throw new Error(await readError(res));
      const { job_id: jobId } = await res.json();

      // Poll rather than stream: this reuses the same export-job registry (and
      // the same status route) every other long-running task here already uses.
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const statusRes = await fetch(`${API_BASE}/api/export/status/${jobId}`);
        if (!statusRes.ok) throw new Error(await readError(statusRes));
        const status: JobState = await statusRes.json();
        setJob(status);
        if (status.state === 'error') throw new Error(status.error || 'Build failed');
        if (status.state === 'done') break;
      }
      // The volume now exists — re-resolve so the viewer picks it up.
      await queryClient.invalidateQueries({ queryKey: ['volume-node'] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setJob(null);
    }
  }, [source, serverUri, queryClient]);

  const running = job !== null && job.state !== 'done';
  const percent = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="w-full max-w-lg text-center text-sky-200">
        <Cube size={40} className="mx-auto mb-3 opacity-60" />
        <p className="mb-1 font-medium">No 3D volume for this dataset yet</p>
        <p className="mb-5 text-sm opacity-80">{message}</p>

        {isLoading && <p className="text-sm opacity-70">Checking this dataset…</p>}

        {inspectError && (
          <p className="flex items-center justify-center gap-2 text-sm text-amber-300">
            <Warning size={16} />
            {inspectError instanceof Error ? inspectError.message : String(inspectError)}
          </p>
        )}

        {info && !info.already_small && (
          <>
            <div className="mb-4 rounded border border-sky-800/60 bg-sky-950/40 p-3 text-left text-xs">
              <p className="mb-1">
                <span className="opacity-70">Source</span>{' '}
                {info.full_shape.join(' × ')} <span className="opacity-70">{info.dtype}</span>
              </p>
              <p className="mb-1">
                <span className="opacity-70">Will build</span>{' '}
                {info.pyramid_plan.map((l) => l.shape.join('×')).join(', ')}
              </p>
              <p className="opacity-70">
                Reads {info.slices_to_read} slices once. Full resolution is not copied —
                it stays where it is, and the 3D view never loads a level that large.
              </p>
            </div>

            {!running && (
              <button
                type="button"
                onClick={build}
                className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500"
              >
                Build 3D volume
              </button>
            )}

            {running && (
              <div className="text-sm">
                <p className="mb-2 flex items-center justify-center gap-2">
                  <CircleNotch size={16} className="animate-spin" />
                  {job.phase} — {job.done}/{job.total}
                </p>
                <div className="h-2 w-full overflow-hidden rounded bg-sky-950">
                  <div
                    className="h-full bg-sky-500 transition-[width] duration-300"
                    style={{ width: `${percent}%` }}
                  />
                </div>
              </div>
            )}
          </>
        )}

        {error && (
          <p className="mt-4 flex items-center justify-center gap-2 text-sm text-amber-300">
            <Warning size={16} />
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
