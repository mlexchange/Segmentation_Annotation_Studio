/**
 * RebuildVolumeControl — re-trigger "Build 3D volume" for a dataset that
 * already has one, so a fidelity-affecting change (e.g. raising
 * `tiff_stack_source.TARGET_DIM`) can actually reach volumes built before
 * the change. `volume_build.build_volume` already replaces any prior build
 * for the same key safely (see its own "re-running must refresh the volume"
 * comment) — this only adds the UI path to trigger that for an EXISTING
 * volume; `BuildVolumePanel` covers the "none exists yet" case.
 *
 * Deliberately no staleness detection (comparing what's registered against
 * what a fresh build would produce) — a plain always-available button is
 * simpler and correct for what is, for now, a one-time fidelity bump.
 */
import { useCallback, useState } from 'react';
import { ArrowsClockwise, CircleNotch, Warning } from '@phosphor-icons/react';
import { API_BASE } from '@/config';

interface JobState {
  state: 'pending' | 'running' | 'done' | 'error';
  phase: string;
  done: number;
  total: number;
  error: string | null;
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') return body.detail;
    return JSON.stringify(body);
  } catch {
    return `Request failed (${res.status})`;
  }
}

interface RebuildVolumeControlProps {
  source: string;
  serverUri: string | null;
  /** Called once the rebuild finishes — the caller should bump whatever key
   * forces VolumeViewer to remount, since the resolved path doesn't change
   * on a rebuild (same key, same location) the way a brand-new build does. */
  onRebuilt: () => void;
}

export default function RebuildVolumeControl({ source, serverUri, onRebuilt }: RebuildVolumeControlProps) {
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const rebuild = useCallback(async () => {
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

      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const statusRes = await fetch(`${API_BASE}/api/export/status/${jobId}`);
        if (!statusRes.ok) throw new Error(await readError(statusRes));
        const status: JobState = await statusRes.json();
        setJob(status);
        if (status.state === 'error') throw new Error(status.error || 'Rebuild failed');
        if (status.state === 'done') break;
      }
      setJob(null);
      onRebuilt();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setJob(null);
    }
  }, [source, serverUri, onRebuilt]);

  const running = job !== null;
  const percent = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="pointer-events-auto flex flex-col items-start gap-1">
      <button
        type="button"
        disabled={running}
        onClick={() => void rebuild()}
        title="Re-build this dataset's 3D pyramid — picks up any fidelity/quality changes made since it was last built"
        className="flex items-center gap-1.5 rounded-lg border border-sky-800/60 bg-sky-950/80 px-2.5 py-1.5 text-xs text-sky-200 backdrop-blur-sm hover:bg-sky-900/70 disabled:opacity-60"
      >
        {running ? <CircleNotch size={13} className="animate-spin" /> : <ArrowsClockwise size={13} />}
        {running ? `${job.phase} ${job.total > 0 ? `(${percent}%)` : ''}` : 'Rebuild volume'}
      </button>
      {error && (
        <p className="flex max-w-64 items-start gap-1 rounded bg-sky-950/80 px-2 py-1 text-[11px] text-amber-300 backdrop-blur-sm">
          <Warning size={12} className="mt-0.5 shrink-0" />
          {error}
        </p>
      )}
    </div>
  );
}
