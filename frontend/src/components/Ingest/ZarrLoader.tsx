/**
 * ZarrLoader — load an on-disk Zarr volume by path, without uploading anything.
 *
 * The dropzone next to this streams files through the browser; a tomography
 * volume is far too large for that (the reference data is 7-56 GB). Tiled can
 * read a Zarr store in place, so loading one is really just registering its path:
 * nothing is copied, and slices are read lazily as you annotate.
 *
 * Flow: paste a server-side path -> Inspect (shows the pyramid) -> pick a level
 * -> Load. Inspect is separate from Load so you can see what was found before
 * anything is written to the catalog.
 */
import { useCallback, useState } from 'react';
import { Stack, Warning, CheckCircle, MagnifyingGlass } from '@phosphor-icons/react';
import { API_BASE } from '@/config';

/** One resolution level of a multiscale volume. */
interface ZarrLevel {
  path: string;
  shape: [number, number, number];
  dtype: string;
  n_slices: number;
  height: number;
  width: number;
  /** Full-res voxels spanned per voxel here, as [z, y, x]. */
  downsample: [number, number, number];
}

interface ZarrInfo {
  name: string;
  path: string;
  levels: ZarrLevel[];
  full_shape: [number, number, number];
  dtype: string;
  voxel_size: number[] | null;
  voxel_unit: string | null;
}

/** What already occupies the destination key, when there is a collision. */
interface ExistingInfo {
  child_count: number;
  external: boolean;
  sample_name: string;
  n_images: number | null;
}

interface ZarrLoaderProps {
  serverUri: string;
  onBrowse?: (containerPath: string, sampleCount: number) => void;
  onAnnotate?: (tiledPath: string) => void;
}

/** Human-readable byte-ish size of a level, for a sense of scale. */
function describeLevel(level: ZarrLevel): string {
  const [z, y, x] = level.shape;
  return `${z} × ${y} × ${x}`;
}

export default function ZarrLoader({ serverUri, onBrowse, onAnnotate }: ZarrLoaderProps) {
  const [path, setPath] = useState('');
  const [containerPath, setContainerPath] = useState('browse');
  const [description, setDescription] = useState('');
  const [info, setInfo] = useState<ZarrInfo | null>(null);
  const [levelIdx, setLevelIdx] = useState(0);
  const [busy, setBusy] = useState<'inspect' | 'register' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [existing, setExisting] = useState<ExistingInfo | null>(null);
  const [loaded, setLoaded] = useState<{ tiledPath: string; nSlices: number } | null>(null);

  /** Pull the FastAPI `detail` message out of an error response. */
  const readError = async (res: Response): Promise<string> => {
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') return body.detail;
      return JSON.stringify(body);
    } catch {
      return `Request failed (${res.status})`;
    }
  };

  const inspect = useCallback(async () => {
    setBusy('inspect');
    setError(null);
    setInfo(null);
    setExisting(null);
    setLoaded(null);
    try {
      const res = await fetch(`${API_BASE}/api/zarr/inspect?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error(await readError(res));
      const data: ZarrInfo = await res.json();
      setInfo(data);
      setLevelIdx(0);

      // Warn about a name collision before the user commits to loading.
      const pf = await fetch(`${API_BASE}/api/zarr/preflight`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, container_path: containerPath, server_uri: serverUri }),
      });
      if (pf.ok) {
        const conflict = await pf.json();
        setExisting(conflict.exists ? conflict.existing : null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }, [path, containerPath, serverUri]);

  const register = useCallback(
    async (onConflict: 'fail' | 'replace') => {
      if (!info) return;
      setBusy('register');
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/zarr/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: info.path,
            container_path: containerPath,
            description,
            on_conflict: onConflict,
            server_uri: serverUri,
          }),
        });
        if (!res.ok) throw new Error(await readError(res));
        const data = await res.json();
        const level = info.levels[levelIdx];
        setLoaded({
          tiledPath: `${data.tiled_path}/${level.path}`,
          nSlices: info.levels[0].n_slices,
        });
        setExisting(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(null);
      }
    },
    [info, containerPath, description, serverUri, levelIdx],
  );

  const level = info?.levels[levelIdx];
  const zFactor = level ? level.downsample[0] : 1;
  const coarse = levelIdx > 0;

  return (
    <div className="space-y-3">
      <p className="text-xs text-sky-300/90">
        Point at a <code className="text-sky-200">.zarr</code> directory on the server. Nothing is
        copied — Tiled reads it in place, so even a 50&nbsp;GB volume loads in seconds.
      </p>

      <div className="flex gap-2">
        <input
          type="text"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && path.trim()) inspect(); }}
          placeholder="/absolute/path/to/volume.zarr"
          className="flex-1 min-w-0 rounded-md border border-white/15 bg-white/5 px-2 py-1.5 text-sm text-white placeholder:text-white/30"
        />
        <button
          type="button"
          onClick={inspect}
          disabled={!path.trim() || busy !== null}
          className="flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
        >
          <MagnifyingGlass size={15} />
          {busy === 'inspect' ? 'Inspecting…' : 'Inspect'}
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-2 text-xs text-amber-200">
          <Warning size={15} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {info && (
        <div className="space-y-3 rounded-md border border-white/10 bg-black/20 p-3">
          <div className="text-sm text-white">
            <span className="font-medium">{info.name}</span>
            <span className="ml-2 text-xs text-sky-300/80">
              {info.full_shape[0]} × {info.full_shape[1]} × {info.full_shape[2]} · {info.dtype}
              {info.voxel_size ? ` · ${info.voxel_size[0]} ${info.voxel_unit ?? ''}/voxel` : ''}
            </span>
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-sky-100">Resolution level</label>
            <div className="flex flex-wrap gap-1">
              {info.levels.map((lv, i) => (
                <button
                  key={lv.path}
                  type="button"
                  onClick={() => setLevelIdx(i)}
                  className={[
                    'rounded-md border px-2 py-1 text-[11px] transition-colors',
                    i === levelIdx
                      ? 'border-sky-400 bg-sky-600 text-white'
                      : 'border-white/15 bg-white/5 text-sky-100 hover:bg-white/10',
                  ].join(' ')}
                  title={describeLevel(lv)}
                >
                  {i === 0 ? 'Full res' : `1/${Math.round(lv.downsample[1])}`}
                  <span className="ml-1 opacity-60">{lv.width}²</span>
                </button>
              ))}
            </div>
            {coarse && level && (
              // Both consequences of annotating a coarse level, stated plainly:
              // strokes are stored as if full-res, and z is subsampled too.
              <p className="text-[11px] leading-snug text-amber-200/90">
                Annotations are stored in full-resolution coordinates, so strokes drawn here carry
                less precision than they appear to. This level also has {level.n_slices} slices, so
                it addresses roughly every {zFactor.toFixed(zFactor % 1 ? 1 : 0)}
                <sup>th</sup> full-resolution slice.
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="block text-xs font-medium text-sky-100">Destination</label>
              <input
                type="text"
                value={containerPath}
                onChange={(e) => setContainerPath(e.target.value)}
                className="w-full rounded-md border border-white/15 bg-white/5 px-2 py-1 text-xs text-white"
              />
            </div>
            <div className="space-y-1">
              <label className="block text-xs font-medium text-sky-100">Keywords (optional)</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="tomography, sand"
                className="w-full rounded-md border border-white/15 bg-white/5 px-2 py-1 text-xs text-white placeholder:text-white/30"
              />
            </div>
          </div>

          {existing && (
            <div className="space-y-2 rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-2 text-xs text-amber-200">
              <div className="flex items-start gap-2">
                <Warning size={15} className="mt-0.5 flex-shrink-0" />
                <span>
                  <b>{info.name.replace(/\.zarr$/, '')}</b> already exists in{' '}
                  <b>{containerPath}</b>
                  {existing.external
                    ? ' as a previously loaded Zarr. Replacing it only updates the catalog entry.'
                    : ` and holds ${existing.child_count} uploaded images. That is different data —
                       replacing it would delete those files, so choose another destination.`}
                </span>
              </div>
              {existing.external && (
                <button
                  type="button"
                  onClick={() => register('replace')}
                  disabled={busy !== null}
                  className="rounded-md border border-amber-300/40 px-2 py-1 font-medium hover:bg-amber-400/10 disabled:opacity-40"
                >
                  Replace it
                </button>
              )}
            </div>
          )}

          {!loaded && (
            <button
              type="button"
              onClick={() => register('fail')}
              disabled={busy !== null || (existing !== null && !existing.external)}
              className="flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
            >
              <Stack size={15} />
              {busy === 'register' ? 'Loading…' : 'Load volume'}
            </button>
          )}

          {loaded && (
            <div className="space-y-2">
              <div className="flex items-start gap-2 text-xs text-emerald-300">
                <CheckCircle size={15} className="mt-0.5 flex-shrink-0" />
                <span>Loaded {loaded.nSlices} slices — no data was copied.</span>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => onAnnotate?.(loaded.tiledPath)}
                  className="rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
                >
                  Annotate
                </button>
                <button
                  type="button"
                  onClick={() => onBrowse?.(containerPath, 1)}
                  className="rounded-md border border-white/15 px-3 py-1.5 text-xs font-medium text-sky-100 hover:bg-white/10"
                >
                  Browse
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
