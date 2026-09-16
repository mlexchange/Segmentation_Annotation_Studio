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
import { Stack, Warning, CheckCircle, MagnifyingGlass, Folder, FolderOpen } from '@phosphor-icons/react';
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

interface DirEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

/** Response of POST /api/scan-datasets — bulk-registers every Zarr store AND
 * folder of image slices found directly under a directory, for pointing a
 * mounted folder of already-reconstructed volumes at Tiled in one action
 * instead of one at a time (or one per data kind). */
interface ScanResult {
  scanned: number;
  registered: { name: string; key: string; tiled_path: string }[];
  skipped: string[];
  // A candidate whose natural key collides with an UNRELATED, different-kind
  // registration sharing the same stem name (e.g. a raw image folder and its
  // own already-registered Zarr reconstruction) — nothing registered, so it
  // doesn't silently masquerade as "already present". Retry with a different
  // key (see `renames`) to keep both.
  shadowed: { name: string; key: string; existing_kind: string; suggested_key: string }[];
  errors: { name: string; error: string }[];
}

/** Join an absolute root with a root-relative path (as returned by /api/local/list). */
function joinPath(root: string, rel: string): string {
  if (!rel) return root;
  return `${root.replace(/\/$/, '')}/${rel}`;
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

  // Server-side directory browser — the path field takes an absolute path on
  // the server's own filesystem (nothing is uploaded), which is easy to get
  // wrong by typing a path from your own machine when that's NOT the same
  // filesystem the server sees (e.g. inside Docker, only what's actually
  // bind-mounted is visible). Browsing what the server can really see avoids
  // that guesswork, mirroring the same root+relative-listing pattern the
  // Connect page's "Local Folder" mode already uses — including letting the
  // root itself be overridden: running via start_all.sh (no container), the
  // backend is a native process with the same filesystem access as the rest
  // of your machine, so granting a different root here searches anywhere
  // you'd expect, exactly like "Local Folder" already does. Running via
  // Docker, the SAME override mechanism naturally can't escape the
  // container's own sandbox — an unmounted host path just comes back empty,
  // which is the correct, honest reflection of what's actually mounted
  // (see LOCAL_SOURCE_DIR in the deployment docs for making a real directory
  // visible there instead of fighting this).
  const [browsing, setBrowsing] = useState(false);
  const [defaultRoot, setDefaultRoot] = useState<string | null>(null);
  const [rootInput, setRootInput] = useState('');
  const [browseRoot, setBrowseRoot] = useState<string | null>(null);
  const [browseRel, setBrowseRel] = useState('');
  const [browseEntries, setBrowseEntries] = useState<DirEntry[]>([]);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [renameInputs, setRenameInputs] = useState<Record<string, string>>({});

  const listDir = useCallback(async (root: string, rel: string) => {
    setBrowseLoading(true);
    setBrowseError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/local/list?root=${encodeURIComponent(root)}&rel=${encodeURIComponent(rel)}`,
      );
      if (!res.ok) throw new Error(await readError(res));
      const entries: DirEntry[] = await res.json();
      setBrowseEntries(entries.filter((e) => e.is_dir));
      setBrowseRoot(root);
      setBrowseRel(rel);
    } catch (err) {
      setBrowseError(err instanceof Error ? err.message : String(err));
    } finally {
      setBrowseLoading(false);
    }
  }, []);

  const openBrowser = useCallback(async () => {
    setBrowsing(true);
    let root = defaultRoot;
    if (root === null) {
      try {
        const res = await fetch(`${API_BASE}/api/local/root`);
        if (!res.ok) throw new Error(await readError(res));
        ({ root } = await res.json());
        setDefaultRoot(root);
        setRootInput(root ?? '');
      } catch (err) {
        setBrowseError(err instanceof Error ? err.message : String(err));
        return;
      }
    }
    await listDir(browseRoot ?? root ?? '', browseRoot !== null ? browseRel : '');
  }, [defaultRoot, browseRoot, browseRel, listDir]);

  const grantRoot = () => {
    const root = rootInput.trim();
    if (root) void listDir(root, '');
  };

  const chooseDir = (entry: DirEntry) => {
    if (browseRoot === null) return;
    setPath(joinPath(browseRoot, entry.path));
    setBrowsing(false);
  };

  const runScan = useCallback(
    async (renames?: Record<string, string>) => {
      if (browseRoot === null) return null;
      const res = await fetch(`${API_BASE}/api/scan-datasets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_root: joinPath(browseRoot, browseRel),
          container_path: containerPath,
          server_uri: serverUri,
          ...(renames ? { renames } : {}),
        }),
      });
      if (!res.ok) throw new Error(await readError(res));
      return (await res.json()) as ScanResult;
    },
    [browseRoot, browseRel, containerPath, serverUri],
  );

  const scanFolder = useCallback(async () => {
    setScanning(true);
    setScanError(null);
    setScanResult(null);
    try {
      setScanResult(await runScan());
    } catch (err) {
      setScanError(err instanceof Error ? err.message : String(err));
    } finally {
      setScanning(false);
    }
  }, [runScan]);

  /** Retry one shadowed candidate under an alternate key, merging the result
   * into the existing summary rather than replacing it (the rest of the
   * scan's findings are still valid and shouldn't disappear from view). */
  const retryShadowed = useCallback(
    async (name: string, altKey: string) => {
      setScanning(true);
      setScanError(null);
      try {
        const retryResult = await runScan({ [name]: altKey });
        if (!retryResult) return;
        setScanResult((prev) => {
          const base = prev ?? { scanned: 0, registered: [], skipped: [], shadowed: [], errors: [] };
          return {
            scanned: base.scanned,
            registered: [...base.registered, ...retryResult.registered],
            skipped: base.skipped,
            shadowed: base.shadowed.filter((s) => s.name !== name),
            errors: [...base.errors, ...retryResult.errors],
          };
        });
      } catch (err) {
        setScanError(err instanceof Error ? err.message : String(err));
      } finally {
        setScanning(false);
      }
    },
    [runScan],
  );

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
        // A bare-array store (no OME-NGFF multiscale group) registers as a
        // single leaf array node directly at the container key — its level
        // has no sub-path to descend into (level.path === ''), so appending
        // '/' + '' would produce a wrong, trailing-slash Tiled path.
        setLoaded({
          tiledPath: level.path ? `${data.tiled_path}/${level.path}` : data.tiled_path,
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
        <button
          type="button"
          onClick={() => (browsing ? setBrowsing(false) : openBrowser())}
          className="flex items-center gap-1.5 rounded-md border border-white/15 px-3 py-1.5 text-sm font-medium text-sky-100 hover:bg-white/10"
        >
          <FolderOpen size={15} />
          Browse…
        </button>
      </div>

      {browsing && (
        <div className="space-y-2 rounded-md border border-white/10 bg-black/20 p-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={rootInput}
              onChange={(e) => setRootInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') grantRoot(); }}
              placeholder="Root to browse from"
              className="flex-1 min-w-0 rounded-md border border-white/15 bg-white/5 px-2 py-1 font-mono text-xs text-white placeholder:text-white/30"
            />
            <button
              type="button"
              onClick={grantRoot}
              disabled={!rootInput.trim()}
              className="rounded-md border border-white/15 px-2 py-1 text-xs font-medium text-sky-100 hover:bg-white/10 disabled:opacity-40"
            >
              Go
            </button>
          </div>
          <p className="text-[11px] leading-snug text-sky-300/80">
            Only what this server can actually see is listed — running locally
            (<code className="text-sky-200">start_all.sh</code>), that's anywhere on
            your machine; in Docker, only what's bind-mounted (e.g. via{' '}
            <code className="text-sky-200">LOCAL_SOURCE_DIR</code>).
          </p>

          {browseRoot !== null && (
            <div className="flex flex-wrap items-center gap-1 text-xs text-sky-300/90">
              <button
                type="button"
                className="rounded px-1.5 py-0.5 font-mono hover:bg-white/10 text-sky-200"
                onClick={() => listDir(browseRoot, '')}
              >
                {browseRoot}
              </button>
              {browseRel.split('/').filter(Boolean).map((seg, i, arr) => {
                const target = arr.slice(0, i + 1).join('/');
                return (
                  <span key={target} className="flex items-center gap-1">
                    <span>/</span>
                    <button
                      type="button"
                      className="rounded px-1.5 py-0.5 hover:bg-white/10 text-sky-200"
                      onClick={() => listDir(browseRoot, target)}
                    >
                      {seg}
                    </button>
                  </span>
                );
              })}
            </div>
          )}

          {browseError && (
            <div className="flex items-start gap-2 rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-2 text-xs text-amber-200">
              <Warning size={15} className="mt-0.5 flex-shrink-0" />
              <span>{browseError}</span>
            </div>
          )}

          {!browseError && (
            <div className="max-h-48 overflow-y-auto rounded-md border border-white/10 bg-white/5 text-sm">
              {browseLoading && (
                <div className="px-3 py-3 text-sky-300/80 text-xs">Loading…</div>
              )}
              {!browseLoading && browseEntries.length === 0 && (
                <div className="space-y-1.5 px-3 py-3 text-xs text-amber-200">
                  <p>No sub-folders here.</p>
                  <p className="leading-snug text-amber-200/90">
                    If you expected your own data to show up: unlike the image dropzone
                    above (which uploads through the browser), a Zarr volume has to
                    already be visible to this <em>server</em> — nothing gets copied. If
                    you're running this in Docker, that means bind-mounting it in first,
                    then restarting:
                  </p>
                  <pre className="overflow-x-auto rounded bg-black/30 px-2 py-1 font-mono text-[11px] text-amber-100">
                    LOCAL_SOURCE_DIR=/path/to/your/data docker compose -f docker-compose.full.yml up -d
                  </pre>
                  <p className="leading-snug text-amber-200/90">
                    (or set <code className="text-amber-100">LOCAL_SOURCE_DIR</code> in a{' '}
                    <code className="text-amber-100">.env</code> file at the repo root —
                    see the Production deployment docs). Not using Docker? Grant a
                    different root above instead — the server can see anywhere on that
                    machine.
                  </p>
                </div>
              )}
              {!browseLoading && browseEntries.map((entry) => {
                const isZarr = entry.name.toLowerCase().endsWith('.zarr');
                return (
                  <div
                    key={entry.path}
                    className="flex items-center gap-2 border-b border-white/10 px-3 py-2 last:border-b-0 hover:bg-white/10"
                  >
                    <button
                      type="button"
                      className="flex flex-1 items-center gap-2 text-left text-sky-100 min-w-0"
                      onClick={() => (isZarr || browseRoot === null ? chooseDir(entry) : listDir(browseRoot, entry.path))}
                    >
                      {isZarr ? (
                        <Stack size={14} className="text-emerald-400 shrink-0" />
                      ) : (
                        <Folder size={14} className="text-sky-400 shrink-0" />
                      )}
                      <span className="font-mono truncate">{entry.name}</span>
                    </button>
                    {isZarr && (
                      <span className="shrink-0 rounded bg-emerald-600/30 px-1.5 py-0.5 text-[10px] font-medium text-emerald-200">
                        .zarr
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {browseRoot !== null && !browseError && (
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => { setPath(joinPath(browseRoot, browseRel)); setBrowsing(false); }}
                className="text-xs font-medium text-sky-200 hover:underline"
              >
                Use this folder ({browseRel || 'root'})
              </button>
              <button
                type="button"
                onClick={scanFolder}
                disabled={scanning}
                className="flex items-center gap-1.5 rounded-md border border-white/15 px-2 py-1 text-xs font-medium text-sky-100 hover:bg-white/10 disabled:opacity-40"
              >
                <FolderOpen size={13} />
                {scanning
                  ? 'Scanning…'
                  : `Scan folder for datasets (${browseRel || 'root'})`}
              </button>
            </div>
          )}

          {scanError && (
            <div className="flex items-start gap-2 rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-2 text-xs text-amber-200">
              <Warning size={15} className="mt-0.5 flex-shrink-0" />
              <span>{scanError}</span>
            </div>
          )}

          {scanResult && (
            <div className="space-y-1.5 rounded-md border border-white/10 bg-black/20 p-2.5 text-xs">
              <div className="flex items-start gap-2 text-emerald-300">
                <CheckCircle size={15} className="mt-0.5 flex-shrink-0" />
                <span>
                  Scanned {scanResult.scanned} — registered {scanResult.registered.length} new,
                  skipped {scanResult.skipped.length} already present
                  {scanResult.shadowed.length > 0 ? `, ${scanResult.shadowed.length} shadowed` : ''}
                  {scanResult.errors.length > 0 ? `, ${scanResult.errors.length} failed` : ''}.
                </span>
              </div>
              {scanResult.registered.length > 0 && (
                <ul className="ml-5 list-disc text-sky-200/90">
                  {scanResult.registered.map((r) => (
                    <li key={r.key} className="font-mono">{r.name}</li>
                  ))}
                </ul>
              )}
              {scanResult.shadowed.length > 0 && (
                <div className="space-y-1.5">
                  {scanResult.shadowed.map((s) => (
                    <div key={s.name} className="rounded-md border border-amber-400/30 bg-amber-400/10 p-2">
                      <p className="text-amber-200">
                        <span className="font-mono">{s.name}</span> — same name already registered as a{' '}
                        <span className="font-mono">{s.existing_kind}</span> dataset. Register this one too, under
                        a different name:
                      </p>
                      <div className="mt-1.5 flex gap-2">
                        <input
                          type="text"
                          defaultValue={s.suggested_key}
                          onChange={(e) => setRenameInputs((prev) => ({ ...prev, [s.name]: e.target.value }))}
                          className="flex-1 min-w-0 rounded-md border border-white/15 bg-white/5 px-2 py-1 font-mono text-xs text-white"
                        />
                        <button
                          type="button"
                          disabled={scanning}
                          onClick={() => retryShadowed(s.name, renameInputs[s.name] ?? s.suggested_key)}
                          className="rounded-md border border-amber-300/40 px-2 py-1 text-xs font-medium text-amber-100 hover:bg-amber-400/10 disabled:opacity-40"
                        >
                          Register as this
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {scanResult.errors.length > 0 && (
                <ul className="ml-5 list-disc text-amber-200">
                  {scanResult.errors.map((e) => (
                    <li key={e.name}>
                      <span className="font-mono">{e.name}</span>: {e.error}
                    </li>
                  ))}
                </ul>
              )}
              {scanResult.registered.length > 0 && (
                <button
                  type="button"
                  onClick={() => { onBrowse?.(containerPath, scanResult.registered.length); setBrowsing(false); }}
                  className="rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
                >
                  Go to Browse
                </button>
              )}
            </div>
          )}
        </div>
      )}

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
