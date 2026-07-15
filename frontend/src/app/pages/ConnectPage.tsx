/**
 * ConnectPage — choose a Tiled server or local folder, establish the connection,
 * see the sample count, then navigate to Browse to pick individual samples.
 *
 * Tiled mode also supports picking a specific browse container and ingesting
 * new data via drag-and-drop. Local mode lets the user grant access to any
 * absolute folder on the backend machine.
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { PlugsConnected, Folder, HardDrives, Stack } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useConnectionStore } from '@/stores/connectionStore';
import { useOpenInAnnotate } from '@/hooks/useOpenInAnnotate';
import IngestDropzone from '@/components/Ingest/IngestDropzone';
import SessionSetupPanel from '@/components/SessionSetupPanel';

interface ServerInfo {
  name: string;
  uri: string;
  has_api_key: boolean;
}

interface LocalEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
}

interface TiledEntry {
  name: string;
  path: string;
  is_dir: boolean;
  is_array: boolean;
}

export default function ConnectPage() {
  const navigate = useNavigate();
  const { setConnection } = useConnectionStore();
  const { openTiledArray } = useOpenInAnnotate();

  const [mode, setMode] = useState<'tiled' | 'local'>('tiled');

  // Tiled: pick the server + optionally a browse container
  const [selectedServerUri, setSelectedServerUri] = useState<string>('');
  const [containerDir, setContainerDir] = useState<string>(''); // currently-browsed container node
  const [selectedContainer, setSelectedContainer] = useState<string>(''); // chosen browse target

  // Local: grant an absolute root, then browse subfolders until you pick a folder
  const [grantedRoot, setGrantedRoot] = useState<string>('');
  const [rootInput, setRootInput] = useState<string>('');
  const [browseDir, setBrowseDir] = useState<string>('');
  const [selectedFolder, setSelectedFolder] = useState<string>('');

  const [status, setStatus] = useState<string>('');
  const [connecting, setConnecting] = useState(false);

  // --- Servers list ---
  const { data: servers = [] } = useQuery<ServerInfo[]>({
    queryKey: ['servers'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/config/servers`);
      if (!res.ok) throw new Error('Failed to load servers');
      return res.json();
    },
  });

  useEffect(() => {
    if (!selectedServerUri && servers.length > 0) setSelectedServerUri(servers[0].uri);
  }, [servers, selectedServerUri]);

  // --- Tiled container listing (for the optional browse-target picker) ---
  const { data: tiledEntries = [] } = useQuery<TiledEntry[]>({
    queryKey: ['tiledList', selectedServerUri, containerDir],
    queryFn: async () => {
      const res = await fetch(
        `${API_BASE}/api/tiled/list?server_uri=${encodeURIComponent(selectedServerUri)}&path=${encodeURIComponent(containerDir)}`,
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    enabled: mode === 'tiled' && !!selectedServerUri,
  });

  // --- Local directory listing ---
  const { data: localEntries = [], isLoading: localLoading } = useQuery<LocalEntry[]>({
    queryKey: ['localList', grantedRoot, browseDir],
    queryFn: async () => {
      const res = await fetch(
        `${API_BASE}/api/local/list?root=${encodeURIComponent(grantedRoot)}&rel=${encodeURIComponent(browseDir)}`,
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    enabled: mode === 'local' && !!grantedRoot,
  });

  const canConnect = mode === 'tiled' ? !!selectedServerUri : !!grantedRoot && !!selectedFolder;

  /** Set the Tiled connection (optional browse container) and, by default, navigate to Browse. */
  const connectTiled = (containerPath: string | null, gotoBrowse = true) => {
    setConnection({
      kind: 'tiled',
      serverUri: selectedServerUri,
      browseContainerPath: containerPath,
      label: servers.find((s) => s.uri === selectedServerUri)?.name ?? selectedServerUri,
      sampleCount: 0,
    });
    if (gotoBrowse) navigate('/browse');
  };

  // Jump straight from ingest to the Annotate tab for the first uploaded sample.
  const annotateIngested = (containerPath: string, firstKey: string) => {
    connectTiled(containerPath, false); // set connection context, don't navigate to Browse
    void openTiledArray(`${containerPath}/${firstKey}`, selectedServerUri); // → Preprocess
  };

  /** Fetch a connection summary for the chosen Tiled server or local folder, store it, then go to Browse. */
  const handleConnect = async () => {
    setStatus('Connecting…');
    setConnecting(true);
    try {
      const params = new URLSearchParams({ kind: mode });
      if (mode === 'tiled') {
        params.set('server_uri', selectedServerUri);
        if (selectedContainer) params.set('container_path', selectedContainer);
      } else {
        params.set('root', grantedRoot);
        params.set('rel', selectedFolder);
      }
      const res = await fetch(`${API_BASE}/api/connect/summary?${params}`);
      if (!res.ok) throw new Error(await res.text());
      const summary = await res.json();

      setConnection({
        kind: summary.kind,
        serverUri: summary.server_uri ?? null,
        browseContainerPath:
          mode === 'tiled'
            ? (selectedContainer || summary.container_path || null)
            : null,
        localRoot: mode === 'local' ? grantedRoot : null,
        localRel: mode === 'local' ? selectedFolder : null,
        label: summary.label,
        sampleCount: summary.sample_count,
      });

      setStatus(`Connected — ${summary.sample_count} sample${summary.sample_count === 1 ? '' : 's'} found`);
      setTimeout(() => navigate('/browse'), 600);
    } catch (e) {
      setStatus(`Failed: ${e}`);
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="flex h-full items-start justify-center pt-16 px-6 overflow-y-auto">
      <div className="w-full max-w-lg space-y-6 pb-16">
        <div className="flex items-center gap-3">
          <PlugsConnected size={28} className="text-sky-300" />
          <h2 className="text-2xl font-semibold text-white">Connect to Dataset</h2>
        </div>

        {/* Mode selector */}
        <div className="flex gap-2">
          {(['tiled', 'local'] as const).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setStatus(''); }}
              className={`px-4 py-2 rounded-md text-sm font-medium border transition-colors ${
                mode === m
                  ? 'bg-sky-600 text-white border-sky-700'
                  : 'bg-white/10 text-sky-100 border-white/20 hover:bg-white/20'
              }`}
            >
              {m === 'tiled' ? (
                <span className="flex items-center gap-2"><HardDrives size={16} /> Tiled Server</span>
              ) : (
                <span className="flex items-center gap-2"><Folder size={16} /> Local Folder</span>
              )}
            </button>
          ))}
        </div>

        {/* Tiled: server dropdown + optional container picker + ingest */}
        {mode === 'tiled' && (
          <div className="space-y-5">
            <div>
              <label className="text-sm font-medium text-sky-100 block mb-1">Server</label>
              {servers.length === 0 ? (
                <p className="text-sky-300 text-sm">Loading servers…</p>
              ) : (
                <select
                  className="w-full border border-white/20 rounded-md px-3 py-2 text-sm bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                  value={selectedServerUri}
                  onChange={(e) => { setSelectedServerUri(e.target.value); setContainerDir(''); setSelectedContainer(''); }}
                >
                  <option value="">— select server —</option>
                  {servers.map((s) => (
                    <option key={s.uri} value={s.uri} className="bg-slate-800">
                      {s.name} ({s.uri})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Optional browse-target container picker */}
            {selectedServerUri && (
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm font-medium text-sky-100">
                  <Stack size={15} /> Dataset to view <span className="text-sky-300/60 font-normal">(optional)</span>
                </label>
                <p className="text-xs text-sky-300/70">
                  Which collection on the server the <span className="font-medium">Browse</span> tab will show.
                  Most people can leave this blank — it auto-detects, and ingesting below fills it in for you.
                </p>
                <div className="flex flex-wrap items-center gap-1 text-sm">
                  <button
                    className="px-2 py-0.5 rounded hover:bg-white/10 text-sky-200 font-medium"
                    onClick={() => { setContainerDir(''); setSelectedContainer(''); }}
                  >
                    root
                  </button>
                  {containerDir.split('/').filter(Boolean).map((seg, i, arr) => {
                    const target = arr.slice(0, i + 1).join('/');
                    return (
                      <span key={target} className="flex items-center gap-1">
                        <span className="text-sky-300/70">/</span>
                        <button
                          className="px-2 py-0.5 rounded hover:bg-white/10 text-sky-200"
                          onClick={() => { setContainerDir(target); setSelectedContainer(''); }}
                        >
                          {seg}
                        </button>
                      </span>
                    );
                  })}
                </div>
                <div className="border border-white/20 rounded-md max-h-48 overflow-y-auto text-sm bg-white/5">
                  {containerDir && (
                    <button
                      className={`w-full text-left px-3 py-2 border-b border-white/10 flex items-center gap-2 ${
                        selectedContainer === containerDir ? 'bg-sky-700/40 text-sky-100' : 'text-sky-200 hover:bg-white/10'
                      }`}
                      onClick={() => setSelectedContainer(containerDir)}
                    >
                      <Stack size={14} className="text-sky-400" />
                      <span className="font-medium">Browse "{containerDir}"</span>
                    </button>
                  )}
                  {tiledEntries.filter((e) => e.is_dir).map((e) => (
                    <button
                      key={e.path}
                      className="w-full flex items-center gap-2 px-3 py-2 border-b border-white/10 last:border-b-0 hover:bg-white/10 text-sky-100"
                      onClick={() => { setContainerDir(e.path); setSelectedContainer(''); }}
                    >
                      <Folder size={14} className="text-sky-400 shrink-0" />
                      <span className="font-mono truncate">{e.name}</span>
                    </button>
                  ))}
                  {tiledEntries.filter((e) => e.is_dir).length === 0 && (
                    <div className="px-3 py-3 text-sky-300/60 text-xs">
                      No sub-containers. Leave unset to auto-discover, or pick a parent.
                    </div>
                  )}
                </div>
                {selectedContainer && (
                  <p className="text-xs text-emerald-300/80">
                    Browse will show <span className="font-mono">{selectedContainer}</span>
                  </p>
                )}
              </div>
            )}

            {/* Drag-and-drop ingest */}
            {selectedServerUri && (
              <div className="border-t border-white/10 pt-4">
                <IngestDropzone
                  serverUri={selectedServerUri}
                  onBrowse={(containerPath) => connectTiled(containerPath)}
                  onAnnotate={annotateIngested}
                />
              </div>
            )}
          </div>
        )}

        {/* Local: grant a root, then browse subfolders */}
        {mode === 'local' && (
          <div className="space-y-3">
            <label className="text-sm font-medium text-sky-100 block">
              Grant access to a folder
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={rootInput}
                onChange={(e) => setRootInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { setGrantedRoot(rootInput.trim()); setBrowseDir(''); setSelectedFolder(''); } }}
                placeholder="/absolute/path/to/data"
                className="flex-1 border border-white/20 rounded-md px-3 py-2 text-sm font-mono bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
              />
              <button
                onClick={() => { setGrantedRoot(rootInput.trim()); setBrowseDir(''); setSelectedFolder(''); }}
                disabled={!rootInput.trim()}
                className="px-3 py-2 rounded-md bg-sky-600 text-white text-sm font-medium hover:bg-sky-700 disabled:opacity-40"
              >
                Grant
              </button>
            </div>

            {grantedRoot && (
              <>
                <div className="flex flex-wrap items-center gap-1 text-sm">
                  <button
                    className="px-2 py-0.5 rounded hover:bg-white/10 text-sky-200 font-medium font-mono"
                    onClick={() => { setBrowseDir(''); setSelectedFolder(''); }}
                  >
                    {grantedRoot}
                  </button>
                  {browseDir.split('/').filter(Boolean).map((seg, i, arr) => {
                    const target = arr.slice(0, i + 1).join('/');
                    return (
                      <span key={target} className="flex items-center gap-1">
                        <span className="text-sky-300/70">/</span>
                        <button
                          className="px-2 py-0.5 rounded hover:bg-white/10 text-sky-200"
                          onClick={() => { setBrowseDir(target); setSelectedFolder(''); }}
                        >
                          {seg}
                        </button>
                      </span>
                    );
                  })}
                </div>

                <div className="border border-white/20 rounded-md max-h-64 overflow-y-auto text-sm bg-white/5">
                  <button
                    className={`w-full text-left px-3 py-2 border-b border-white/10 transition-colors flex items-center gap-2 ${
                      selectedFolder === browseDir ? 'bg-sky-700/40 text-sky-100' : 'text-sky-200 hover:bg-white/10'
                    }`}
                    onClick={() => setSelectedFolder(browseDir)}
                  >
                    <Folder size={14} weight="fill" className="text-sky-400" />
                    <span className="font-medium">Use "{browseDir || 'this root'}" as dataset folder</span>
                  </button>

                  {browseDir && (
                    <button
                      className="w-full text-left px-3 py-2 hover:bg-white/10 border-b border-white/10 text-sky-300"
                      onClick={() => { setBrowseDir(browseDir.split('/').slice(0, -1).join('/')); setSelectedFolder(''); }}
                    >
                      ↑ up one level
                    </button>
                  )}

                  {localLoading && <div className="px-3 py-4 text-sky-300">Loading…</div>}
                  {!localLoading &&
                    localEntries
                      .filter((e) => e.is_dir)
                      .map((e) => (
                        <button
                          key={e.path}
                          className="w-full flex items-center gap-2 px-3 py-2 border-b border-white/10 last:border-b-0 hover:bg-white/10 text-sky-100"
                          onClick={() => { setBrowseDir(e.path); setSelectedFolder(''); }}
                        >
                          <Folder size={14} className="text-sky-400 shrink-0" />
                          <span className="font-mono truncate">{e.name}</span>
                        </button>
                      ))}

                  {!localLoading && localEntries.filter((e) => e.is_dir).length === 0 && (
                    <div className="px-3 py-4 text-sky-300/60 text-xs">No sub-folders here.</div>
                  )}
                </div>

                {selectedFolder !== '' && (
                  <p className="text-sm text-sky-100">
                    Selected folder:{' '}
                    <span className="font-mono text-sky-200">{selectedFolder || 'root'}</span>
                  </p>
                )}
              </>
            )}
          </div>
        )}

        {status && (
          <p className={`text-sm ${status.startsWith('Failed') ? 'text-red-400' : 'text-sky-100'}`}>
            {status}
          </p>
        )}

        <SessionSetupPanel />

        <button
          onClick={handleConnect}
          disabled={!canConnect || connecting}
          className="w-full bg-sky-600 text-white rounded-md py-2.5 text-sm font-medium hover:bg-sky-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          <PlugsConnected size={18} />
          {connecting ? 'Connecting…' : 'Connect'}
        </button>
      </div>
    </div>
  );
}
