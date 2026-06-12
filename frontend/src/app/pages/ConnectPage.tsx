/**
 * ConnectPage — choose a Tiled server or local folder, establish the connection,
 * see the sample count, then navigate to Browse to pick individual samples.
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { PlugsConnected, Folder, HardDrives } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { useConnectionStore } from '@/stores/connectionStore';

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

export default function ConnectPage() {
  const navigate = useNavigate();
  const { setConnection } = useConnectionStore();

  const [mode, setMode] = useState<'tiled' | 'local'>('tiled');

  // Tiled: just pick the server
  const [selectedServerUri, setSelectedServerUri] = useState<string>('');

  // Local: browse directories until you pick a folder
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

  // Default to the first server
  useEffect(() => {
    if (!selectedServerUri && servers.length > 0) setSelectedServerUri(servers[0].uri);
  }, [servers, selectedServerUri]);

  // --- Local directory listing ---
  const { data: localEntries = [], isLoading: localLoading } = useQuery<LocalEntry[]>({
    queryKey: ['localList', browseDir],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/local/list?rel=${encodeURIComponent(browseDir)}`);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    enabled: mode === 'local',
  });

  const canConnect =
    mode === 'tiled' ? !!selectedServerUri : !!selectedFolder;

  const handleConnect = async () => {
    setStatus('Connecting…');
    setConnecting(true);
    try {
      let summaryUrl: string;
      if (mode === 'tiled') {
        summaryUrl = `${API_BASE}/api/connect/summary?kind=tiled&server_uri=${encodeURIComponent(selectedServerUri)}`;
      } else {
        summaryUrl = `${API_BASE}/api/connect/summary?kind=local&rel=${encodeURIComponent(selectedFolder)}`;
      }
      const res = await fetch(summaryUrl);
      if (!res.ok) throw new Error(await res.text());
      const summary = await res.json();

      setConnection({
        kind: summary.kind,
        serverUri: summary.server_uri ?? null,
        localRoot: mode === 'local' ? selectedFolder : null,
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
    <div className="flex h-full items-start justify-center pt-16 px-6">
      <div className="w-full max-w-lg space-y-6">
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

        {/* Tiled: server dropdown only */}
        {mode === 'tiled' && (
          <div>
            <label className="text-sm font-medium text-sky-100 block mb-1">Server</label>
            {servers.length === 0 ? (
              <p className="text-sky-300 text-sm">Loading servers…</p>
            ) : (
              <select
                className="w-full border border-white/20 rounded-md px-3 py-2 text-sm bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                value={selectedServerUri}
                onChange={(e) => setSelectedServerUri(e.target.value)}
              >
                <option value="">— select server —</option>
                {servers.map((s) => (
                  <option key={s.uri} value={s.uri} className="bg-slate-800">
                    {s.name} ({s.uri})
                  </option>
                ))}
              </select>
            )}
            <p className="text-xs text-sky-300/70 mt-2">
              After connecting, use Browse to filter and select individual samples.
            </p>
          </div>
        )}

        {/* Local: directory browser — pick a folder */}
        {mode === 'local' && (
          <div className="space-y-2">
            <label className="text-sm font-medium text-sky-100 block">
              Choose a folder (<span className="font-mono text-sky-200">LOCAL_DATA_ROOT</span>)
            </label>

            {/* Breadcrumbs */}
            <div className="flex flex-wrap items-center gap-1 text-sm">
              <button
                className="px-2 py-0.5 rounded hover:bg-white/10 text-sky-200 font-medium"
                onClick={() => { setBrowseDir(''); setSelectedFolder(''); }}
              >
                root
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

            {/* Folder listing */}
            <div className="border border-white/20 rounded-md max-h-64 overflow-y-auto text-sm bg-white/5">
              {/* "Use this folder" button */}
              <button
                className={`w-full text-left px-3 py-2 border-b border-white/10 transition-colors flex items-center gap-2 ${
                  selectedFolder === browseDir
                    ? 'bg-sky-700/40 text-sky-100'
                    : 'text-sky-200 hover:bg-white/10'
                }`}
                onClick={() => setSelectedFolder(browseDir)}
              >
                <Folder size={14} weight="fill" className="text-sky-400" />
                <span className="font-medium">Use "{browseDir || 'root'}" as dataset folder</span>
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
          </div>
        )}

        {status && (
          <p className={`text-sm ${status.startsWith('Failed') ? 'text-red-400' : 'text-sky-100'}`}>
            {status}
          </p>
        )}

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
