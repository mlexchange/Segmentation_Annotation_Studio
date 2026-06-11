/**
 * ConnectPage — pick a Tiled dataset or local folder, populate datasetStore,
 * offer to restore a previous annotation session.
 */
import { useState } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { loadDraft } from '@/hooks/useDraftSync';

interface ServerInfo {
  name: string;
  uri: string;
  has_api_key: boolean;
}

export default function ConnectPage() {
  const navigate = useNavigate();
  const { setDataset } = useDatasetStore();
  const { setClasses } = useClassStore();
  const { loadFromDraft } = useAnnotationStore();

  const [selectedServer, setSelectedServer] = useState<ServerInfo | null>(null);
  const [tiledPath, setTiledPath] = useState('');
  const [localPath, setLocalPath] = useState('');
  const [mode, setMode] = useState<'tiled' | 'local'>('tiled');
  const [status, setStatus] = useState('');
  const [restoreModal, setRestoreModal] = useState<{ savedAt: string; draft: Record<string, unknown> } | null>(null);

  const { data: servers = [] } = useQuery<ServerInfo[]>({
    queryKey: ['servers'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/config/servers`);
      if (!res.ok) throw new Error('Failed to load servers');
      return res.json();
    },
  });

  const { data: localEntries = [] } = useQuery({
    queryKey: ['localList', localPath],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/local/list?rel=${encodeURIComponent(localPath)}`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: mode === 'local',
  });

  const handleConnect = async () => {
    const source = mode === 'tiled' ? tiledPath : localPath;
    const serverUri = mode === 'tiled' ? selectedServer?.uri ?? null : null;
    if (!source) { setStatus('Please enter a path.'); return; }

    setStatus('Loading…');
    try {
      const params = new URLSearchParams({ source, kind: mode });
      if (serverUri) params.set('server_uri', serverUri);
      const res = await fetch(`${API_BASE}/api/image/meta?${params}`);
      if (!res.ok) { setStatus(`Error: ${res.status} ${await res.text()}`); return; }
      const meta = await res.json();
      setDataset(mode, source, serverUri, {
        nSlices: meta.n_slices,
        height: meta.height,
        width: meta.width,
        dtype: meta.dtype,
        isRgb: meta.is_rgb,
        valueRange: meta.value_range,
      });

      // Check for a saved draft
      const sourceKey = mode === 'tiled' ? `tiled:${serverUri ?? ''}:${source}` : `local:${source}`;
      const draft = await loadDraft(sourceKey);
      if (draft?.payload) {
        setRestoreModal({ savedAt: draft.saved_at as string, draft: draft.payload as Record<string, unknown> });
      } else {
        navigate('/annotate');
      }
    } catch (e) {
      setStatus(`Failed: ${e}`);
    }
  };

  const handleRestore = (payload: Record<string, unknown>) => {
    if (Array.isArray(payload.classes)) setClasses(payload.classes as any);
    loadFromDraft({
      byImage: (payload.slices ?? {}) as any,
      splitBySlice: (payload.split_by_slice ?? {}) as any,
      negativeSlices: (payload.negative_slices ?? []) as any,
    });
    setRestoreModal(null);
    navigate('/annotate');
  };

  return (
    <div className="flex h-full items-start justify-center pt-16 px-6">
      <div className="w-full max-w-lg space-y-6">
        <h2 className="text-2xl font-semibold text-gray-800">Connect to Dataset</h2>

        {/* Mode selector */}
        <div className="flex gap-2">
          {(['tiled', 'local'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-2 rounded-md text-sm font-medium border transition-colors ${
                mode === m ? 'bg-sky-600 text-white border-sky-700' : 'bg-white border-gray-300 hover:bg-gray-50'
              }`}
            >
              {m === 'tiled' ? 'Tiled Server' : 'Local Folder'}
            </button>
          ))}
        </div>

        {mode === 'tiled' && (
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">Server</label>
              <select
                className="w-full border rounded-md px-3 py-2 text-sm"
                value={selectedServer?.uri ?? ''}
                onChange={(e) => setSelectedServer(servers.find((s) => s.uri === e.target.value) ?? null)}
              >
                <option value="">— select server —</option>
                {servers.map((s) => (
                  <option key={s.uri} value={s.uri}>{s.name} ({s.uri})</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">Tiled path</label>
              <input
                className="w-full border rounded-md px-3 py-2 text-sm font-mono"
                placeholder="e.g. browse/generated_data/gen_010005"
                value={tiledPath}
                onChange={(e) => setTiledPath(e.target.value)}
              />
            </div>
          </div>
        )}

        {mode === 'local' && (
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">Local path (relative to LOCAL_DATA_ROOT)</label>
              <input
                className="w-full border rounded-md px-3 py-2 text-sm font-mono"
                placeholder="e.g. my_stack.tif"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
              />
            </div>
            {localEntries.length > 0 && (
              <div className="border rounded-md max-h-48 overflow-y-auto text-sm">
                {localEntries.map((e: any) => (
                  <button
                    key={e.path}
                    className="w-full text-left px-3 py-1.5 hover:bg-sky-50 font-mono border-b last:border-b-0"
                    onClick={() => setLocalPath(e.path)}
                  >
                    {e.is_dir ? '📁 ' : '📄 '}{e.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {status && <p className="text-sm text-gray-600">{status}</p>}

        <button
          onClick={handleConnect}
          className="w-full bg-sky-600 text-white rounded-md py-2.5 text-sm font-medium hover:bg-sky-700 transition-colors"
        >
          Connect
        </button>
      </div>

      {/* Restore modal */}
      {restoreModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-sm w-full space-y-4">
            <h3 className="text-lg font-semibold">Restore session?</h3>
            <p className="text-sm text-gray-600">
              Found a saved session from {new Date(restoreModal.savedAt).toLocaleString()}.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => { setRestoreModal(null); navigate('/annotate'); }}
                className="px-4 py-2 text-sm rounded-md border hover:bg-gray-50"
              >
                Start fresh
              </button>
              <button
                onClick={() => handleRestore(restoreModal.draft)}
                className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white hover:bg-sky-700"
              >
                Restore
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
