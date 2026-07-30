/**
 * BrowsePage — sample picker; Tiled=faceted metadata browser, Local=flat file list.
 * Requires an active connection (set via ConnectPage).
 */
import { useState } from 'react';
import { useNavigate } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { PlugsConnected } from '@phosphor-icons/react';
import ColumnBrowser from '@/components/Browse/ColumnBrowser';
import LocalSampleBrowser from '@/components/Browse/LocalSampleBrowser';
import { useConnectionStore } from '@/stores/connectionStore';
import { useOpenInAnnotate } from '@/hooks/useOpenInAnnotate';
import { API_BASE } from '@/config';
import type { ServerInfo } from '@/types/server';
import type { AnnotationFilter } from '@/types/annotationFilter';

export default function BrowsePage() {
  const navigate = useNavigate();
  const {
    kind,
    serverUri,
    browseContainerPath,
    browseFocusPath,
    localRoot,
    localRel,
    label,
    sampleCount,
    setConnection,
  } = useConnectionStore();
  const { openLocalFile } = useOpenInAnnotate();

  const [annotationFilter, setAnnotationFilter] = useState<AnnotationFilter>('all');

  // Server list (needed by ColumnBrowser toolbar)
  const { data: servers = [] } = useQuery<ServerInfo[]>({
    queryKey: ['servers'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/config/servers`);
      if (!res.ok) throw new Error('Failed to load servers');
      return res.json();
    },
    enabled: kind === 'tiled',
  });

  /** Switch the active Tiled connection to the chosen server URI. */
  const handleServerChange = (uri: string) => {
    setConnection({
      kind: 'tiled',
      serverUri: uri,
      label: servers.find((s) => s.uri === uri)?.name ?? uri,
      sampleCount: sampleCount ?? 0,
    });
  };

  // Not connected yet
  if (!kind) {
    return (
      <div className="flex flex-col h-full items-center justify-center gap-4">
        <PlugsConnected size={48} className="text-sky-700" />
        <p className="text-sky-200 text-sm">No dataset connected.</p>
        <button
          onClick={() => navigate('/connect')}
          className="px-4 py-2 rounded-md bg-sky-600 text-white text-sm hover:bg-sky-700"
        >
          Go to Connect
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Connection banner */}
      <div className="shrink-0 flex items-center justify-between px-4 py-1.5 bg-slate-800 border-b border-slate-700 text-xs text-slate-400">
        <span>
          <span className="font-medium text-slate-300">{label}</span>
          {sampleCount !== null && (
            <span className="ml-2 text-slate-400">
              · {sampleCount} sample{sampleCount !== 1 ? 's' : ''}
            </span>
          )}
        </span>
        <button
          className="text-sky-400 hover:text-sky-300 underline underline-offset-2"
          onClick={() => navigate('/connect')}
        >
          Change
        </button>
      </div>

      {/* Main browser — flex-1 so it fills the space under the banner and its
          own internal scroll areas are bounded (otherwise the bottom is clipped). */}
      <div className="flex-1 min-h-0">
        {kind === 'tiled' && serverUri && (
          <ColumnBrowser
            key={`${serverUri}:${browseContainerPath ?? ''}`}
            serverUri={serverUri}
            containerPath={browseContainerPath}
            focusPath={browseFocusPath}
            servers={servers}
            selectedServerUri={serverUri}
            onServerChange={handleServerChange}
            annotationFilter={annotationFilter}
            onAnnotationFilterChange={setAnnotationFilter}
          />
        )}
        {kind === 'local' && (
          <LocalSampleBrowser
            root={localRoot ?? ''}
            rel={localRel ?? ''}
            onOpenInAnnotate={openLocalFile}
            annotationFilter={annotationFilter}
            onAnnotationFilterChange={setAnnotationFilter}
          />
        )}
      </div>
    </div>
  );
}
