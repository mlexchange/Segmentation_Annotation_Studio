/**
 * useConnectionHealth — periodic Tiled reachability check, backed by
 * connectionStore's `status` field. Mounted once in the app shell so
 * connection health is visible from anywhere (HubHeader), not just Browse's
 * own local facets poll or Connect's initial probe.
 *
 * Local connections have no network dependency to check, so status simply
 * stays 'unknown' (hidden in the UI) for kind === 'local' or no connection.
 */
import { useEffect, useRef } from 'react';
import { API_BASE } from '@/config';
import { useConnectionStore } from '@/stores/connectionStore';

const BASE_INTERVAL_MS = 20_000;
const MAX_INTERVAL_MS = 120_000;

export function useConnectionHealth(): void {
  const kind = useConnectionStore((s) => s.kind);
  const serverUri = useConnectionStore((s) => s.serverUri);
  const setStatus = useConnectionStore((s) => s.setStatus);
  const intervalRef = useRef(BASE_INTERVAL_MS);

  useEffect(() => {
    if (kind !== 'tiled') {
      setStatus('unknown');
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    intervalRef.current = BASE_INTERVAL_MS;

    const check = async () => {
      try {
        const params = new URLSearchParams({ path: '' });
        if (serverUri) params.set('server_uri', serverUri);
        const res = await fetch(`${API_BASE}/api/tiled/list?${params}`);
        if (cancelled) return;
        if (res.ok) {
          setStatus('ok');
          intervalRef.current = BASE_INTERVAL_MS;
        } else {
          setStatus('error');
          intervalRef.current = Math.min(intervalRef.current * 2, MAX_INTERVAL_MS);
        }
      } catch {
        if (cancelled) return;
        setStatus('error');
        intervalRef.current = Math.min(intervalRef.current * 2, MAX_INTERVAL_MS);
      } finally {
        if (!cancelled) timer = setTimeout(check, intervalRef.current);
      }
    };

    void check();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [kind, serverUri, setStatus]);
}
