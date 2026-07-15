/**
 * useFeatureChannels — compute feature bank via ipred and load channel PNGs.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE } from '@/config';
import { useConnectionStore } from '@/stores/connectionStore';
import {
  ipredChannelUrl,
  ipredPreprocess,
  openIpredSession,
} from '@/lib/ipredApi';

export interface FeatureChannelInfo {
  index: number;
  label: string;
}

export interface FeatureJobInfo {
  jobId: string;
  width: number;
  height: number;
  channels: FeatureChannelInfo[];
  hasSam: boolean;
  setupId?: string;
  cacheHit?: boolean;
}

export interface FeatureParams {
  sigmaMin: number;
  sigmaMax: number;
  intensity: boolean;
  edges: boolean;
  texture: boolean;
  clahe: boolean;
  /** Concat SlimSAM embeddings into CatBoost (legacy in-proc path). */
  includeSam: boolean;
}

export const DEFAULT_FEATURE_PARAMS: FeatureParams = {
  sigmaMin: 1,
  sigmaMax: 8,
  intensity: true,
  edges: true,
  texture: true,
  clahe: true,
  includeSam: true,
};

export interface UseFeatureChannelsArgs {
  source: string | null;
  kind: string | null;
  sliceIndex: number;
  serverUri: string | null;
}

/**
 * Owns feature compute + channel blob URL lifecycle via ipred.
 * `channelIndex === null` means show the original rendered slice.
 */
export function useFeatureChannels({
  source,
  kind,
  sliceIndex,
  serverUri,
}: UseFeatureChannelsArgs) {
  const [params, setParams] = useState<FeatureParams>(DEFAULT_FEATURE_PARAMS);
  const [job, setJob] = useState<FeatureJobInfo | null>(null);
  const [channelIndex, setChannelIndex] = useState<number | null>(null);
  const [channelUrl, setChannelUrl] = useState<string | null>(null);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [samAvailable, setSamAvailable] = useState(false);
  const channelUrlRef = useRef<string | null>(null);

  const preferredFeatureSetupId = useConnectionStore((s) => s.preferredFeatureSetupId);
  const preferredCompositionId = useConnectionStore((s) => s.preferredCompositionId);
  const ipredSessionId = useConnectionStore((s) => s.ipredSessionId);
  const localRoot = useConnectionStore((s) => s.localRoot);
  const setIpredSession = useConnectionStore((s) => s.setIpredSession);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/image/features/sam-status`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { available?: boolean } | null) => {
        if (!cancelled && data) setSamAvailable(!!data.available);
      })
      .catch(() => {
        if (!cancelled) setSamAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const revokeChannelUrl = useCallback(() => {
    if (channelUrlRef.current) {
      URL.revokeObjectURL(channelUrlRef.current);
      channelUrlRef.current = null;
    }
    setChannelUrl(null);
  }, []);

  useEffect(() => {
    setJob(null);
    setChannelIndex(null);
    revokeChannelUrl();
    setError(null);
  }, [source, kind, sliceIndex, serverUri, revokeChannelUrl]);

  useEffect(() => {
    if (!job || channelIndex === null) {
      revokeChannelUrl();
      return;
    }
    let cancelled = false;
    const url = ipredChannelUrl(job.jobId, channelIndex);
    (async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Channel fetch failed: ${res.status}`);
        const blob = await res.blob();
        if (cancelled) return;
        const objUrl = URL.createObjectURL(blob);
        if (channelUrlRef.current) URL.revokeObjectURL(channelUrlRef.current);
        channelUrlRef.current = objUrl;
        setChannelUrl(objUrl);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          revokeChannelUrl();
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [job, channelIndex, revokeChannelUrl]);

  useEffect(
    () => () => {
      if (channelUrlRef.current) URL.revokeObjectURL(channelUrlRef.current);
    },
    [],
  );

  const ensureSession = useCallback(async (): Promise<string> => {
    if (ipredSessionId) return ipredSessionId;
    if (!source || !kind) throw new Error('No sample open');
    const session = await openIpredSession({
      kind,
      source,
      server_uri: serverUri,
      root: kind === 'local' ? localRoot : null,
    });
    setIpredSession({
      sessionId: session.session_id,
      projectId: session.project_id,
    });
    return session.session_id;
  }, [ipredSessionId, source, kind, serverUri, localRoot, setIpredSession]);

  const compute = useCallback(async () => {
    if (!source || !kind || computing) return;
    const compositionId = preferredCompositionId || preferredFeatureSetupId;
    if (!compositionId) {
      setError('Select a composition on the Ipred tab first.');
      return;
    }
    setComputing(true);
    setError(null);
    try {
      const sessionId = await ensureSession();
      const data = await ipredPreprocess({
        session_id: sessionId,
        composition_id: compositionId,
        slice_index: sliceIndex,
      });
      const channels: FeatureChannelInfo[] = (data.labels ?? []).map((label, index) => ({
        index,
        label,
      }));
      if (channels.length === 0) {
        for (let i = 0; i < data.n_channels; i += 1) {
          channels.push({ index: i, label: `channel ${i}` });
        }
      }
      setJob({
        jobId: data.feature_id,
        width: data.width,
        height: data.height,
        channels,
        hasSam:
          compositionId.includes('slimsam') ||
          compositionId.includes('sam') ||
          compositionId.includes('mark'),
        setupId: data.setup_id,
        cacheHit: data.cache_hit,
      });
      setChannelIndex(0);
      if (data.cache_hit) {
        // Soft notice — not an error.
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setJob(null);
      setChannelIndex(null);
    } finally {
      setComputing(false);
    }
  }, [
    source,
    kind,
    sliceIndex,
    computing,
    preferredCompositionId,
    preferredFeatureSetupId,
    ensureSession,
  ]);

  const selectChannel = useCallback((index: number | null) => {
    setChannelIndex(index);
  }, []);

  const cycleChannel = useCallback(
    (delta: number) => {
      if (!job || job.channels.length === 0) return;
      setChannelIndex((prev) => {
        const cur = prev ?? 0;
        const n = job.channels.length;
        return (((cur + delta) % n) + n) % n;
      });
    },
    [job],
  );

  const clearSelection = useCallback(() => {
    setChannelIndex(null);
  }, []);

  const invalidateJob = useCallback(() => {
    setJob(null);
    setChannelIndex(null);
    revokeChannelUrl();
  }, [revokeChannelUrl]);

  /** Adopt a feature bank produced elsewhere (e.g. Train auto-preprocess). */
  const adoptFeatureBank = useCallback(
    (payload: {
      featureId: string;
      width: number;
      height: number;
      labels?: string[];
      nChannels?: number;
      setupId?: string;
      cacheHit?: boolean;
    }) => {
      const n = payload.nChannels ?? payload.labels?.length ?? 0;
      const channels: FeatureChannelInfo[] = (payload.labels ?? []).map((label, index) => ({
        index,
        label,
      }));
      if (channels.length === 0) {
        for (let i = 0; i < n; i += 1) {
          channels.push({ index: i, label: `channel ${i}` });
        }
      }
      setJob({
        jobId: payload.featureId,
        width: payload.width,
        height: payload.height,
        channels,
        hasSam: (payload.setupId ?? '').includes('sam'),
        setupId: payload.setupId,
        cacheHit: payload.cacheHit,
      });
    },
    [],
  );

  return {
    params,
    setParams,
    job,
    channelIndex,
    channelUrl,
    computing,
    error,
    samAvailable,
    preferredFeatureSetupId,
    preferredCompositionId,
    compute,
    selectChannel,
    cycleChannel,
    clearSelection,
    invalidateJob,
    adoptFeatureBank,
  };
}
