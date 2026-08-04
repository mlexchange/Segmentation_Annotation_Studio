/**
 * useSam — React wrapper around the SAM worker singleton.
 *
 * Exposes the load/encode status, an `ensureEncoded(image)` that encodes a
 * slice at most once (cached per HTMLImageElement), and `segment(points)` for
 * per-click decoding. Used by the AnnotationCanvas magic tool.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { samClient, webgpuAvailable, type SamStatus, type PromptPoint, type PromptBox, type Granularity, type SamMask } from '@/lib/sam/samClient';

/**
 * Subscribes to the SAM worker and, when `enabled`, kicks off model load.
 * Returns status/error/backend info plus `ensureEncoded` and `segment`.
 */
export function useSam(enabled: boolean) {
  const [status, setStatus] = useState<SamStatus>(samClient.getStatus());
  const [error, setError] = useState<string | null>(null);
  // The encode is keyed by a string so that a brightness/contrast change (which
  // alters the pixels SAM sees) invalidates it and forces a re-encode.
  const encodedKeyRef = useRef<string | null>(null);
  const encodePromiseRef = useRef<Promise<void> | null>(null);
  const failedRef = useRef<Set<string>>(new Set()); // keys whose encode threw

  useEffect(() => samClient.subscribe(setStatus), []);

  // Kick off model load as soon as the tool is active.
  useEffect(() => {
    if (enabled && samClient.getStatus() === 'idle') {
      samClient.init().catch(() => { /* status flips to 'unsupported' */ });
    }
  }, [enabled]);

  /** Encode the source for `key` unless already encoded. `makeSource` is only
   *  called when an encode is actually needed (it can be expensive to build). */
  const ensureEncoded = useCallback(async (key: string, makeSource: () => CanvasImageSource): Promise<boolean> => {
    if (encodedKeyRef.current === key) return true;
    if (failedRef.current.has(key)) return false; // known-bad; don't retry
    if (encodePromiseRef.current) {
      try { await encodePromiseRef.current; } catch { /* fall through to re-encode */ }
      if (encodedKeyRef.current === key) return true;
    }
    const p = samClient.encode(makeSource()).then(() => { encodedKeyRef.current = key; });
    encodePromiseRef.current = p;
    try {
      await p;
      setError(null);
      return true;
    } catch (e) {
      failedRef.current.add(key);
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      if (encodePromiseRef.current === p) encodePromiseRef.current = null;
    }
  }, []);

  /** Decode a mask from prompt points/box at the given granularity/threshold.
   *  Returns null when there is no prompt or the decode fails. */
  const segment = useCallback(async (
    points: PromptPoint[],
    box: PromptBox | null,
    granularity: Granularity,
    threshold: number,
  ): Promise<SamMask | null> => {
    if (points.length === 0 && !box) return null;
    try {
      return await samClient.decode(points, box, granularity, threshold);
    } catch {
      return null;
    }
  }, []);

  return {
    status,
    error,
    backend: samClient.getBackend(),
    webgpu: webgpuAvailable(),
    supported: status !== 'unsupported',
    ensureEncoded,
    segment,
  };
}
