/**
 * VolumeViewer — React wrapper around the vendored WebGPU volume renderer.
 *
 * The renderer (pinned as a git submodule, see `.gitmodules` and
 * `tsconfig.app.json`'s `@zarrviewer/*` alias) is not a component: it is an
 * imperative `run(canvas, { zarrUrl, hudMount })` that boots a volume view into
 * a canvas and returns a disposable handle. This wrapper owns only the React
 * lifecycle around it — create the hosts, boot, dispose. Everything about *how*
 * the volume looks belongs upstream.
 *
 * Adapted from the viewer repo's own `src/WebGpuNative.tsx`, which had already
 * solved the mount/dispose shape; kept here rather than imported because it is
 * host-app glue, not renderer code.
 *
 * Auth: chunks are fetched with a plain `fetch()` straight to the Tiled origin.
 * That works because anonymous access is read-only-enabled there — see
 * `lib/zarrUrl.ts`. Deliberately no token interceptor: the Tiled API key is
 * write-scoped and stays server-side.
 */
import { useEffect, useRef } from 'react';
import { run, type WebGpuViewerInstance } from '@zarrviewer/ome-zarr-viewer';

export type { WebGpuViewerInstance };

/** Width of the docked HUD column, in px. Mirrored into the renderer's stage. */
const HUD_WIDTH = 320;

/**
 * Whether the renderer can run here, with a reason when it cannot.
 *
 * WebGPU is gated on a *secure context*, so over plain http `navigator.gpu` is
 * simply undefined. Distinguishing that from "this browser has no WebGPU"
 * matters: the first is a deployment fix, the second is not, and collapsing
 * them sends people looking in the wrong place.
 */
export function webGpuAvailability(): { ok: boolean; reason: string } {
  if (typeof navigator !== 'undefined' && (navigator as Navigator & { gpu?: unknown }).gpu) {
    return { ok: true, reason: '' };
  }
  if (typeof window !== 'undefined' && !window.isSecureContext) {
    return {
      ok: false,
      reason:
        'WebGPU needs a secure context. Open the app over https, or via localhost / 127.0.0.1 rather than a LAN address.',
    };
  }
  return {
    ok: false,
    reason: 'This browser does not support WebGPU. Chrome or Edge 113+ is required for the 3D view.',
  };
}

interface VolumeViewerProps {
  /** Zarr store root, from `buildZarrUrl`. */
  zarrUrl: string;
  /** Receives the handle when the renderer boots, and `null` when it is torn down. */
  onReady?: (instance: WebGpuViewerInstance | null) => void;
  /** Receives a boot failure (bad store, unsupported codec, no multiscales). */
  onError?: (error: unknown) => void;
}

export default function VolumeViewer({ zarrUrl, onReady, onError }: VolumeViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Held in refs so a changed callback identity never restarts the renderer —
  // rebooting a WebGPU device on every parent render would be ruinous.
  const onReadyRef = useRef(onReady);
  const onErrorRef = useRef(onError);
  onReadyRef.current = onReady;
  onErrorRef.current = onError;

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !zarrUrl || !webGpuAvailability().ok) return;

    let cancelled = false;
    let handle: WebGpuViewerInstance | null = null;

    // The renderer treats `canvas.parentElement` as its stage and re-fits the
    // backing store to that box every frame, so a canvas that shares the row
    // with the HUD sidebar sizes itself correctly with no resize plumbing here.
    const canvasHost = document.createElement('div');
    canvasHost.style.flex = '1';
    canvasHost.style.minWidth = '0';
    canvasHost.style.position = 'relative';

    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    canvasHost.appendChild(canvas);

    const hudHost = document.createElement('div');
    hudHost.style.width = `${HUD_WIDTH}px`;
    hudHost.style.flexShrink = '0';
    hudHost.style.overflow = 'auto';

    container.appendChild(canvasHost);
    container.appendChild(hudHost);

    run(canvas, { zarrUrl, hudMount: hudHost })
      .then((created) => {
        // StrictMode double-invokes effects in dev, so a run can resolve after
        // its own cleanup. Disposing here is what stops GPU devices piling up.
        if (cancelled) {
          created.dispose();
          return;
        }
        handle = created;
        onReadyRef.current?.(created);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        console.error('VolumeViewer: renderer failed to start:', error);
        onErrorRef.current?.(error);
      });

    return () => {
      cancelled = true;
      try {
        handle?.dispose();
      } catch (error) {
        console.warn('VolumeViewer: dispose failed:', error);
      }
      if (handle) onReadyRef.current?.(null);
      container.replaceChildren();
    };
  }, [zarrUrl]);

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        minHeight: 0,
        width: '100%',
        position: 'relative',
        display: 'flex',
        flexDirection: 'row',
      }}
    />
  );
}
