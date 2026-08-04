/**
 * VolumeCanvas — a dumb, reusable WebGL2 canvas for the 3D volume view.
 *
 * Owns a `VolumeRenderer` instance and a canvas element; everything it knows
 * about the scene arrives through props (raw/label volumes, LUT, render
 * options). It has no knowledge of `datasetStore`/`annotationStore`/
 * `classStore` — wiring those up to these props is the page component's job,
 * not this one's, so this canvas can be reused or tested independent of the
 * rest of the app.
 *
 * Pointer drag orbits the camera (shift+drag pans instead), the wheel zooms,
 * and a double-click resets the view. Render quality is dropped to
 * 'interactive' for the duration of a drag and restored to 'full' on
 * release, so large volumes stay responsive while orbiting.
 */
import { useEffect, useRef, useState } from 'react';
import type { VolumeDims } from '@/lib/volume/volumeDims';
import { VolumeRenderer, VolumeRendererError, type VolumeRenderOptions } from '@/lib/volume/volumeRenderer';

export interface VolumeCanvasProps {
  raw: { data: Uint8Array; dims: VolumeDims } | null;
  label: { data: Uint8Array; dims: VolumeDims } | null;
  /** Length-1024 (256 RGBA entries) class color lookup table. */
  lut: Uint8Array;
  options: Partial<VolumeRenderOptions>;
}

type DragMode = 'orbit' | 'pan';

interface DragState {
  pointerId: number;
  mode: DragMode;
  lastX: number;
  lastY: number;
}

export default function VolumeCanvas({ raw, label, lut, options }: VolumeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<VolumeRenderer | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [initError, setInitError] = useState<{ webgl2Unavailable: boolean; message: string } | null>(null);

  // Construct the renderer once on mount, observe the parent for resizes, and
  // tear both down on unmount. Deliberately empty deps — the renderer is a
  // stable imperative object that later effects push prop changes into,
  // rather than something recreated when props change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let renderer: VolumeRenderer;
    try {
      renderer = new VolumeRenderer(canvas);
    } catch (err) {
      if (err instanceof VolumeRendererError) {
        setInitError({ webgl2Unavailable: err.code === 'webgl2-unavailable', message: err.message });
      } else {
        setInitError({ webgl2Unavailable: false, message: err instanceof Error ? err.message : String(err) });
      }
      return;
    }
    rendererRef.current = renderer;

    const target = canvas.parentElement ?? canvas;
    const ro = new ResizeObserver(() => renderer.resize());
    ro.observe(target);

    // React's onWheel prop is passive in some setups, which silently drops
    // preventDefault(); a native non-passive listener reliably stops the page
    // from scrolling while zooming over the canvas.
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      renderer.zoomBy(e.deltaY);
    };
    canvas.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      ro.disconnect();
      canvas.removeEventListener('wheel', handleWheel);
      rendererRef.current = null;
      renderer.dispose();
    };
  }, []);

  // Push the raw volume whenever it changes.
  useEffect(() => {
    if (!raw) return;
    rendererRef.current?.setRawVolume(raw.data, raw.dims);
  }, [raw]);

  // Push the label volume whenever it (or a fallback dims source) changes.
  // `setLabelVolume` requires dims even for `null` data (it still needs to
  // size an empty texture); reuse raw's dims when there's no label volume of
  // its own, or skip entirely if there's nothing to size against yet.
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    if (label) {
      renderer.setLabelVolume(label.data, label.dims);
    } else if (raw) {
      renderer.setLabelVolume(null, raw.dims);
    }
    // else: neither volume is loaded yet — nothing to size a texture against.
  }, [label, raw]);

  // Push the class LUT whenever it changes.
  useEffect(() => {
    rendererRef.current?.setClassLUT(lut);
  }, [lut]);

  // Push render options whenever they change.
  useEffect(() => {
    rendererRef.current?.setOptions(options);
  }, [options]);

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      pointerId: e.pointerId,
      mode: e.shiftKey ? 'pan' : 'orbit',
      lastX: e.clientX,
      lastY: e.clientY,
    };
    renderer.setOptions({ quality: 'interactive' });
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const renderer = rendererRef.current;
    const drag = dragRef.current;
    if (!renderer || !drag || drag.pointerId !== e.pointerId) return;
    const dx = e.clientX - drag.lastX;
    const dy = e.clientY - drag.lastY;
    if (drag.mode === 'pan') renderer.panBy(dx, dy);
    else renderer.orbit(dx, dy);
    drag.lastX = e.clientX;
    drag.lastY = e.clientY;
  };

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const renderer = rendererRef.current;
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    dragRef.current = null;
    renderer?.setOptions({ quality: 'full' });
  };

  const handleDoubleClick = () => {
    rendererRef.current?.resetView();
  };

  if (initError) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-300 text-sm text-center p-4">
        {initError.webgl2Unavailable
          ? '3D view requires WebGL2, which your browser does not support or has disabled.'
          : `3D view failed to start: ${initError.message}`}
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full block"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
      onDoubleClick={handleDoubleClick}
    />
  );
}
