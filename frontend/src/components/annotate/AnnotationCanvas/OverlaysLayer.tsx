/**
 * OverlaysLayer — iPred proba / conformal-prediction / manifold-suggest overlays.
 *
 * Extracted from AnnotationCanvas so its PNG-decode work (fetch → ImageData →
 * recolor) stays isolated from the main canvas's per-frame render path. Each
 * overlay is gated by the matching `layerVisibilityStore` group, already wired
 * up in LayersPanel — this component only needs to render when told to.
 */
import { useEffect, useRef, useState } from 'react';
import { Layer, Image as KonvaImage, Rect } from 'react-konva';
import { colorizeConformalOverlay, loadLabelPng } from '@/lib/pixelClf';
import { colorizeManifoldHeatmap } from '@/lib/featureManifold';
import type { ManifoldPoint } from '@/lib/featureManifold';
import { manifoldMarkerRect } from '@/lib/featureManifold';

/** Loads a blob: URL into an <img> element; null while loading/absent. */
function useImageFromUrl(url: string | null): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) {
      setImg(null);
      return;
    }
    let cancelled = false;
    const el = new Image();
    el.onload = () => {
      if (!cancelled) setImg(el);
    };
    el.src = url;
    return () => {
      cancelled = true;
    };
  }, [url]);
  return img;
}

export interface OverlaysLayerProps {
  width: number;
  height: number;
  imageClip?: { clipX?: number; clipY?: number; clipWidth?: number; clipHeight?: number };

  showFeatures: boolean;
  featureChannelUrl: string | null;

  showProba: boolean;
  probaOverlayUrl: string | null;
  probaOpacity: number;

  showPredictions: boolean;
  clfCommitUrl: string | null;
  clfStatusUrl: string | null;
  predictionsOpacity: number;
  classColorById: Map<number, string>;
  predictionClassVisible: Record<number, boolean>;
  showPredictionMulti: boolean;
  showPredictionAbstain: boolean;

  showManifold: boolean;
  manifoldHeatmapUrl: string | null;
  manifoldHeatmapOpacity: number;
  manifoldShowHeatmap: boolean;
  manifoldMarkers: ManifoldPoint[];
  manifoldShowMarkers: boolean;
  manifoldBoxSize: number;
}

/** Fetches + decodes the conformal commit/status PNGs into a colorized canvas. */
function useConformalOverlayCanvas(
  commitUrl: string | null,
  statusUrl: string | null,
  colorById: Map<number, string>,
  classVisible: Record<number, boolean>,
  showMulti: boolean,
  showAbstain: boolean,
): HTMLCanvasElement | null {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (!commitUrl || !statusUrl) {
      setCanvas(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const [commit, status] = await Promise.all([loadLabelPng(commitUrl), loadLabelPng(statusUrl)]);
        if (cancelled) return;
        const out = colorizeConformalOverlay(commit.data, status.data, commit.width, commit.height, colorById, {
          classVisible: (cid) => classVisible[cid] !== false,
          showMulti,
          showAbstain,
        });
        if (!cancelled) setCanvas(out);
      } catch {
        if (!cancelled) setCanvas(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [commitUrl, statusUrl, colorById, classVisible, showMulti, showAbstain]);
  return canvas;
}

/** Fetches + decodes the manifold coverage PNG into a colorized canvas. */
function useManifoldHeatmapCanvas(url: string | null, opacity: number): HTMLCanvasElement | null {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (!url) {
      setCanvas(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const { data, width, height } = await loadLabelPng(url);
        if (cancelled) return;
        setCanvas(colorizeManifoldHeatmap(data, width, height, opacity));
      } catch {
        if (!cancelled) setCanvas(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [url, opacity]);
  return canvas;
}

export default function OverlaysLayer({
  width,
  height,
  imageClip,
  showFeatures,
  featureChannelUrl,
  showProba,
  probaOverlayUrl,
  probaOpacity,
  showPredictions,
  clfCommitUrl,
  clfStatusUrl,
  predictionsOpacity,
  classColorById,
  predictionClassVisible,
  showPredictionMulti,
  showPredictionAbstain,
  showManifold,
  manifoldHeatmapUrl,
  manifoldHeatmapOpacity,
  manifoldShowHeatmap,
  manifoldMarkers,
  manifoldShowMarkers,
  manifoldBoxSize,
}: OverlaysLayerProps) {
  const featureImg = useImageFromUrl(showFeatures ? featureChannelUrl : null);
  const probaImg = useImageFromUrl(showProba ? probaOverlayUrl : null);
  const conformalCanvas = useConformalOverlayCanvas(
    showPredictions ? clfCommitUrl : null,
    showPredictions ? clfStatusUrl : null,
    classColorById,
    predictionClassVisible,
    showPredictionMulti,
    showPredictionAbstain,
  );
  const manifoldCanvas = useManifoldHeatmapCanvas(
    showManifold && manifoldShowHeatmap ? manifoldHeatmapUrl : null,
    manifoldHeatmapOpacity,
  );
  const featureRef = useRef(null);
  const probaRef = useRef(null);
  const predRef = useRef(null);
  const manifoldRef = useRef(null);

  return (
    <>
      {showFeatures && featureImg && (
        <Layer listening={false} {...imageClip}>
          <KonvaImage ref={featureRef} image={featureImg} width={width} height={height} listening={false} />
        </Layer>
      )}
      {showProba && probaImg && (
        <Layer listening={false} opacity={probaOpacity} {...imageClip}>
          <KonvaImage ref={probaRef} image={probaImg} width={width} height={height} listening={false} />
        </Layer>
      )}
      {showPredictions && conformalCanvas && (
        <Layer listening={false} opacity={predictionsOpacity} {...imageClip}>
          <KonvaImage ref={predRef} image={conformalCanvas} width={width} height={height} listening={false} />
        </Layer>
      )}
      {showManifold && (
        <Layer listening={false} {...imageClip}>
          {manifoldShowHeatmap && manifoldCanvas && (
            <KonvaImage ref={manifoldRef} image={manifoldCanvas} width={width} height={height} listening={false} />
          )}
          {manifoldShowMarkers &&
            manifoldMarkers.map((pt, i) => {
              const r = manifoldMarkerRect(pt, { side: manifoldBoxSize, width, height });
              return (
                <Rect
                  key={i}
                  x={r.x}
                  y={r.y}
                  width={r.width}
                  height={r.height}
                  stroke="#22d3ee"
                  strokeWidth={1.5}
                  dash={[4, 3]}
                  listening={false}
                />
              );
            })}
        </Layer>
      )}
    </>
  );
}
