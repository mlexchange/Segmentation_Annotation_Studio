/**
 * Feature-manifold heatmap helpers for Suggest Labels overlay.
 */
import { loadLabelPng } from '@/lib/pixelClf';

export interface ManifoldPoint {
  x: number;
  y: number;
  cluster: number;
  score?: number;
  /** Exclusion / packing radius — spacing only, not box half-size. */
  radius?: number;
  dist?: number;
  /** Full square side in image pixels (preferred for drawing). */
  box_size?: number;
  /** Axis-aligned box; x1/y1 are exclusive max coords (width = x1 − x0). */
  box?: { x0: number; y0: number; x1: number; y1: number };
}

/**
 * Axis-aligned marker rect in image pixels for a suggest-label center.
 *
 * Prefer the server `box` (already clipped, non-overlapping). Never use
 * exclusion `radius` for geometry. When reconstructing, **clip** — do not
 * shift to preserve side length (shifting causes overlays to overlap).
 */
export function manifoldMarkerRect(
  point: Pick<ManifoldPoint, 'x' | 'y' | 'box_size' | 'box'>,
  opts: { side?: number; width: number; height: number },
): { x: number; y: number; width: number; height: number } {
  if (point.box) {
    const { x0, y0, x1, y1 } = point.box;
    return {
      x: x0,
      y: y0,
      width: Math.max(1, x1 - x0),
      height: Math.max(1, y1 - y0),
    };
  }
  const side = Math.max(8, opts.side ?? point.box_size ?? 64);
  const half = side / 2;
  const x0 = Math.max(0, point.x - half);
  const y0 = Math.max(0, point.y - half);
  const x1 = Math.min(opts.width, point.x + half);
  const y1 = Math.min(opts.height, point.y + half);
  return {
    x: x0,
    y: y0,
    width: Math.max(1, x1 - x0),
    height: Math.max(1, y1 - y0),
  };
}

/** Cyan→magenta coverage colormap; alpha scales with score. */
export function colorizeManifoldHeatmap(
  gray: Uint8Array,
  width: number,
  height: number,
  opacity = 0.45,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const img = ctx.createImageData(width, height);
  const out = img.data;
  const aScale = Math.round(Math.min(1, Math.max(0, opacity)) * 255);
  for (let i = 0; i < gray.length; i++) {
    const t = gray[i] / 255;
    // low residual (dark) → cool; high residual interestingness → warm
    const r = Math.round(40 + 200 * t);
    const g = Math.round(180 * (1 - t) + 40 * t);
    const b = Math.round(220 * (1 - 0.3 * t));
    const p = i * 4;
    out[p] = r;
    out[p + 1] = g;
    out[p + 2] = b;
    out[p + 3] = Math.round(aScale * (0.15 + 0.85 * t));
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Load grayscale coverage PNG and colorize. */
export async function loadManifoldHeatmapCanvas(
  url: string,
  opacity = 0.45,
): Promise<HTMLCanvasElement> {
  const { data, width, height } = await loadLabelPng(url);
  return colorizeManifoldHeatmap(data, width, height, opacity);
}
