/**
 * Pixel-classifier helpers: label PNG → colorized overlay / polygon shapes.
 */
import { v4 as uuidv4 } from 'uuid';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import { rasterizeUnion } from '@/lib/rasterize';
import type { PolygonShape, Shape } from '@/stores/annotationStore';
import { viridisRgb } from '@/lib/viridis';

/** Parse #rgb / #rrggbb into [r,g,b]. */
export function parseHexColor(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  if (h.length === 3) {
    return [
      parseInt(h[0] + h[0], 16),
      parseInt(h[1] + h[1], 16),
      parseInt(h[2] + h[2], 16),
    ];
  }
  return [
    parseInt(h.slice(0, 2), 16) || 0,
    parseInt(h.slice(2, 4), 16) || 0,
    parseInt(h.slice(4, 6), 16) || 0,
  ];
}

/** Mask-set name for a thresholded softmax class cache. */
export function classProbaMaskSetName(
  classId: number,
  threshold: number,
  classLabel?: string | null,
): string {
  const t = Math.round(Math.min(1, Math.max(0, threshold)) * 100);
  const label = (classLabel ?? '').trim() || `class ${classId}`;
  return `${label} p≥${t}%`;
}

/**
 * Colorize grayscale probability RGBA with viridis; pixels below ``threshold``
 * are transparent. Values at/above the cut are remapped to [0,1] so the full
 * colormap spans the visible range.
 */
export function applyProbaThresholdRgba(
  rgba: Uint8ClampedArray,
  threshold: number,
): void {
  const t = Math.min(1, Math.max(0, threshold));
  const span = Math.max(1e-6, 1 - t);
  for (let i = 0; i < rgba.length; i += 4) {
    const p = rgba[i] / 255;
    if (p < t) {
      rgba[i] = 0;
      rgba[i + 1] = 0;
      rgba[i + 2] = 0;
      rgba[i + 3] = 0;
      continue;
    }
    const [r, g, b] = viridisRgb((p - t) / span);
    rgba[i] = r;
    rgba[i + 1] = g;
    rgba[i + 2] = b;
    rgba[i + 3] = 255;
  }
}

/** Render a softmax grayscale PNG as viridis, clipped below ``threshold``. */
export async function thresholdProbaPngBlob(
  sourceBlob: Blob,
  threshold: number,
): Promise<Blob> {
  const url = URL.createObjectURL(sourceBlob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Failed to decode probability PNG'));
      el.src = url;
    });
    const width = img.naturalWidth;
    const height = img.naturalHeight;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D context unavailable');
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, width, height);
    applyProbaThresholdRgba(imageData.data, threshold);
    ctx.putImageData(imageData, 0, 0);
    const out = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
        'image/png',
      );
    });
    return out;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Load a grayscale classId PNG into Uint8Array + dimensions. */
export async function loadLabelPng(
  url: string,
): Promise<{ data: Uint8Array; width: number; height: number }> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('Failed to load prediction PNG'));
    el.src = url;
  });
  const width = img.naturalWidth;
  const height = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D context unavailable');
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const data = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < data.length; i++, p += 4) data[i] = rgba[p];
  return { data, width, height };
}

/**
 * Build a translucent RGBA canvas (one pixel per classId with known color).
 * Unknown / 0 stays transparent.
 */
export function colorizeLabelMap(
  labels: Uint8Array,
  width: number,
  height: number,
  colorByClass: Map<number, string>,
  alpha = 110,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const img = ctx.createImageData(width, height);
  const out = img.data;
  const rgbCache = new Map<number, [number, number, number]>();
  for (const [cid, hex] of colorByClass) rgbCache.set(cid, parseHexColor(hex));

  for (let i = 0; i < labels.length; i++) {
    const cid = labels[i];
    if (!cid) continue;
    const rgb = rgbCache.get(cid);
    if (!rgb) continue;
    const p = i * 4;
    out[p] = rgb[0];
    out[p + 1] = rgb[1];
    out[p + 2] = rgb[2];
    out[p + 3] = alpha;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Build conformal overlay: singleton = class color, multi = amber hatch,
 * abstain = dark translucent.
 *
 * Optional filters hide prediction classes / multi / abstain for the Layers panel.
 */
export function colorizeConformalOverlay(
  commit: Uint8Array,
  status: Uint8Array,
  width: number,
  height: number,
  colorByClass: Map<number, string>,
  opts?: {
    classVisible?: (classId: number) => boolean;
    showMulti?: boolean;
    showAbstain?: boolean;
  },
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const img = ctx.createImageData(width, height);
  const out = img.data;
  const rgbCache = new Map<number, [number, number, number]>();
  for (const [cid, hex] of colorByClass) rgbCache.set(cid, parseHexColor(hex));
  const classVisible = opts?.classVisible ?? (() => true);
  const showMulti = opts?.showMulti !== false;
  const showAbstain = opts?.showAbstain !== false;

  for (let i = 0; i < commit.length; i++) {
    const st = status[i];
    const p = i * 4;
    const x = i % width;
    const y = (i / width) | 0;
    if (st === 1) {
      // singleton
      const cid = commit[i];
      if (!classVisible(cid)) continue;
      const rgb = rgbCache.get(cid);
      if (!rgb) continue;
      out[p] = rgb[0];
      out[p + 1] = rgb[1];
      out[p + 2] = rgb[2];
      out[p + 3] = 120;
    } else if (st === 2) {
      if (!showMulti) continue;
      // multi — diagonal hatch in amber
      const on = ((x + y) & 3) < 2;
      out[p] = on ? 245 : 180;
      out[p + 1] = on ? 158 : 120;
      out[p + 2] = on ? 11 : 40;
      out[p + 3] = on ? 140 : 70;
    } else if (showAbstain) {
      // abstain
      out[p] = 30;
      out[p + 1] = 30;
      out[p + 2] = 40;
      out[p + 3] = 90;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Convert a classId label map into polygon shapes.
 * Each pixel contributes to at most one class (labels are already argmax).
 * Uses outer+hole rings so one class does not fill through another (or through
 * preserved scribble gaps) — matching the pixel overlay.
 * Pixels covered by *preserveShapes* are skipped so Commit keeps annotations.
 */
export function labelMapToPolygonShapes(
  labels: Uint8Array,
  width: number,
  height: number,
  classIds: number[],
  {
    minRegion = 64,
    preserveShapes,
    smooth = 1,
  }: { minRegion?: number; preserveShapes?: Shape[]; smooth?: number } = {},
): PolygonShape[] {
  const preserve =
    preserveShapes && preserveShapes.length > 0
      ? rasterizeUnion(preserveShapes, width, height, 1)
      : null;

  const shapes: PolygonShape[] = [];
  for (const classId of classIds) {
    const mask = new Uint8Array(width * height);
    let any = false;
    for (let i = 0; i < labels.length; i++) {
      if (preserve && preserve[i]) continue;
      if (labels[i] === classId) {
        mask[i] = 1;
        any = true;
      }
    }
    if (!any) continue;
    const polys = maskToPolygonsWithHoles(mask, width, height, {
      minRegion,
      scale: 1,
      smooth,
    });
    for (const { points, holes } of polys) {
      if (points.length < 6) continue;
      shapes.push({
        id: uuidv4(),
        classId,
        kind: 'polygon',
        points,
        ...(holes.length ? { holes } : {}),
      });
    }
  }
  return shapes;
}
