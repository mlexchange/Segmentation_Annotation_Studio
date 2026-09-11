/**
 * Coordinate geometry utilities.
 * All shape coordinates are stored in IMAGE pixels.
 * The Stage transform (scaleX/scaleY/x/y) is display-only.
 */
import type { Shape } from '@/stores/annotationStore';

export interface Point {
  x: number;
  y: number;
}

/** Axis-aligned bounding box in image pixels. */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** An empty box that intersects nothing (used for degenerate/empty shapes). */
const EMPTY_BBOX: BBox = { x: 0, y: 0, w: -1, h: -1 };

/** Extend `acc` (as [minX,minY,maxX,maxY]) to cover a flat [x,y,…] ring.
 *  Tolerates a missing/short array: shapes restored from an old draft or version
 *  payload are not guaranteed to match the current type exactly, and a throw here
 *  would abort an entire commit. */
function growByFlat(acc: number[], pts: number[] | undefined, pad = 0): void {
  if (!pts) return;
  for (let i = 0; i + 1 < pts.length; i += 2) {
    const x = pts[i], y = pts[i + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x - pad < acc[0]) acc[0] = x - pad;
    if (y - pad < acc[1]) acc[1] = y - pad;
    if (x + pad > acc[2]) acc[2] = x + pad;
    if (y + pad > acc[3]) acc[3] = y + pad;
  }
}

/**
 * Axis-aligned bounds of a shape, in image pixels.
 *
 * Used to skip expensive geometry work (boolean ops, full-resolution
 * rasterization) for shapes that cannot possibly interact — see the callers in
 * `clipToClasses` / `mergeSameClass`. Correctness rests on this being a true
 * SUPERSET of the shape's covered pixels, so it deliberately over-covers:
 *
 * - Brush strokes expand by their radius; `erase` strokes are included even
 *   though they only subtract, since a superset is always safe.
 * - Polygon holes are included for the same reason — they can only clear pixels,
 *   but counting them costs nothing and removes a class of edge case.
 * - Vector `erased` carve-outs likewise cannot extend the shape and are ignored.
 */
export function shapeBBox(shape: Shape): BBox {
  if (shape.kind === 'rectangle') {
    return normalizeRect(shape.x, shape.y, shape.w, shape.h);
  }
  if (shape.kind === 'ellipse') {
    const rx = Math.abs(shape.rx), ry = Math.abs(shape.ry);
    return { x: shape.cx - rx, y: shape.cy - ry, w: rx * 2, h: ry * 2 };
  }

  const acc = [Infinity, Infinity, -Infinity, -Infinity];
  if (shape.kind === 'polygon') {
    growByFlat(acc, shape.points);
    for (const hole of shape.holes ?? []) growByFlat(acc, hole);
  } else {
    // Brush: every stroke's polyline, fattened by its own radius.
    for (const st of shape.strokes ?? []) growByFlat(acc, st?.points, st?.radius ?? 0);
  }
  if (!Number.isFinite(acc[0])) return EMPTY_BBOX;
  return { x: acc[0], y: acc[1], w: acc[2] - acc[0], h: acc[3] - acc[1] };
}

/** Union of several shapes' bounds (empty when the list is empty). */
export function unionBBox(shapes: Shape[]): BBox {
  const acc = [Infinity, Infinity, -Infinity, -Infinity];
  for (const s of shapes) {
    const b = shapeBBox(s);
    if (b.w < 0 || b.h < 0) continue;
    if (b.x < acc[0]) acc[0] = b.x;
    if (b.y < acc[1]) acc[1] = b.y;
    if (b.x + b.w > acc[2]) acc[2] = b.x + b.w;
    if (b.y + b.h > acc[3]) acc[3] = b.y + b.h;
  }
  if (!Number.isFinite(acc[0])) return EMPTY_BBOX;
  return { x: acc[0], y: acc[1], w: acc[2] - acc[0], h: acc[3] - acc[1] };
}

/** True if two AABBs overlap (strict — touching edges do not count). */
export function bboxIntersects(a: BBox, b: BBox): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

/**
 * Overlap test with a tolerance, for deciding whether two shapes can interact.
 *
 * Prefer this over `bboxIntersects` when the answer gates real geometry work: a
 * pair whose bounds merely touch can still produce adjacent set pixels once
 * rasterized, and a shape excluded here is never examined again. `pad` should be
 * at least one grid cell of whatever raster the caller compares on.
 */
export function bboxNear(a: BBox, b: BBox, pad = 1): boolean {
  if (a.w < 0 || a.h < 0 || b.w < 0 || b.h < 0) return false;
  return (
    a.x - pad < b.x + b.w && a.x + a.w + pad > b.x &&
    a.y - pad < b.y + b.h && a.y + a.h + pad > b.y
  );
}

export interface StageTransform {
  scaleX: number;
  scaleY: number;
  x: number;
  y: number;
}

/** Convert a stage (screen) point to image pixels. */
export function toImage(pt: Point, transform: StageTransform): Point {
  return {
    x: (pt.x - transform.x) / transform.scaleX,
    y: (pt.y - transform.y) / transform.scaleY,
  };
}

/** Convert an image-pixel point to stage (screen) coordinates. */
export function toStage(pt: Point, transform: StageTransform): Point {
  return {
    x: pt.x * transform.scaleX + transform.x,
    y: pt.y * transform.scaleY + transform.y,
  };
}

/** Normalize a rectangle so w and h are always >= 0 (fixes negative-drag bug). */
export function normalizeRect(
  x: number,
  y: number,
  w: number,
  h: number
): { x: number; y: number; w: number; h: number } {
  return {
    x: w < 0 ? x + w : x,
    y: h < 0 ? y + h : y,
    w: Math.abs(w),
    h: Math.abs(h),
  };
}

/** Normalize an ellipse so rx and ry are always >= 0. */
export function normalizeEllipse(
  cx: number,
  cy: number,
  rx: number,
  ry: number
): { cx: number; cy: number; rx: number; ry: number } {
  return { cx, cy, rx: Math.abs(rx), ry: Math.abs(ry) };
}

/** Clamp a value between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
