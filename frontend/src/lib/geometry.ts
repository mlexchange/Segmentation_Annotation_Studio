/**
 * Coordinate geometry utilities.
 * All shape coordinates are stored in IMAGE pixels.
 * The Stage transform (scaleX/scaleY/x/y) is display-only.
 */

export interface Point {
  x: number;
  y: number;
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
