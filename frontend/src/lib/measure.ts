/**
 * Per-region geometric measurements for the Annotate tab. Area / centroid /
 * bounding box come from the shape rasterizer (so they match exported masks);
 * perimeter is computed analytically per shape kind. All values in IMAGE pixels;
 * an optional pixel size converts to physical units.
 *
 * Intensity statistics (min/max/mean/std of raw values inside the region) are
 * computed server-side — see POST /api/measure — because the browser only has
 * the display-rendered image, not raw scientific intensities.
 */
import type { Shape, PolygonShape, RectShape, EllipseShape } from '@/stores/annotationStore';
import { gridFor, rasterizeUnion } from '@/lib/rasterize';

export interface RegionMeasurement {
  /** Number of selected shapes measured. */
  count: number;
  areaPx: number;
  /** Perimeter in px, or null when not defined for the shapes (e.g. brush). */
  perimeterPx: number | null;
  centroid: { x: number; y: number } | null;
  bbox: { x: number; y: number; w: number; h: number } | null;
}

/** Analytic perimeter for one shape (image px), or null if undefined for its kind. */
function shapePerimeter(shape: Shape): number | null {
  if (shape.kind === 'rectangle') {
    const r = shape as RectShape;
    return 2 * (Math.abs(r.w) + Math.abs(r.h));
  }
  if (shape.kind === 'ellipse') {
    const e = shape as EllipseShape;
    const a = Math.abs(e.rx), b = Math.abs(e.ry);
    // Ramanujan approximation.
    return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
  }
  if (shape.kind === 'polygon') {
    const p = shape as PolygonShape;
    let per = ringPerimeter(p.points);
    for (const hole of p.holes ?? []) per += ringPerimeter(hole);
    return per;
  }
  return null; // brush: perimeter not meaningful
}

/** Closed-ring perimeter of a flat [x,y,…] point list. */
function ringPerimeter(pts: number[]): number {
  const n = pts.length / 2;
  if (n < 2) return 0;
  let per = 0;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const dx = pts[i * 2] - pts[j * 2];
    const dy = pts[i * 2 + 1] - pts[j * 2 + 1];
    per += Math.hypot(dx, dy);
  }
  return per;
}

/** Measure area/perimeter/centroid/bbox for the given shapes (union for area/bbox/centroid). */
export function measureRegion(shapes: Shape[], width: number, height: number): RegionMeasurement {
  if (shapes.length === 0) {
    return { count: 0, areaPx: 0, perimeterPx: null, centroid: null, bbox: null };
  }
  const { gw, gh, scale } = gridFor(width, height);
  const px2 = scale * scale;
  const mask = rasterizeUnion(shapes, gw, gh, scale);

  let count = 0, sx = 0, sy = 0;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      if (!mask[y * gw + x]) continue;
      count++;
      sx += x; sy += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  // Sum analytic perimeters; null if any shape lacks one (e.g. brush).
  let perimeterPx: number | null = 0;
  for (const s of shapes) {
    const p = shapePerimeter(s);
    if (p === null) { perimeterPx = null; break; }
    perimeterPx += p;
  }

  const centroid = count > 0 ? { x: (sx / count) * scale, y: (sy / count) * scale } : null;
  const bbox = count > 0
    ? { x: minX * scale, y: minY * scale, w: (maxX - minX + 1) * scale, h: (maxY - minY + 1) * scale }
    : null;

  return { count: shapes.length, areaPx: count * px2, perimeterPx, centroid, bbox };
}
