/**
 * Dataset analytics + QA — pure functions over the annotation store for the
 * Insights view. Computes class balance, slice coverage, and quality flags
 * (slivers, self-intersections, cross-class overlaps, empty-but-not-marked
 * slices) for a single opened sample.
 *
 * Performance: per-instance areas use analytic formulas (shoelace / w·h / π·rx·ry)
 * — no per-shape rasterization. Per-class pixel area + cross-class overlap use one
 * downsampled union mask per class per slice (grid long side ≤ STATS_MAXDIM), and
 * overlap is bbox-guarded so disjoint classes are skipped without scanning.
 */
import type { Shape, PolygonShape, RectShape, EllipseShape, BrushShape } from '@/stores/annotationStore';
import type { AnnotationClass } from '@/stores/classStore';
import { gridFor, rasterizeUnion } from '@/lib/rasterize';

export interface ClassStat {
  classId: number;
  label: string;
  color: string;
  shapeCount: number;
  /** Total labeled area in image pixels² (approximate; from the rasterized grid). */
  pixelArea: number;
}

export interface Coverage {
  totalSlices: number;
  annotatedSlices: number;
  negativeSlices: number;
  /** Slice indices with no shapes and not marked negative — ambiguous for training. */
  emptyUnmarked: number[];
}

export type QAFlagKind = 'sliver' | 'self-intersection' | 'overlap' | 'empty-unmarked';

export interface QAFlag {
  kind: QAFlagKind;
  message: string;
  /** Slice to jump to when the flag is clicked. */
  slice: number;
  classId?: number;
  /** Image-coord region to zoom to + highlight (absent for whole-slice flags). */
  bbox?: { x: number; y: number; w: number; h: number };
}

export interface SampleStats {
  classStats: ClassStat[];
  coverage: Coverage;
  flags: QAFlag[];
}

/** Slivers below this image-pixel² area are flagged as likely accidental. */
const SLIVER_AREA_PX = 6;
/** Grid long-side cap for area/overlap rasterization (accuracy vs. speed). */
const STATS_MAXDIM = 512;

type SlicesMap = Record<string, Shape[]>;

// --- analytic per-instance area (image px²), no rasterization ---------------

function shoelace(pts: number[]): number {
  const n = pts.length / 2;
  if (n < 3) return 0;
  let s = 0;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    s += pts[j * 2] * pts[i * 2 + 1] - pts[i * 2] * pts[j * 2 + 1];
  }
  return Math.abs(s) / 2;
}

function analyticArea(shape: Shape): number {
  switch (shape.kind) {
    case 'rectangle': {
      const r = shape as RectShape;
      return Math.abs(r.w * r.h);
    }
    case 'ellipse': {
      const e = shape as EllipseShape;
      return Math.PI * Math.abs(e.rx * e.ry);
    }
    case 'polygon': {
      const p = shape as PolygonShape;
      let a = shoelace(p.points);
      for (const hole of p.holes ?? []) a -= shoelace(hole);
      return Math.max(0, a);
    }
    case 'brush': {
      // Round-capped strokes: length·(2r) for the body + π r² for the caps.
      const b = shape as BrushShape;
      let a = 0;
      for (const st of b.strokes) {
        if (st.mode !== 'paint') continue;
        const pts = st.points;
        let len = 0;
        for (let i = 0; i + 3 < pts.length; i += 2) {
          len += Math.hypot(pts[i + 2] - pts[i], pts[i + 3] - pts[i + 1]);
        }
        a += len * 2 * st.radius + Math.PI * st.radius * st.radius;
      }
      return a;
    }
    default:
      return 0;
  }
}

// --- shape bounding box (image coords) --------------------------------------

function ringBounds(pts: number[], acc: { minX: number; minY: number; maxX: number; maxY: number }): void {
  for (let i = 0; i + 1 < pts.length; i += 2) {
    const x = pts[i], y = pts[i + 1];
    if (x < acc.minX) acc.minX = x;
    if (x > acc.maxX) acc.maxX = x;
    if (y < acc.minY) acc.minY = y;
    if (y > acc.maxY) acc.maxY = y;
  }
}

/** Axis-aligned bounding box of a shape in image pixels, or null if empty. */
function shapeBBox(shape: Shape): { x: number; y: number; w: number; h: number } | null {
  const acc = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  if (shape.kind === 'rectangle') {
    const r = shape as RectShape;
    const x = Math.min(r.x, r.x + r.w), y = Math.min(r.y, r.y + r.h);
    return { x, y, w: Math.abs(r.w), h: Math.abs(r.h) };
  }
  if (shape.kind === 'ellipse') {
    const e = shape as EllipseShape;
    return { x: e.cx - Math.abs(e.rx), y: e.cy - Math.abs(e.ry), w: 2 * Math.abs(e.rx), h: 2 * Math.abs(e.ry) };
  }
  if (shape.kind === 'polygon') {
    ringBounds((shape as PolygonShape).points, acc);
  } else if (shape.kind === 'brush') {
    for (const st of (shape as BrushShape).strokes) {
      // Expand by radius to cover the stroke width.
      const r = st.radius;
      for (let i = 0; i + 1 < st.points.length; i += 2) {
        const x = st.points[i], y = st.points[i + 1];
        if (x - r < acc.minX) acc.minX = x - r;
        if (x + r > acc.maxX) acc.maxX = x + r;
        if (y - r < acc.minY) acc.minY = y - r;
        if (y + r > acc.maxY) acc.maxY = y + r;
      }
    }
  }
  if (!Number.isFinite(acc.minX)) return null;
  return { x: acc.minX, y: acc.minY, w: acc.maxX - acc.minX, h: acc.maxY - acc.minY };
}

// --- polygon self-intersection ---------------------------------------------

function segmentsIntersect(
  ax: number, ay: number, bx: number, by: number,
  cx: number, cy: number, dx: number, dy: number,
): boolean {
  const d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
  const d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
  const d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
  return ((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0));
}

/** True if the flat [x,y,…] ring has crossing non-adjacent edges. Capped for cost. */
function isSelfIntersecting(pts: number[]): boolean {
  const n = pts.length / 2;
  if (n < 4 || n > 400) return false; // skip pathological point counts
  for (let i = 0; i < n; i++) {
    const a = i, b = (i + 1) % n;
    for (let j = i + 1; j < n; j++) {
      const c = j, d = (j + 1) % n;
      if (a === c || a === d || b === c || b === d) continue;
      if (segmentsIntersect(
        pts[a * 2], pts[a * 2 + 1], pts[b * 2], pts[b * 2 + 1],
        pts[c * 2], pts[c * 2 + 1], pts[d * 2], pts[d * 2 + 1],
      )) return true;
    }
  }
  return false;
}

// --- mask helpers -----------------------------------------------------------

interface MaskInfo {
  classId: number;
  mask: Uint8Array;
  count: number;
  minX: number; minY: number; maxX: number; maxY: number;
}

/** Bit count + bbox of a grid mask in one pass. */
function maskInfo(classId: number, mask: Uint8Array, gw: number, gh: number): MaskInfo {
  let count = 0, minX = gw, minY = gh, maxX = -1, maxY = -1;
  for (let y = 0; y < gh; y++) {
    const row = y * gw;
    for (let x = 0; x < gw; x++) {
      if (!mask[row + x]) continue;
      count++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return { classId, mask, count, minX, minY, maxX, maxY };
}

/**
 * Overlapping-pixel count + grid bbox of two masks, scanning only their bbox
 * intersection. Returns count 0 (and a degenerate bbox) when disjoint.
 */
function overlapCount(a: MaskInfo, b: MaskInfo, gw: number): { count: number; minX: number; minY: number; maxX: number; maxY: number } {
  const x0 = Math.max(a.minX, b.minX), x1 = Math.min(a.maxX, b.maxX);
  const y0 = Math.max(a.minY, b.minY), y1 = Math.min(a.maxY, b.maxY);
  let n = 0, oMinX = gw, oMinY = gw, oMaxX = -1, oMaxY = -1;
  if (x0 <= x1 && y0 <= y1) {
    for (let y = y0; y <= y1; y++) {
      const row = y * gw;
      for (let x = x0; x <= x1; x++) {
        if (a.mask[row + x] && b.mask[row + x]) {
          n++;
          if (x < oMinX) oMinX = x;
          if (x > oMaxX) oMaxX = x;
          if (y < oMinY) oMinY = y;
          if (y > oMaxY) oMaxY = y;
        }
      }
    }
  }
  return { count: n, minX: oMinX, minY: oMinY, maxX: oMaxX, maxY: oMaxY };
}

/**
 * Compute class balance, coverage, and QA flags for one sample.
 *
 * @param slicesForSource `byImage[sourceKey]` — slice key → shapes.
 * @param classes         current class list.
 * @param nSlices         total slices in the volume (for coverage / empty slices).
 * @param negative        slice keys marked as negative examples.
 * @param width,height    image dimensions in pixels.
 */
export function computeSampleStats(
  slicesForSource: SlicesMap,
  classes: AnnotationClass[],
  nSlices: number,
  negative: string[],
  width: number,
  height: number,
): SampleStats {
  const { gw, gh, scale } = gridFor(width, height, STATS_MAXDIM);
  const px2 = scale * scale;
  const negSet = new Set(negative.map((k) => String(k)));

  const statById = new Map<number, ClassStat>(
    classes.map((c) => [c.classId, { classId: c.classId, label: c.label, color: c.color, shapeCount: 0, pixelArea: 0 }]),
  );
  const flags: QAFlag[] = [];
  let annotatedSlices = 0;

  for (const [sliceKey, shapes] of Object.entries(slicesForSource)) {
    if (!Array.isArray(shapes) || shapes.length === 0) continue;
    annotatedSlices++;
    const sliceIdx = Number(sliceKey);

    const byClass = new Map<number, Shape[]>();
    for (const sh of shapes) {
      if (!byClass.has(sh.classId)) byClass.set(sh.classId, []);
      byClass.get(sh.classId)!.push(sh);

      const stat = statById.get(sh.classId);
      if (stat) stat.shapeCount++;

      const area = analyticArea(sh);
      if (area > 0 && area < SLIVER_AREA_PX) {
        flags.push({ kind: 'sliver', slice: sliceIdx, classId: sh.classId,
          bbox: shapeBBox(sh) ?? undefined,
          message: `Tiny ${labelFor(classes, sh.classId)} region (~${area.toFixed(1)} px²) on slice ${sliceIdx}` });
      }
      if (sh.kind === 'polygon' && isSelfIntersecting((sh as PolygonShape).points)) {
        flags.push({ kind: 'self-intersection', slice: sliceIdx, classId: sh.classId,
          bbox: shapeBBox(sh) ?? undefined,
          message: `Self-intersecting ${labelFor(classes, sh.classId)} polygon on slice ${sliceIdx}` });
      }
    }

    // Per-class union mask (for accurate class area) + pairwise overlap.
    const infos: MaskInfo[] = [];
    for (const [classId, clsShapes] of byClass) {
      const info = maskInfo(classId, rasterizeUnion(clsShapes, gw, gh, scale), gw, gh);
      const stat = statById.get(classId);
      if (stat) stat.pixelArea += info.count * px2;
      infos.push(info);
    }
    for (let i = 0; i < infos.length; i++) {
      for (let j = i + 1; j < infos.length; j++) {
        const ov = overlapCount(infos[i], infos[j], gw);
        if (ov.count > 0) {
          // Grid bbox → image coords.
          const bbox = {
            x: ov.minX * scale, y: ov.minY * scale,
            w: (ov.maxX - ov.minX + 1) * scale, h: (ov.maxY - ov.minY + 1) * scale,
          };
          flags.push({ kind: 'overlap', slice: sliceIdx, classId: infos[i].classId, bbox,
            message: `${labelFor(classes, infos[i].classId)} and ${labelFor(classes, infos[j].classId)} overlap (~${(ov.count * px2).toFixed(0)} px²) on slice ${sliceIdx}` });
        }
      }
    }
  }

  // Empty-but-not-marked slices (ambiguous for training).
  const emptyUnmarked: number[] = [];
  for (let s = 0; s < nSlices; s++) {
    const key = String(s);
    const shapes = slicesForSource[key];
    const hasShapes = Array.isArray(shapes) && shapes.length > 0;
    if (!hasShapes && !negSet.has(key)) emptyUnmarked.push(s);
  }
  if (emptyUnmarked.length > 0 && emptyUnmarked.length <= 20) {
    for (const s of emptyUnmarked) {
      flags.push({ kind: 'empty-unmarked', slice: s,
        message: `Slice ${s} has no annotations and isn't marked as a negative example` });
    }
  }

  return {
    classStats: [...statById.values()],
    coverage: { totalSlices: nSlices, annotatedSlices, negativeSlices: negSet.size, emptyUnmarked },
    flags,
  };
}

function labelFor(classes: AnnotationClass[], classId: number): string {
  return classes.find((c) => c.classId === classId)?.label ?? `class ${classId}`;
}
