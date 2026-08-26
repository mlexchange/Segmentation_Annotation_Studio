/**
 * AnnotationCanvas — react-konva Stage with image + shape layers.
 *
 * Layer 0: image (Konva filters: Brighten/Contrast)
 * Layer 1: committed shapes (listening=false, cached for opacity compositing)
 * Layer 2: draft polygon + drag preview
 * Layer 3: in-progress brush stroke (imperative, zero React re-renders per move)
 * Layer 4: brush/eraser size cursor preview (imperative position update)
 *
 * Lag fix: brush/eraser strokes are buffered in a ref during mousemove and
 * committed to the Zustand store ONCE on mouseup, eliminating the per-frame
 * store updates + zundo history snapshots that caused cursor lag.
 */
import { useRef, useState, useEffect, useLayoutEffect, useCallback, useMemo, useId } from 'react';
import {
  Stage, Layer, Image as KonvaImage, Line, Rect, Ellipse, Group, Circle, Transformer,
  Shape as KonvaShape,
} from 'react-konva';
import { Trash } from '@phosphor-icons/react';
import type Konva from 'konva';
import { v4 as uuidv4 } from 'uuid';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore, type Shape, type PolygonShape, type BrushStroke } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { toImage, normalizeRect, normalizeEllipse, shapeBBox, unionBBox, bboxNear, bboxIntersects, type BBox } from '@/lib/geometry';
import { buildCostMap, dijkstra, tracePath, imageToGrid, simplifyPath, type CostMap } from '@/lib/livewire';
import { useImageSlice } from '@/hooks/useImageSlice';
import { buildSourceKey } from '@/lib/sourceKey';
import { buildField, magicSelect, maskToPolygons, maskToPolygonsWithHoles, type GrayField } from '@/lib/magicwand';
import { useSam } from '@/hooks/useSam';
import { renderAdjusted, renderPreprocessOnly } from '@/lib/sam/adjust';
import { gridFor, fullResGridFor, rasterizeShapes, rasterizeUnion, stampStroke } from '@/lib/rasterize';
import { keepComponentsAtPoints, dilate, erode, removeSmallComponents } from '@/lib/morphology';
import { unionShapesToPolygons, unionShapesToMultiPolygon, unionShapesChecked, eraseStampToMultiPolygon, subtractFromShape, regionsToMultiPolygon } from '@/lib/polybool';
import { computeRegionOps, type RegionOp } from '@/lib/regionOps';
import { clipShapesToOthers, clipShapesToOthersMask, hasOtherClass } from '@/lib/clipToClasses';
import { mergeNewWithSameClass, expandSameClassOverlap } from '@/lib/mergeSameClass';
import { useClipboardStore } from '@/stores/clipboardStore';
import { colormapTables, type ColormapName } from '@/lib/colormaps';
import ShapesLayer from './ShapesLayer';
import { displayAffineFor, displayBandToBase, baseToDisplay, displayToBase } from '@/lib/displayTransform';
import {
  backgroundRing, fitWithBlurSweep, combineSigma, BLUR_CANDIDATES, MAX_RING_WIDTH,
  fitBand, sampleHistograms, type SweepResult,
} from '@/lib/thresholdFit';
import {
  buildChannels, fitProjection, projectToScore, CHANNEL_NAMES,
} from '@/lib/featureChannels';
import { applyGaussianBlurGray } from '@/lib/blur';
import { time } from '@/lib/perf';

// macOS labels the Alt key "Option" (⌥). e.altKey is true for it either way,
// so only the on-screen label needs to differ.
const IS_MAC = typeof navigator !== 'undefined' && /mac/i.test(navigator.userAgent);
const REMOVE_KEY_LABEL = IS_MAC ? 'Option' : 'Alt';

/** Shared stable empty shape list — see `storeShapes` for why identity matters. */
const EMPTY_SHAPES: Shape[] = [];

/** Outcome of a Sampler lasso: the fitted band plus what it took to get there. */
export interface SamplerFit extends SweepResult {
  /** The band converted into the DISPLAYED space the threshold knobs use. */
  displayLo: number;
  displayHi: number;
  /** True when the display transform squashes the fitted range to one value, so
   *  the band cannot be expressed at the current brightness/contrast/levels. */
  collapsed: boolean;
  /** Band and blur in force before the fit, so the UI can offer Revert. */
  previousBand: [number, number];
  previousBlur: number;
  /** Blur the fit settled on (current combined with the swept extra). */
  appliedBlur: number;
  /** Which signal the brush now gates on: raw displayed intensity, or a fitted
   *  combination of intensity and texture channels. */
  mode?: 'intensity' | 'projected';
  /** Skill intensity alone managed, so the gain from projecting is visible. */
  intensitySkill?: number;
  /** Contribution of each channel to the projection, for the readout. */
  weights?: Array<{ name: string; weight: number }>;
}

/** How far (image px) the cursor travels before the sampler lasso drops a new
 *  live-wire anchor. Larger = fewer dijkstra runs but looser snapping. */
const SAMPLER_ANCHOR_STEP = 40;

/** Smallest island (grid cells) a threshold stroke keeps after regularization.
 *  Below this a region is thresholding noise, not a feature. */
const THRESHOLD_MIN_REGION = 12;

/** Upscale budget: the working resolution is clamped back to 1x past this many
 *  pixels so a large slice at 4x can't exhaust memory (~64 MP ≈ 256 MB of RGBA). */
const MAX_WORKING_PIXELS = 64e6;

interface AnnotationCanvasProps {
  brightness: number;
  contrast: number;
  /** Client-side levels window (0–255) applied to the displayed image. */
  levelsLo: number;
  levelsHi: number;
  /** Display-only false-color map + gamma (do not affect exported pixels or tools). */
  colormap?: ColormapName;
  gamma?: number;
  /** Display-only nonlinear preprocessors baked into the image base (affect the
   *  tools' baked view but NOT the exported pixels). */
  clahe?: boolean;
  sharpen?: boolean;
  /** Gaussian pre-blur sigma in image pixels (0 = off). */
  blur?: number;
  /** Working-resolution multiplier (1, 2, 4): resamples the slice for the display
   *  base and every tool field so small features get more pixels. Annotation
   *  coordinates stay in NATIVE image pixels — they just gain sub-pixel precision. */
  upscale?: number;
  /** Emits the current slice's 256-bin luminance histogram when it loads. */
  onHistogram?: (bins: number[]) => void;
  /** Result of a Sampler lasso fit (null when it could not fit), for the Toolbar. */
  onSamplerFit?: (fit: SamplerFit | null) => void;
  /** Lets the Sampler apply the blur sigma it chose. */
  onBlurChange?: (sigma: number) => void;
  activeClassId: number | null;
  activeBrushShapeId: string | null;
  onNewBrushInstance: (id: string) => void;
  /** When non-null, the canvas renders these shapes read-only (version preview). */
  previewShapes?: Shape[] | null;
  /** Class definitions used for preview shape colors (the version's own classes). */
  previewClasses?: AnnotationClass[] | null;
  /** Zoom to + highlight this image-coord region (e.g. from an Insights QA flag).
   *  `nonce` changes to re-trigger the same region. */
  focusRegion?: { x: number; y: number; w: number; h: number; nonce: number } | null;
}

// ---- Point-in-shape hit testing (used by the eraser to pick a target) ----

/** Build an even-odd path (outer + hole rings) on a Konva context. */
function buildRingsPath(ctx: Konva.Context, rings: number[][]): void {
  ctx.beginPath();
  for (const r of rings) {
    if (r.length < 6) continue;
    ctx.moveTo(r[0], r[1]);
    for (let i = 2; i < r.length; i += 2) ctx.lineTo(r[i], r[i + 1]);
    ctx.closePath();
  }
}

/** Translate a shape by (dx,dy) in image coords (used for paste offset). */
function offsetShape(shape: Shape, dx: number, dy: number): Shape {
  const shiftFlat = (pts: number[]) => pts.map((v, i) => (i % 2 === 0 ? v + dx : v + dy));
  if (shape.kind === 'polygon') {
    return { ...shape, points: shiftFlat(shape.points), holes: shape.holes?.map(shiftFlat) };
  }
  if (shape.kind === 'rectangle') return { ...shape, x: shape.x + dx, y: shape.y + dy };
  if (shape.kind === 'ellipse') return { ...shape, cx: shape.cx + dx, cy: shape.cy + dy };
  // brush
  return { ...shape, strokes: shape.strokes.map((st) => ({ ...st, points: shiftFlat(st.points) })) };
}

/** Even-odd ray cast: true if point (px,py) is inside the flat [x,y,…] polygon. */
function pointInPolygon(px: number, py: number, pts: number[]): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 2; i < pts.length; j = i, i += 2) {
    const xi = pts[i], yi = pts[i + 1];
    const xj = pts[j], yj = pts[j + 1];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** hex (#rgb or #rrggbb) → an rgba() string with the given alpha. Falls back to
 *  the input untouched if it isn't a hex color. */
function withAlpha(hex: string, a: number): string {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return hex;
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** hex (#rgb or #rrggbb) → [r,g,b]. Falls back to mid-grey if it isn't a hex color. */
function hexToRgb(hex: string): [number, number, number] {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return [128, 128, 128];
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Shortest distance from point (px,py) to segment AB. */
function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

/** True if image-coord point (x,y) lies inside any shape kind; brush strokes
 *  count as filled within `radius` of their segments (erase strokes ignored). */
function shapeContainsPoint(shape: Shape, x: number, y: number): boolean {
  if (shape.kind === 'polygon') {
    if (!pointInPolygon(x, y, shape.points)) return false;
    // A point inside a hole is NOT inside the polygon (so an encompassed shape in
    // the hole stays selectable and this polygon doesn't claim the click).
    for (const hole of shape.holes ?? []) if (pointInPolygon(x, y, hole)) return false;
    return true;
  }
  if (shape.kind === 'rectangle') {
    return x >= shape.x && x <= shape.x + shape.w && y >= shape.y && y <= shape.y + shape.h;
  }
  if (shape.kind === 'ellipse') {
    const nx = (x - shape.cx) / (shape.rx || 1);
    const ny = (y - shape.cy) / (shape.ry || 1);
    return nx * nx + ny * ny <= 1;
  }
  if (shape.kind === 'brush') {
    for (const st of shape.strokes) {
      if (st.mode === 'erase') continue;
      const p = st.points;
      for (let i = 0; i + 3 < p.length; i += 2) {
        if (distToSegment(x, y, p[i], p[i + 1], p[i + 2], p[i + 3]) <= st.radius) return true;
      }
      if (p.length >= 2 && distToSegment(x, y, p[0], p[1], p[0], p[1]) <= st.radius) return true;
    }
  }
  return false;
}

/** True if (x,y) is within `r` of the shape (inside it, or within `r` of its
 *  boundary). Used for eraser targeting so a brush that grazes a shape's edge —
 *  its disk overlaps the shape even though the center is outside — still erases. */
function shapeNearPoint(shape: Shape, x: number, y: number, r: number): boolean {
  if (r <= 0) return shapeContainsPoint(shape, x, y);
  if (shapeContainsPoint(shape, x, y)) return true;
  if (shape.kind === 'polygon') {
    const rings = [shape.points, ...(shape.holes ?? [])];
    for (const ring of rings) {
      const n = ring.length;
      for (let i = 0; i + 1 < n; i += 2) {
        const bx = ring[(i + 2) % n], by = ring[(i + 3) % n];
        if (distToSegment(x, y, ring[i], ring[i + 1], bx, by) <= r) return true;
      }
    }
    return false;
  }
  if (shape.kind === 'rectangle') {
    const cx = Math.max(shape.x, Math.min(x, shape.x + shape.w));
    const cy = Math.max(shape.y, Math.min(y, shape.y + shape.h));
    return Math.hypot(x - cx, y - cy) <= r;
  }
  if (shape.kind === 'ellipse') {
    // Approximate: test against the ellipse grown by `r` on each axis.
    const nx = (x - shape.cx) / ((shape.rx || 1) + r);
    const ny = (y - shape.cy) / ((shape.ry || 1) + r);
    return nx * nx + ny * ny <= 1;
  }
  if (shape.kind === 'brush') {
    for (const st of shape.strokes) {
      if (st.mode === 'erase') continue;
      const p = st.points;
      for (let i = 0; i + 3 < p.length; i += 2) {
        if (distToSegment(x, y, p[i], p[i + 1], p[i + 2], p[i + 3]) <= st.radius + r) return true;
      }
      if (p.length >= 2 && distToSegment(x, y, p[0], p[1], p[0], p[1]) <= st.radius + r) return true;
    }
  }
  return false;
}

/** True if a new brush stroke (points + radius) touches any paint stroke already in
 *  `strokes` — i.e. their thick outlines overlap. Used to decide whether a stroke
 *  extends the current brush region or starts a new (disconnected) one. */
function strokeTouchesBrush(points: number[], radius: number, strokes: BrushStroke[]): boolean {
  for (const st of strokes) {
    if (st.mode === 'erase') continue;
    const p = st.points;
    const tol = st.radius + radius;
    for (let i = 0; i + 1 < points.length; i += 2) {
      const x = points[i], y = points[i + 1];
      for (let j = 0; j + 3 < p.length; j += 2) {
        if (distToSegment(x, y, p[j], p[j + 1], p[j + 2], p[j + 3]) <= tol) return true;
      }
      if (p.length >= 2 && distToSegment(x, y, p[0], p[1], p[0], p[1]) <= tol) return true;
    }
  }
  return false;
}

/** Insert a vertex on the polygon edge (outer ring or a hole) nearest to (px,py),
 *  at the projected point on that segment. Used for double-click "add node". */
function insertVertexNearest(shape: PolygonShape, px: number, py: number): PolygonShape {
  let best = { d: Infinity, ring: -1, insertVi: 0, x: 0, y: 0 }; // ring -1 = outer
  const scan = (pts: number[], ring: number) => {
    const n = pts.length / 2;
    if (n < 2) return;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const ax = pts[i * 2], ay = pts[i * 2 + 1], bx = pts[j * 2], by = pts[j * 2 + 1];
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
      let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
      t = Math.max(0, Math.min(1, t));
      const cx = ax + t * dx, cy = ay + t * dy;
      const d = Math.hypot(px - cx, py - cy);
      // Insert before vertex j; when j wraps to 0, append at the ring's end.
      if (d < best.d) best = { d, ring, insertVi: j === 0 ? n : j, x: cx, y: cy };
    }
  };
  scan(shape.points, -1);
  (shape.holes ?? []).forEach((h, hi) => scan(h, hi));
  if (best.ring === -1) {
    const np = shape.points.slice();
    np.splice(best.insertVi * 2, 0, best.x, best.y);
    return { ...shape, points: np };
  }
  const hc = (shape.holes ?? []).map((r) => r.slice());
  hc[best.ring].splice(best.insertVi * 2, 0, best.x, best.y);
  return { ...shape, holes: hc };
}

/** A representative point that lies inside *shape* (best-effort), in image
 *  coords — used to seed SAM with negative ("not") prompts for other-class
 *  regions. Returns null when no interior point can be derived. */
function interiorPoint(shape: Shape): { x: number; y: number } | null {
  if (shape.kind === 'rectangle') return { x: shape.x + shape.w / 2, y: shape.y + shape.h / 2 };
  if (shape.kind === 'ellipse') return { x: shape.cx, y: shape.cy };
  if (shape.kind === 'polygon') {
    const p = shape.points;
    if (p.length < 6) return null;
    let cx = 0, cy = 0;
    const n = p.length / 2;
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let i = 0; i < p.length; i += 2) {
      cx += p[i]; cy += p[i + 1];
      minx = Math.min(minx, p[i]); maxx = Math.max(maxx, p[i]);
      miny = Math.min(miny, p[i + 1]); maxy = Math.max(maxy, p[i + 1]);
    }
    cx /= n; cy /= n;
    if (pointInPolygon(cx, cy, p)) return { x: cx, y: cy };
    // Concave polygon: centroid may fall outside — scan a grid for an inside point.
    const steps = 8;
    for (let gy = 1; gy < steps; gy++) {
      for (let gx = 1; gx < steps; gx++) {
        const x = minx + ((maxx - minx) * gx) / steps;
        const y = miny + ((maxy - miny) * gy) / steps;
        if (pointInPolygon(x, y, p)) return { x, y };
      }
    }
    return { x: cx, y: cy };
  }
  if (shape.kind === 'brush') {
    for (const st of shape.strokes) {
      if (st.mode === 'erase') continue;
      const pp = st.points;
      if (pp.length >= 2) {
        const mid = Math.min(pp.length - 2, Math.floor(pp.length / 4) * 2);
        return { x: pp[mid], y: pp[mid + 1] };
      }
    }
    return null;
  }
  return null;
}

/** True if segments AB and CD intersect. */
function segIntersects(
  ax: number, ay: number, bx: number, by: number,
  cx: number, cy: number, dx: number, dy: number,
): boolean {
  const d = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if (d === 0) return false; // parallel/collinear — endpoint cases caught elsewhere
  const t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / d;
  const u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / d;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

/**
 * True if a shape's actual geometry overlaps the marquee rect — NOT just its
 * bounding box. A bbox test wrongly selects a concave/edge-hugging shape (e.g. a
 * C-shape or perimeter stroke) when the marquee is drawn in its empty middle.
 */
function shapeIntersectsRect(shape: Shape, r: BBox): boolean {
  const corners: Array<[number, number]> = [
    [r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h],
  ];
  const inRect = (x: number, y: number) =>
    x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;

  if (shape.kind === 'rectangle') {
    return bboxIntersects({ x: shape.x, y: shape.y, w: shape.w, h: shape.h }, r);
  }

  if (shape.kind === 'ellipse') {
    const rx = shape.rx || 1, ry = shape.ry || 1;
    // Closest point on the rect to the ellipse center, normalised into the unit
    // circle — handles overlap, rect-inside-ellipse, and center-in-rect.
    const clx = Math.max(r.x, Math.min(shape.cx, r.x + r.w));
    const cly = Math.max(r.y, Math.min(shape.cy, r.y + r.h));
    if (((clx - shape.cx) / rx) ** 2 + ((cly - shape.cy) / ry) ** 2 <= 1) return true;
    for (const [x, y] of corners) {
      if (((x - shape.cx) / rx) ** 2 + ((y - shape.cy) / ry) ** 2 <= 1) return true;
    }
    return false;
  }

  if (shape.kind === 'polygon') {
    const p = shape.points;
    if (p.length < 6) return false;
    // Test the outer ring AND any holes: a ring vertex inside the rect, or a ring
    // edge crossing a rect edge, means the marquee touches the shape boundary.
    const rings = [p, ...(shape.holes ?? [])];
    for (const ring of rings) {
      for (let i = 0; i < ring.length; i += 2) if (inRect(ring[i], ring[i + 1])) return true;
      for (let i = 0; i < ring.length; i += 2) {
        const ax = ring[i], ay = ring[i + 1];
        const bx = ring[(i + 2) % ring.length], by = ring[(i + 3) % ring.length];
        for (let k = 0; k < 4; k++) {
          const [c1x, c1y] = corners[k];
          const [c2x, c2y] = corners[(k + 1) % 4];
          if (segIntersects(ax, ay, bx, by, c1x, c1y, c2x, c2y)) return true;
        }
      }
    }
    // A rect corner inside the SOLID part only (inside the outer ring, outside every
    // hole) — so a marquee that lies entirely within a hole does NOT select this
    // shape (e.g. selecting an inner shape sitting in the hole of an outer one).
    for (const [x, y] of corners) if (shapeContainsPoint(shape, x, y)) return true;
    return false;
  }

  if (shape.kind === 'brush') {
    for (const st of shape.strokes) {
      if (st.mode === 'erase') continue;
      const pts = st.points;
      for (let i = 0; i + 1 < pts.length; i += 2) if (inRect(pts[i], pts[i + 1])) return true;
      // A stroke has width: a rect corner within `radius` of a segment counts.
      for (let i = 0; i + 3 < pts.length; i += 2) {
        for (const [x, y] of corners) {
          if (distToSegment(x, y, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]) <= st.radius) return true;
        }
      }
    }
    return false;
  }

  return false;
}

/** The annotation viewport: image + shape layers and all drawing/editing tools.
 *  Shapes are stored in IMAGE pixel coords; `transform` is display-only (pan/zoom). */
export default function AnnotationCanvas({
  brightness,
  contrast,
  levelsLo,
  levelsHi,
  colormap = 'gray',
  gamma = 1,
  clahe = false,
  sharpen = false,
  blur = 0,
  upscale = 1,
  onHistogram,
  onSamplerFit,
  onBlurChange,
  activeClassId,
  activeBrushShapeId,
  onNewBrushInstance,
  previewShapes = null,
  previewClasses = null,
  focusRegion = null,
}: AnnotationCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<Konva.Image>(null);
  const imageLayerRef = useRef<Konva.Layer>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const shapesLayerRef = useRef<Konva.Layer>(null);
  // Resize handles for the selected rect/ellipse (select tool).
  const transformerRef = useRef<Konva.Transformer>(null);

  // Imperative refs for lag-free brush drawing
  const draftStrokeLayerRef = useRef<Konva.Layer>(null);
  const draftLineRef = useRef<Konva.Line>(null);
  const thresholdPreviewRef = useRef<Konva.Image>(null);
  const brushCursorLayerRef = useRef<Konva.Layer>(null);
  const brushCursorRef = useRef<Konva.Circle>(null);

  // Buffered in-progress stroke — populated on mousedown, flushed on mouseup.
  // Never stored in React state so mousemove causes zero re-renders.
  const draftStrokeRef = useRef<{
    shapeId: string;
    mode: 'paint' | 'erase';
    points: number[];
    radius: number;
    /** For erase: whether the target is a brush shape or a vector shape. */
    eraseTargetKind?: 'brush' | 'vector';
  } | null>(null);

  // Threshold-brush stroke, buffered exactly like `draftStrokeRef`: a binary mask
  // over the threshold field's grid that accumulates only in-band pixels, plus the
  // offscreen canvas mirroring it for the live preview. Flushed on mouseup.
  const thresholdStrokeRef = useRef<{
    mode: 'paint' | 'erase';
    /** Grid geometry of the field this stroke was started against. */
    gw: number; gh: number; scale: number;
    mask: Uint8Array;
    /** In-band gate for the whole slice (same grid) — the stroke can't leave it. */
    gate: Uint8Array;
    /** Last pointer position in image coords, for segment stamping. */
    last: { x: number; y: number };
    canvas: HTMLCanvasElement;
    imageData: ImageData;
  } | null>(null);

  // Sampler lasso: a loop drawn to teach the Threshold Brush what to select.
  // Buffered in a ref like the brush so dragging causes no re-renders, and purely
  // a measurement gesture — it commits no shape and touches no history.
  //
  // It snaps to edges using the same live-wire machinery as the magnetic tool:
  // an anchor is dropped every `SAMPLER_ANCHOR_STEP` image pixels and the segment
  // between anchors is the least-cost path, so the sample follows the feature's
  // real boundary instead of a shaky hand-drawn one. A tighter sample means a
  // cleaner positive histogram, which is what the whole fit rests on.
  const samplerLassoRef = useRef<{
    /** Snapped points locked in so far (flat [x,y,…] image coords). */
    committed: number[];
    /** Where the current live-wire segment starts. */
    anchor: { x: number; y: number };
    /** First point, so the loop can be closed back to it. */
    start: { x: number; y: number };
    cm: CostMap | null;
    /** Dijkstra predecessor map from `anchor`; null when snapping is unavailable. */
    prev: Int32Array | null;
  } | null>(null);

  const { kind, source, serverUri, meta, currentSlice, renderOpts, denoise } = useDatasetStore();
  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;
  const { byImage, addShape, addShapes, appendBrushStroke, updateShape, removeShapes, setShapes, setClassForShapes } = useAnnotationStore();
  // In-progress polygon/lasso draft lives in the store so it rides the undo/redo timeline
  // (each node = one step; the close reopens on undo). Previews stay local (below).
  const draft = useAnnotationStore((s) => s.draft);
  const addPolyNode = useAnnotationStore((s) => s.addPolyNode);
  const addMagneticNode = useAnnotationStore((s) => s.addMagneticNode);
  const clearDraft = useAnnotationStore((s) => s.clearDraft);
  const commitDraftShapes = useAnnotationStore((s) => s.commitDraftShapes);
  const clipboard = useClipboardStore();
  const { tool, brushSize, fillOpacity, eraseAllClasses, selectScope, panReturnTool, selectedShapeIds, setSelectedShapeId, setSelectedShapeIds } = useToolStore();
  // While hold-Space panning, `tool` is 'pan' but we still want brush/eraser UI
  // (cursor circle) to reflect the tool we'll return to.
  const underlyingTool = tool === 'pan' && panReturnTool ? panReturnTool : tool;
  // Single-selection id — drives move/resize/vertex editing (those need exactly one).
  const selectedId = selectedShapeIds.length === 1 ? selectedShapeIds[0] : null;
  const magicTolerance = useToolStore((s) => s.magicTolerance);
  const magicMode = useToolStore((s) => s.magicMode);
  const magicSigma = useToolStore((s) => s.magicSigma);
  const magicEdgeStop = useToolStore((s) => s.magicEdgeStop);
  const magicEngine = useToolStore((s) => s.magicEngine);
  const setMagicEngine = useToolStore((s) => s.setMagicEngine);
  const setTool = useToolStore((s) => s.setTool);
  const samDetail = useToolStore((s) => s.samDetail);
  const samThreshold = useToolStore((s) => s.samThreshold);
  const samAvoidLabeled = useToolStore((s) => s.samAvoidLabeled);
  const samConnectedOnly = useToolStore((s) => s.samConnectedOnly);
  const fitRequestId = useToolStore((s) => s.fitRequestId);
  const clipToOtherClasses = useToolStore((s) => s.clipToOtherClasses);
  const mergeOverlappingSameClass = useToolStore((s) => s.mergeOverlappingSameClass);
  const fillThreshold = useToolStore((s) => s.fillThreshold);
  // NOTE: thresholdLo/thresholdHi are deliberately NOT subscribed — dragging the
  // band would then re-render this entire component every frame. They are read
  // via getState() at use sites and drive the overlay through a store
  // subscription (see the overlay block below).
  const thresholdOverlay = useToolStore((s) => s.thresholdOverlay);
  const thresholdSampleWidth = useToolStore((s) => s.thresholdSampleWidth);
  const setThresholdBand = useToolStore((s) => s.setThresholdBand);
  const { classes } = useClassStore();

  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [transform, setTransform] = useState({ scaleX: 1, scaleY: 1, x: 0, y: 0 });

  // Read aliases over the tracked store draft, gated to the current sample+slice so an
  // undo that resurrects a draft from another slice/sample doesn't render here.
  const draftMatchesContext = draft.sourceKey === sourceKey && draft.sliceKey === String(currentSlice);
  const draftPoly = draftMatchesContext && draft.tool === 'polygon' ? draft.poly : [];
  // Whether the pointer is over the canvas — drives the brush/eraser cursor preview
  // visibility reactively (so it survives re-renders and tool switches, unlike a
  // purely imperative toggle).
  const [pointerInside, setPointerInside] = useState(false);
  // Draft rect/ellipse start
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);
  // Live polygon vertex edit (select tool): { id, points } while dragging a handle.
  const [editPoly, setEditPoly] = useState<{ id: string; points: number[]; holes?: number[][] } | null>(null);

  // Magic-wand selection: seed click + preview polygons (image coords) before commit.
  // Marquee rubber-band (select tool): drag a box to select multiple shapes.
  const [marqueeStart, setMarqueeStart] = useState<{ x: number; y: number } | null>(null);
  const [marqueeRect, setMarqueeRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  // True when the marquee drag began with Shift held — its result is added to
  // (unioned with) the existing selection instead of replacing it.
  const marqueeShiftRef = useRef(false);

  // Magic seeds: each click is a prompt point. label 1 = positive (grow the
  // object / region), 0 = negative (carve back out — SAM only). The classic
  // wand ignores the label and treats each as a fresh seed.
  const [magicSeeds, setMagicSeeds] = useState<Array<{ x: number; y: number; label: 0 | 1 }>>([]);
  const [magicPreview, setMagicPreview] = useState<number[][]>([]);
  const [magicLoading, setMagicLoading] = useState(false);
  // SAM box prompt (image coords): committed box + live drag preview.
  const [magicBox, setMagicBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [magicBoxDraft, setMagicBoxDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const magicDragStartRef = useRef<{ x: number; y: number } | null>(null);

  // Magnetic lasso (livewire). Committed path + seed live in the tracked store draft;
  // the preview and dijkstra/cost caches are local (never on the undo timeline).
  const magneticCommitted = draftMatchesContext && draft.tool === 'magnetic' ? draft.magnetic : [];
  const magneticSeed = draftMatchesContext && draft.tool === 'magnetic' ? draft.magneticSeed : null;
  const [magneticPreview, setMagneticPreview] = useState<number[]>([]);
  const magneticCostRef = useRef<CostMap | null>(null);
  const magneticBuiltForRef = useRef<CanvasImageSource | null>(null);
  const magneticPrevRef = useRef<Int32Array | null>(null);
  // True while a brush/eraser stroke is actively being drawn — suppresses the
  // (expensive) layer re-cache so we don't rebuild the shapes bitmap mid-stroke.
  const [isDrawing, setIsDrawing] = useState(false);

  // Denoising is applied server-side, before normalization — so the PNG this
  // returns is already denoised, and everything downstream that samples it
  // (displayBase, the Threshold Brush field, the Sampler's fit, magic wand,
  // livewire) sees the denoised data with no extra plumbing. That is the point:
  // an intensity tool should act on the image the user is actually looking at.
  const { data: sliceUrl } = useImageSlice(source, kind, currentSlice, renderOpts, serverUri, denoise);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!sliceUrl) return;
    const img = new window.Image();
    img.onload = () => setImageEl(img);
    img.src = sliceUrl;
  }, [sliceUrl]);

  // SAM (in-browser Segment Anything) — the smart magic engine. Only spun up
  // while the magic tool is active and the SAM engine is selected.
  const samActive = tool === 'magic' && magicEngine === 'sam';
  const sam = useSam(samActive);

  // If SAM can't load or encode at all (no WebGPU/WASM, model fetch failed),
  // silently fall back to the classic wand so the tool always works.
  useEffect(() => {
    if ((sam.status === 'unsupported' || sam.error) && magicEngine === 'sam') {
      setMagicEngine('classic');
    }
  }, [sam.status, sam.error, magicEngine, setMagicEngine]);

  // Display-only nonlinear preprocessors baked into an offscreen canvas that
  // becomes the Konva image base; brightness/contrast/levels/gamma/colormap still
  // apply on top via the GPU SVG filter. Cache-key fragment so encodes/fields
  // refresh when toggled.
  const preprocess = useMemo(() => ({ clahe, sharpen, blur }), [clahe, sharpen, blur]);
  const preprocessKey = `${clahe ? 1 : 0}${sharpen ? 1 : 0}b${blur}`;
  // Working resolution, clamped so a huge slice can't blow up memory at 4x
  // (each level costs 4x the pixels for the base canvas AND every tool field).
  const workScale = useMemo(() => {
    if (!meta) return 1;
    const u = Math.max(1, upscale);
    return meta.width * meta.height * u * u > MAX_WORKING_PIXELS ? 1 : u;
  }, [meta, upscale]);
  const displayBase = useMemo<CanvasImageSource | null>(() => {
    if (!imageEl || !meta) return imageEl;
    return (clahe || sharpen || blur > 0 || workScale > 1)
      ? renderPreprocessOnly(imageEl, meta.width, meta.height, preprocess, workScale)
      : imageEl;
  }, [imageEl, meta, clahe, sharpen, blur, workScale, preprocess]);

  // SAM sees the preprocessed + brightness/contrast/levels-adjusted image
  // (windowing a low-contrast slice greatly helps), so the encode is keyed on
  // those — adjusting them re-encodes. Source building is deferred to a real encode.
  const samEncodeKey = imageEl && meta
    ? `${sourceKey}|${currentSlice}|b${brightness}|c${contrast}|l${levelsLo}-${levelsHi}|pp${preprocessKey}`
    : null;
  /** Lazily render the display-adjusted slice (preprocess → brightness/contrast/levels)
   *  that SAM encodes. Deliberately kept at 1x: SAM resizes its input to 1024²
   *  internally, so an upscaled source would only cost memory. */
  const makeSamSource = useCallback(
    () => renderAdjusted(imageEl!, meta!.width, meta!.height, brightness, contrast, levelsLo, levelsHi, preprocess),
    [imageEl, meta, brightness, contrast, levelsLo, levelsHi, preprocess],
  );

  // Proactively encode the slice when SAM is active so the first click is fast.
  // Deliberately NOT keyed on sam.status — ensureEncoded is idempotent and
  // keying on status would re-fire every encoding→ready transition.
  useEffect(() => {
    if (samActive && samEncodeKey) {
      sam.ensureEncoded(samEncodeKey, makeSamSource).catch(() => { /* fallback handled by sam.error */ });
    }
  }, [samActive, samEncodeKey, makeSamSource, sam.ensureEncoded]); // eslint-disable-line react-hooks/exhaustive-deps

  // Display adjustments (brightness + contrast + levels) applied as a single
  // GPU-composited SVG filter on the image layer's canvas — NOT Konva CPU pixel
  // filters. Folds all three into one per-channel linear transform
  // (out = slope*in + intercept), so dragging the sliders re-composites on the
  // GPU with zero per-pixel JS and no re-cache → lag-free at any image size.
  // Math mirrors `renderAdjusted` exactly, so the tools (SAM/wand/Fill) stay in
  // sync with what's displayed.
  const displayFilterId = 'display-adjust-' + useId().replace(/[^a-zA-Z0-9]/g, '');
  const displayAffine = useMemo(() => {
    const { slope, intercept } = displayAffineFor(brightness, contrast, levelsLo, levelsHi);
    // The filter is a no-op only when brightness/contrast/levels AND gamma AND
    // colormap are all identity — otherwise it must stay applied.
    const identity =
      brightness === 0 && contrast === 0 && levelsLo <= 0 && levelsHi >= 255 &&
      gamma === 1 && colormap === 'gray';
    return { slope, intercept, identity };
  }, [brightness, contrast, levelsLo, levelsHi, gamma, colormap]);

  // Colormap LUT (per-channel tableValues) for the display filter; null = gray.
  const cmapTables = useMemo(() => colormapTables(colormap), [colormap]);

  useEffect(() => {
    const layer = imageLayerRef.current;
    if (!layer) return;
    const canvas =
      (layer as unknown as { getNativeCanvasElement?: () => HTMLCanvasElement }).getNativeCanvasElement?.() ??
      (layer.getCanvas() as unknown as { _canvas: HTMLCanvasElement })._canvas;
    if (canvas) canvas.style.filter = displayAffine.identity ? 'none' : `url(#${displayFilterId})`;
  }, [displayAffine.identity, displayFilterId, imageEl, stageSize]);

  // Compute a 256-bin luminance histogram of the current slice (downsampled) for
  // the levels control. Samples the PREPROCESSED base (CLAHE/Sharpen baked) so the
  // histogram reflects what the levels window actually operates on — re-runs when a
  // display preprocessor toggles.
  useEffect(() => {
    if (!imageEl || !onHistogram) return;
    const src = displayBase ?? imageEl;
    const maxDim = 512;
    const scale = Math.max(1, Math.ceil(Math.max(imageEl.width, imageEl.height) / maxDim));
    const w = Math.max(1, Math.floor(imageEl.width / scale));
    const h = Math.max(1, Math.floor(imageEl.height / scale));
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    ctx.drawImage(src, 0, 0, w, h);
    let data: Uint8ClampedArray;
    try { data = ctx.getImageData(0, 0, w, h).data; } catch { return; }
    const bins = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
      const lum = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) | 0;
      bins[lum < 0 ? 0 : lum > 255 ? 255 : lum]++;
    }
    onHistogram(bins);
  }, [imageEl, displayBase, onHistogram]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setStageSize({ width: el.offsetWidth, height: el.offsetHeight });
    });
    ro.observe(el);
    setStageSize({ width: el.offsetWidth, height: el.offsetHeight });
    return () => ro.disconnect();
  }, []);

  // Fit the image into the viewport, centered (also bound to the F shortcut).
  const fitToScreen = useCallback(() => {
    if (!imageEl || !meta) return;
    const scaleX = stageSize.width / meta.width;
    const scaleY = stageSize.height / meta.height;
    const scale = Math.min(scaleX, scaleY, 1);
    setTransform({
      scaleX: scale, scaleY: scale,
      x: (stageSize.width - meta.width * scale) / 2,
      y: (stageSize.height - meta.height * scale) / 2,
    });
  }, [imageEl, meta, stageSize]);

  // Auto-fit when the image or viewport changes.
  useEffect(() => {
    fitToScreen();
  }, [fitToScreen]);

  // Explicit fit requests (F shortcut) — skip the initial id=0 since the
  // auto-fit effect above already handles the first render.
  const lastFitIdRef = useRef(0);
  useEffect(() => {
    if (fitRequestId === lastFitIdRef.current) return;
    lastFitIdRef.current = fitRequestId;
    fitToScreen();
  }, [fitRequestId, fitToScreen]);

  // Focus region (from an Insights QA flag): zoom to the box + flash a highlight.
  const [focusHighlight, setFocusHighlight] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const focusNonceRef = useRef<number | null>(null);
  const focusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!focusRegion || !meta || focusRegion.nonce === focusNonceRef.current) return;
    focusNonceRef.current = focusRegion.nonce;
    const { x, y, w, h } = focusRegion;
    // Zoom so the box fills ~60% of the viewport, centered; capped scale.
    const pad = 1.6;
    const s = Math.min(
      stageSize.width / Math.max(1, w * pad),
      stageSize.height / Math.max(1, h * pad),
      8,
    );
    const scale = Math.max(s, 0.01);
    const cx = x + w / 2, cy = y + h / 2;
    setTransform({
      scaleX: scale, scaleY: scale,
      x: stageSize.width / 2 - cx * scale,
      y: stageSize.height / 2 - cy * scale,
    });
    setFocusHighlight({ x, y, w, h });
    if (focusTimerRef.current) clearTimeout(focusTimerRef.current);
    focusTimerRef.current = setTimeout(() => setFocusHighlight(null), 4000);
  }, [focusRegion, meta, stageSize.width, stageSize.height]);
  useEffect(() => () => { if (focusTimerRef.current) clearTimeout(focusTimerRef.current); }, []);

  const isPreviewing = previewShapes !== null;
  // Stable empty fallback: a fresh `[]` here changes identity every render, which
  // would make the `autoNegPoints` memo (and the SAM preview effect that depends
  // on it) re-run in a loop — leaving `magicLoading` stuck true so "Add"/Enter
  // never enable. Reuse one constant instead.
  const storeShapes = sourceKey ? (byImage[sourceKey]?.[String(currentSlice)] ?? EMPTY_SHAPES) : EMPTY_SHAPES;
  // Shapes actually drawn on Layer 1: previewed version when previewing, else the live store.
  const displayShapes = isPreviewing ? previewShapes! : storeShapes;
  // Classes used for color/visibility lookups when rendering Layer 1.
  const renderClasses = isPreviewing && previewClasses ? previewClasses : classes;

  // Clip stroke/shape layers to the image frame (layer coords == image coords,
  // since pan/zoom lives on the Stage) so painting past the edge is cropped.
  const imageClip = meta
    ? { clipX: 0, clipY: 0, clipWidth: meta.width, clipHeight: meta.height }
    : {};

  // Cache the shapes layer for uniform opacity compositing — skipped mid-stroke
  // so the cache rebuild doesn't stall painting.
  //
  // The cache pixel ratio is quantized to powers of two rather than tracking zoom
  // continuously. Re-caching rasterizes the whole layer, and keying it on the raw
  // scale meant every wheel tick paid for that. Rounding UP to the next bucket
  // means the cache is never coarser than the old `pixelRatio = scale` (at 1.5x it
  // now caches at 2x), so shapes are equally or more crisp while most zoom steps
  // reuse the existing cache outright.
  const cachePixelRatio = useMemo(() => {
    const scale = Math.min(Math.max(transform.scaleX, 1), 4);
    return Math.min(4, Math.pow(2, Math.ceil(Math.log2(scale))));
  }, [transform.scaleX]);

  useEffect(() => {
    const layer = shapesLayerRef.current;
    if (!layer) return;
    if (isDrawing) return;
    time('layer-cache', () => {
      layer.clearCache();
      if (displayShapes.length > 0) {
        layer.cache({ pixelRatio: cachePixelRatio });
      }
      layer.batchDraw();
    });
  }, [displayShapes, renderClasses, fillOpacity, meta, cachePixelRatio, isDrawing]);

  // Class lookups by id. These run per shape, on both shape layers, on every
  // render — a linear scan there is O(shapes × classes) for what should be O(1).
  const classMap = useMemo(
    () => new Map(classes.map((c) => [c.classId, c])),
    [classes],
  );
  const renderClassMap = useMemo(
    () => (renderClasses === classes ? classMap : new Map(renderClasses.map((c) => [c.classId, c]))),
    [renderClasses, classes, classMap],
  );

  const colorForClass = (classId: number) =>
    renderClassMap.get(classId)?.color ?? '#ff0000';
  /** Visibility of a shape's class (defaults to visible for unknown classes). */
  const isShapeVisible = useCallback(
    (s: Shape) => classMap.get(s.classId)?.isVisible !== false,
    [classMap],
  );

  /** Commit new shapes, clipping them against other classes when the toggle is on
   *  (neighbor classes act as a hard boundary), and merging with overlapping
   *  same-class shapes when that toggle is on. One undo step. */
  /** Pure: given new shapes + the slice's current shapes, return the FULL next slice array
   *  after clip-to-other-classes and auto-merge-same-class. Used by both `commitShapes` and
   *  the atomic polygon/lasso close so the whole result is one undo step. */
  const computeCommittedSlice = useCallback((newShapes: Shape[], sliceShapes: Shape[]): Shape[] => {
    let toAdd = newShapes;
    if (clipToOtherClasses && meta && newShapes.some((s) => hasOtherClass(sliceShapes, s.classId))) {
      toAdd = time('clip', () => clipShapesToOthers(newShapes, sliceShapes, meta.width, meta.height, workScale));
    }
    // Auto-merge with overlapping same-class shapes (replaces those + the new shape with
    // one unioned polygon). Runs after clipping so other-class bounds still hold.
    if (mergeOverlappingSameClass && meta && toAdd.length) {
      const { add, removeIds } = time('merge', () => mergeNewWithSameClass(toAdd, sliceShapes, meta.width, meta.height, workScale));
      if (removeIds.length) {
        const kept = sliceShapes.filter((s) => !removeIds.includes(s.id));
        return [...kept, ...add];
      }
      toAdd = add;
    }
    return [...sliceShapes, ...toAdd];
  }, [clipToOtherClasses, mergeOverlappingSameClass, meta, workScale]);

  const commitShapes = useCallback((newShapes: Shape[]) => {
    if (!sourceKey || newShapes.length === 0) return;
    const slice = useDatasetStore.getState().currentSlice;
    const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(slice)] ?? [];
    const next = time('commit', () => computeCommittedSlice(newShapes, sliceShapes));
    setShapes(sourceKey, slice, next);
  }, [sourceKey, computeCommittedSlice, setShapes]);

  const activeColor =
    activeClassId !== null ? colorForClass(activeClassId) : '#4090ff';

  const showBrushCursor =
    (underlyingTool === 'brush' || underlyingTool === 'eraser' || underlyingTool === 'threshold') &&
    !!meta && !isPreviewing;

  /**
   * Fit the Threshold Brush band from a lassoed example and apply it.
   *
   * Everything inside the loop is a positive example; a ring just outside it
   * supplies negatives, so the result means "fill this and not what it touches"
   * rather than "fill everything this bright". Work happens on a crop around the
   * lasso, which keeps the blur sweep cheap on a full-resolution slice.
   */
  const runSamplerFit = (path: number[]): void => {
    if (!meta || path.length < 6) { onSamplerFit?.(null); return; }
    const field = ensureThresholdField();
    if (!field) { onSamplerFit?.(null); return; }
    const { gw, gh, scale } = field;

    // Everything below works on a CROP around the lasso, never the whole slice.
    // Rasterizing and especially dilating full-grid is what made a large sample
    // crawl: dilation costs one pass over its grid per cell of ring width, so on
    // a 2560² field that is billions of operations for a region a few hundred
    // pixels across. Cropping first makes the cost scale with the sample.
    let minGX = Infinity, minGY = Infinity, maxGX = -Infinity, maxGY = -Infinity;
    for (let i = 0; i + 1 < path.length; i += 2) {
      const gx = path[i] / scale;
      const gy = path[i + 1] / scale;
      if (gx < minGX) minGX = gx;
      if (gx > maxGX) maxGX = gx;
      if (gy < minGY) minGY = gy;
      if (gy > maxGY) maxGY = gy;
    }
    if (!Number.isFinite(minGX) || maxGX < minGX) { onSamplerFit?.(null); return; }

    // Margin: the ring, plus slack so the widest blur kernel is not distorted by
    // the crop edge. Ring width is estimated from the lasso's area in cells.
    const approxArea = Math.max(1, (maxGX - minGX) * (maxGY - minGY));
    const ringWidth = Math.min(MAX_RING_WIDTH, Math.max(4, Math.round(Math.sqrt(approxArea) / 2)));
    const margin = ringWidth + Math.ceil(3 * Math.max(...BLUR_CANDIDATES)) + 2;

    const x0 = Math.max(0, Math.floor(minGX) - margin);
    const y0 = Math.max(0, Math.floor(minGY) - margin);
    const x1 = Math.min(gw - 1, Math.ceil(maxGX) + margin);
    const y1 = Math.min(gh - 1, Math.ceil(maxGY) + margin);
    const cw = x1 - x0 + 1;
    const ch = y1 - y0 + 1;
    if (cw < 2 || ch < 2) { onSamplerFit?.(null); return; }

    // Rasterize the lasso directly into crop coordinates by shifting it into the
    // crop's frame (image units), so no full-size mask is ever allocated.
    const shifted = path.map((v, i) => (i % 2 === 0 ? v - x0 * scale : v - y0 * scale));
    const lassoShape: Shape = { id: 'sampler', classId: 0, kind: 'polygon', points: shifted };
    const cropPos = rasterizeShapes([lassoShape], cw, ch, scale);
    const cropNeg = backgroundRing(cropPos, cw, ch, dilate, ringWidth);

    const cropField = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) {
      const src = (y0 + y) * gw + x0;
      const dst = y * cw;
      for (let x = 0; x < cw; x++) cropField[dst + x] = field.gray[src + x];
    }

    const result = fitWithBlurSweep(cropField, cw, ch, cropPos, cropNeg, applyGaussianBlurGray);
    if (!result.ok) { onSamplerFit?.(null); return; }

    // Phase 2: also try a texture-aware score. Intensity alone cannot separate
    // materials that share a grey range but differ in grain, nor a material whose
    // brightness drifts across the slice. Fitting a projection over intensity +
    // band-pass + local-std + local-mean-ratio, then running the SAME band fitter
    // on the projected score, handles both — and because the score is graded by
    // the same skill number, the two options are directly comparable.
    const channels = buildChannels(cropField, cw, ch);
    const projection = fitProjection(channels, cropPos, cropNeg);
    let projected: SweepResult | null = null;
    if (projection) {
      const score = projectToScore(channels, projection);
      const { posHist, negHist } = sampleHistograms(score, cropPos, cropNeg);
      const fit = fitBand(posHist, negHist);
      if (fit.ok) projected = { ...fit, extraSigma: 0 };
    }

    // Only switch to the projected score when it is meaningfully better. Equal
    // results should stay on plain intensity: it is the mode the display sliders
    // steer and the histogram picker describes, so it is the one to prefer.
    const useProjection =
      projection !== null && projected !== null && projected.skill > result.skill + 0.05;

    if (useProjection && projected && projection) {
      projectionRef.current = { projection, key: baseKey };
      scoreFieldRef.current = null; // rebuilt lazily for the whole slice
      setThresholdBand(projected.lo, projected.hi);
      onSamplerFit?.({
        ...projected,
        displayLo: projected.lo,
        displayHi: projected.hi,
        collapsed: false,
        previousBand: [useToolStore.getState().thresholdLo, useToolStore.getState().thresholdHi],
        previousBlur: blur,
        appliedBlur: blur,
        mode: 'projected',
        intensitySkill: result.skill,
        weights: CHANNEL_NAMES.map((name, i) => ({ name, weight: projection.weights[i] })),
      });
      setTool('threshold');
      return;
    }

    // Plain intensity won — drop any projection left from an earlier sample so the
    // brush goes back to gating on what the display shows.
    projectionRef.current = null;
    scoreFieldRef.current = null;

    // The fit lives in the field's BASE space; the stored band is authored in
    // DISPLAYED space. Rather than enumerate the ways that conversion can fail,
    // do it and check it round-trips: convert to display, round as the store
    // will, convert back, and require the original base band. That catches both
    // failure modes at once —
    //   * collapse (contrast squashing the range onto one displayed value), and
    //   * saturation, where an endpoint lands on 0 or 255 and `displayBandToBase`
    //     correctly reopens it to ∓Infinity — which would silently select far
    //     MORE than was fitted.
    const displayLo = Math.round(baseToDisplay(result.lo, displayAffine, gamma));
    const displayHi = Math.round(baseToDisplay(result.hi, displayAffine, gamma));
    const roundTrip = displayBandToBase(displayLo, displayHi, displayAffine, gamma);
    // Storing the band as integers costs up to half a displayed level. How much
    // that is in BASE units depends on the local slope, which gamma makes vary
    // along the range — so measure it at the band edges rather than deriving it
    // from the affine part alone.
    const levelWidth = (d: number): number => {
      const a = displayToBase(d - 0.5, displayAffine, gamma);
      const b = displayToBase(d + 0.5, displayAffine, gamma);
      return a === null || b === null ? Infinity : Math.abs(b - a);
    };
    const tolerance = Math.max(1.5, 1.5 * Math.max(levelWidth(displayLo), levelWidth(displayHi)));
    const collapsed =
      !Number.isFinite(roundTrip.lo) ||
      !Number.isFinite(roundTrip.hi) ||
      Math.abs(roundTrip.lo - result.lo) > tolerance ||
      Math.abs(roundTrip.hi - result.hi) > tolerance;

    onSamplerFit?.({
      ...result,
      displayLo,
      displayHi,
      collapsed,
      previousBand: [useToolStore.getState().thresholdLo, useToolStore.getState().thresholdHi],
      previousBlur: blur,
      appliedBlur: combineSigma(blur, result.extraSigma),
    });

    if (collapsed) return; // leave the band alone; the UI explains why
    setThresholdBand(displayLo, displayHi);
    if (result.extraSigma > 0) onBlurChange?.(combineSigma(blur, result.extraSigma));
    // Sampling is a means, not an end: hand the user straight back to the brush
    // the band was just fitted for.
    setTool('threshold');
  };

  /** Snapped path from the current anchor to `to`, or a straight line if the
   *  live-wire is unavailable (no edge map yet, or an off-grid point). */
  const samplerTraceTo = (to: { x: number; y: number }): number[] => {
    const st = samplerLassoRef.current;
    if (!st) return [];
    if (!st.cm || !st.prev) return [to.x, to.y];
    try {
      // `.slice(2)` drops the anchor itself, which is already committed.
      return tracePath(st.cm, st.prev, imageToGrid(st.cm, to.x, to.y)).slice(2);
    } catch {
      return [to.x, to.y];
    }
  };

  /** Re-seed the live-wire at `at` so subsequent segments trace from there. */
  const samplerReseed = (at: { x: number; y: number }): void => {
    const st = samplerLassoRef.current;
    if (!st) return;
    st.anchor = at;
    st.prev = st.cm ? dijkstra(st.cm, imageToGrid(st.cm, at.x, at.y)) : null;
  };

  /** Close the sampler lasso, run the fit, and clear the transient state. */
  const finishSamplerLasso = (): void => {
    const st = samplerLassoRef.current;
    samplerLassoRef.current = null;
    if (draftLineRef.current) {
      draftLineRef.current.visible(false);
      draftLineRef.current.closed(false);
      draftStrokeLayerRef.current?.batchDraw();
    }
    if (!st) return;
    // Close the loop along the edge too, rather than cutting straight across it.
    const path = [...st.committed, ...samplerTraceTo(st.start)];
    runSamplerFit(path);
  };

  /** Current pointer position mapped from stage/display coords to image pixels. */
  const getPointerImagePos = () => {
    const stage = stageRef.current;
    if (!stage) return null;
    const pos = stage.getPointerPosition();
    if (!pos) return null;
    return toImage(pos, transform);
  };

  // Build (and cache, per rendered base) the live-wire edge cost map. Uses the
  // preprocessed display base so the magnetic lasso traces the enhanced image.
  const ensureCostMap = (): CostMap | null => {
    if (!imageEl || !meta) return null;
    const base = displayBase ?? imageEl;
    if (magneticCostRef.current && magneticBuiltForRef.current === base) {
      return magneticCostRef.current;
    }
    const cm = time('cost-map', () => buildCostMap(base, meta.width, meta.height, 512, workScale));
    magneticCostRef.current = cm;
    magneticBuiltForRef.current = base;
    return cm;
  };

  /** Clear ONLY the local magnetic derived state (dijkstra + preview). The committed path
   *  + seed live in the store draft, cleared separately via clearDraft()/commitDraftShapes()
   *  so they stay on the undo timeline. */
  const resetMagnetic = useCallback(() => {
    magneticPrevRef.current = null;
    setMagneticPreview([]);
  }, []);

  /** Clear all magic-wand/SAM seeds, box, and preview state. */
  const resetMagic = useCallback(() => {
    setMagicSeeds([]);
    setMagicPreview([]);
    setMagicLoading(false);
    setMagicBox(null);
    setMagicBoxDraft(null);
    magicDragStartRef.current = null;
  }, []);

  // Build (and cache) the grayscale field used by the wand + Fill. Built from the
  // DISPLAY-adjusted image (brightness/contrast/levels) so these tools operate on
  // what the user sees, exactly like SAM. Cache key includes the display settings.
  const magicFieldRef = useRef<GrayField | null>(null);
  const magicFieldForRef = useRef<string | null>(null);
  /** Cache-key fragment for fields built from the display-adjusted slice (wand, SAM):
   *  brightness/contrast/levels are part of what those tools see. */
  const adjustedKey = `${sourceKey}|${currentSlice}|b${brightness}|c${contrast}|l${levelsLo}-${levelsHi}|pp${preprocessKey}|u${workScale}`;
  /** Cache-key fragment for fields built from the PREPROCESSED base (threshold brush
   *  + its overlay, matching the histogram). Deliberately excludes brightness/
   *  contrast/levels so those sliders never invalidate a field. */
  const baseKey = `${sourceKey}|${currentSlice}|pp${preprocessKey}|u${workScale}`;
  const ensureMagicField = useCallback((): GrayField | null => {
    if (!imageEl || !meta) return null;
    if (magicFieldRef.current && magicFieldForRef.current === adjustedKey) return magicFieldRef.current;
    const src = renderAdjusted(imageEl, meta.width, meta.height, brightness, contrast, levelsLo, levelsHi, preprocess, workScale);
    const f = time('magic-field', () => buildField(src, meta.width, meta.height, 1600, workScale));
    magicFieldRef.current = f;
    magicFieldForRef.current = adjustedKey;
    return f;
  }, [imageEl, meta, brightness, contrast, levelsLo, levelsHi, preprocess, workScale, adjustedKey]);

  // Threshold-brush field. Built at FULL working resolution (no 1600-px cap) — a
  // brush needs per-pixel accuracy where a click-based wand can afford a coarser grid.
  //
  // Source is the PREPROCESSED base, not the brightness/contrast/levels-adjusted
  // render the wand and SAM build. That is NOT a change in semantics: the band is
  // still evaluated against displayed intensity, so the sliders steer the brush.
  // The equivalence is moved rather than lost — `bandInBaseSpace` maps the band
  // backwards through the display transform (monotonic, hence exactly the same
  // pixel set; see lib/displayTransform.ts), which turns a per-tick full-image
  // re-render into a couple of `Math.pow` calls. Only blur/CLAHE/sharpen, the
  // slice, and the working scale invalidate this field.
  // Phase 2: when a sample shows texture beats brightness, the brush gates on a
  // fitted projection of several channels instead of raw intensity. Held in refs
  // (not state) so activating it does not re-render the canvas.
  const projectionRef = useRef<{
    projection: { weights: number[]; centers: number[]; scales: number[] };
    key: string;
  } | null>(null);
  /** Whole-slice projected score, built lazily from `projectionRef`. */
  const scoreFieldRef = useRef<GrayField | null>(null);

  const thresholdFieldRef = useRef<GrayField | null>(null);
  const thresholdFieldForRef = useRef<string | null>(null);
  const ensureThresholdField = useCallback((): GrayField | null => {
    if (!imageEl || !meta) return null;
    if (thresholdFieldRef.current && thresholdFieldForRef.current === baseKey) return thresholdFieldRef.current;
    // No gradient: the brush gates purely on intensity, and a Sobel pass over a
    // full-resolution (possibly 4x) grid would be pure waste.
    const f = time('threshold-field', () => buildField(displayBase ?? imageEl, meta.width, meta.height, Math.max(meta.width, meta.height), workScale, false));
    thresholdFieldRef.current = f;
    thresholdFieldForRef.current = baseKey;
    return f;
  }, [imageEl, meta, displayBase, workScale, baseKey]);

  // Release the cached tool fields when the slice or sample changes. Each holds a
  // Float32 gray plane (plus a gradient, for the wand) sized to the working
  // resolution — up to tens of MB at 2x/4x — and without this they stay resident
  // until the next use happens to replace them, which may be never.
  useEffect(() => {
    return () => {
      magicFieldRef.current = null;
      magicFieldForRef.current = null;
      thresholdFieldRef.current = null;
      thresholdFieldForRef.current = null;
      overlayFieldRef.current = null;
      overlayFieldForRef.current = null;
      gateRef.current = null;
      magneticCostRef.current = null;
      magneticBuiltForRef.current = null;
    };
  }, [sourceKey, currentSlice]);

  /**
   * The field the Threshold Brush gates on.
   *
   * Normally the intensity field. Once a sample shows that texture separates the
   * feature better, this becomes the fitted projection of all channels, computed
   * across the whole slice and cached — the brush, the overlay and the band all
   * read it, so they cannot disagree about what is selected.
   */
  const ensureGateField = useCallback((): GrayField | null => {
    const base = ensureThresholdField();
    const active = projectionRef.current;
    if (!base || !active) return base;
    // A projection is only valid for the field it was fitted on; a slice or
    // preprocessing change invalidates it rather than silently misapplying it.
    if (active.key !== baseKey) {
      projectionRef.current = null;
      scoreFieldRef.current = null;
      return base;
    }
    if (scoreFieldRef.current) return scoreFieldRef.current;
    const channels = buildChannels(base.gray, base.gw, base.gh);
    const gray = projectToScore(channels, active.projection);
    const score: GrayField = { gw: base.gw, gh: base.gh, scale: base.scale, gray };
    scoreFieldRef.current = score;
    return score;
  }, [ensureThresholdField, baseKey]);

  /** True while the brush is gating on a fitted score rather than brightness. */
  const usingProjection = (): boolean =>
    projectionRef.current !== null && projectionRef.current.key === baseKey;

  /** The threshold band, mapped from the DISPLAYED intensities the user authored
   *  it in into the base space the cached fields live in. Brightness/contrast/
   *  levels/gamma therefore steer the brush exactly as they steer the image —
   *  without any of them invalidating a field or touching a pixel. */
  const bandInBaseSpace = useCallback(() => {
    const { thresholdLo, thresholdHi } = useToolStore.getState();
    // With a projection active the gate field is a fitted score, not displayed
    // brightness, so inverting the display transform would be meaningless — the
    // band is already in the score's own units. (A consequence worth knowing:
    // in that mode the display sliders no longer steer the brush.)
    if (projectionRef.current && projectionRef.current.key === baseKey) {
      return { lo: thresholdLo, hi: thresholdHi };
    }
    return displayBandToBase(thresholdLo, thresholdHi, displayAffine, gamma);
  }, [displayAffine, gamma, baseKey]);

  /** Binary gate over a field: 1 where the pixel reads as in-band on screen. This
   *  is what the brush may paint. Reads the band from the store rather than a
   *  subscribed value — see the overlay below for why the canvas deliberately does
   *  NOT re-render on band changes. */
  //  Cached across strokes: the gate only changes when the band, the display
  //  transform, or the underlying field changes — not per mousedown, where
  //  rebuilding it meant a fresh multi-megabyte pass at the start of every stroke.
  const gateRef = useRef<{ key: string; field: GrayField; gate: Uint8Array } | null>(null);
  const buildThresholdGate = useCallback((field: GrayField): Uint8Array => {
    const { lo, hi } = bandInBaseSpace();
    const key = `${baseKey}|${lo}|${hi}`;
    const cached = gateRef.current;
    if (cached && cached.key === key && cached.field === field) return cached.gate;

    const gate = new Uint8Array(field.gw * field.gh);
    const { gray } = field;
    for (let i = 0; i < gate.length; i++) {
      const v = gray[i];
      if (v >= lo && v <= hi) gate[i] = 1;
    }
    gateRef.current = { key, field, gate };
    return gate;
  }, [bandInBaseSpace, baseKey]);

  // ---- Red in-band overlay (ImageJ's threshold display) -------------------
  // Shows exactly which pixels the brush may paint. Three things keep dragging
  // the band (or any display slider) smooth, all of which cost real frames when
  // done the obvious way:
  //   1. Its own SMALL field (≤1024 px, no upscale, no gradient) instead of the
  //      brush's full-resolution one — activating the tool or nudging brightness
  //      must not trigger a full-res renderAdjusted + buildField.
  //   2. Repainted IMPERATIVELY into a persistent canvas — routing it through
  //      React state re-rendered this whole (very large) component per frame.
  //   3. Driven by a store subscription, so the canvas never re-renders on a band
  //      change at all. That's why thresholdLo/Hi are not subscribed above.
  const overlayFieldRef = useRef<GrayField | null>(null);
  const overlayFieldForRef = useRef<string | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayImageDataRef = useRef<ImageData | null>(null);
  const overlayImageRef = useRef<Konva.Image>(null);
  const showThresholdOverlay = tool === 'threshold' && thresholdOverlay && !isPreviewing;

  const ensureOverlayField = useCallback((): GrayField | null => {
    if (!imageEl || !meta) return null;
    // With a projection active the overlay must show the PROJECTED selection, or
    // it would advertise a different set of pixels than the brush paints. The
    // gate field is already cached, so reuse it rather than fitting a second one.
    if (projectionRef.current && projectionRef.current.key === baseKey) {
      return ensureGateField();
    }
    if (overlayFieldRef.current && overlayFieldForRef.current === baseKey) return overlayFieldRef.current;
    // Same preprocessed source as the brush field (see above), so the overlay shows
    // exactly what the brush would paint — including as the display sliders move,
    // since both read the band through `bandInBaseSpace`.
    const f = time('overlay-field', () => buildField(displayBase ?? imageEl, meta.width, meta.height, 1024, 1, false));
    overlayFieldRef.current = f;
    overlayFieldForRef.current = baseKey;
    return f;
  }, [imageEl, meta, displayBase, baseKey, ensureGateField]);

  /** Repaint the overlay canvas for the current band and push it to Konva. */
  const paintThresholdOverlay = useCallback(() => time('overlay-paint', () => {
    const node = overlayImageRef.current;
    if (!node) return;
    const field = ensureOverlayField();
    if (!field) return;
    const { gw, gh, gray } = field;
    let canvas = overlayCanvasRef.current;
    if (!canvas || canvas.width !== gw || canvas.height !== gh) {
      canvas = document.createElement('canvas');
      canvas.width = gw;
      canvas.height = gh;
      overlayCanvasRef.current = canvas;
      overlayImageDataRef.current = null;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { lo, hi } = bandInBaseSpace();
    // Reuse the pixel buffer across repaints — a band drag repaints every frame
    // and a fresh multi-MB ImageData per frame is pure GC churn.
    let img = overlayImageDataRef.current;
    if (!img) {
      img = ctx.createImageData(gw, gh);
      overlayImageDataRef.current = img;
    }
    const d = img.data;
    for (let i = 0; i < gray.length; i++) {
      const v = gray[i];
      const o = i * 4;
      if (v < lo || v > hi) { d[o + 3] = 0; continue; }
      d[o] = 239; d[o + 1] = 68; d[o + 2] = 68; d[o + 3] = 255; // red-500
    }
    ctx.putImageData(img, 0, 0);
    node.image(canvas);
    node.getLayer()?.batchDraw();
  }), [ensureOverlayField, bandInBaseSpace]);

  useEffect(() => {
    if (!showThresholdOverlay) return;
    // Coalesce bursts (a band drag commits once per frame) into one repaint.
    let pending = 0;
    const schedule = () => {
      if (pending) return;
      pending = requestAnimationFrame(() => { pending = 0; paintThresholdOverlay(); });
    };
    schedule();
    const unsub = useToolStore.subscribe((s, prev) => {
      if (s.thresholdLo !== prev.thresholdLo || s.thresholdHi !== prev.thresholdHi) schedule();
    });
    return () => { unsub(); if (pending) cancelAnimationFrame(pending); };
  }, [showThresholdOverlay, paintThresholdOverlay]);

  // Auto negative ("not") prompts for SAM: interior points of nearby other-class
  // regions, so a new selection won't bleed into already-labeled areas. Anchored
  // to the current click (last positive seed, else the box center), capped to the
  // 12 nearest, and skipping any region a positive click sits inside.
  const autoNegPoints = useMemo<Array<{ x: number; y: number }>>(() => {
    if (magicEngine !== 'sam' || !samAvoidLabeled || activeClassId === null) return [];
    const positives = magicSeeds.filter((s) => s.label === 1);
    const ref =
      positives.length > 0
        ? positives[positives.length - 1]
        : magicBox
          ? { x: magicBox.x + magicBox.w / 2, y: magicBox.y + magicBox.h / 2 }
          : null;
    if (!ref) return [];

    const isClassVisible = isShapeVisible;

    const cands: Array<{ x: number; y: number }> = [];
    for (const shape of storeShapes) {
      if (shape.classId === activeClassId || !isClassVisible(shape)) continue;
      // Don't fight the user's own target: skip a region a positive click is in.
      if (positives.some((p) => shapeContainsPoint(shape, p.x, p.y))) continue;
      const ip = interiorPoint(shape);
      if (ip) cands.push(ip);
    }
    cands.sort(
      (a, b) =>
        (a.x - ref.x) ** 2 + (a.y - ref.y) ** 2 - ((b.x - ref.x) ** 2 + (b.y - ref.y) ** 2),
    );
    return cands.slice(0, 12);
  }, [magicEngine, samAvoidLabeled, activeClassId, magicSeeds, magicBox, storeShapes, classes]);

  // (Re)compute the preview whenever the seeds or params change.
  // SAM engine: encode the slice (cached) then decode the point prompts into a
  // mask → polygons. Classic engine: client-side flood/threshold via magicSelect.
  // Magic sliders already debounce their store commits (DebouncedSlider).
  useEffect(() => {
    if (tool === 'magic' && magicEngine === 'sam') {
      if ((magicSeeds.length === 0 && !magicBox) || !imageEl || !meta) { setMagicPreview([]); return; }
      let cancelled = false;
      setMagicLoading(true);
      (async () => {
        const ready = samEncodeKey ? await sam.ensureEncoded(samEncodeKey, makeSamSource) : false;
        if (cancelled) return;
        if (!ready) { setMagicLoading(false); return; } // status flips → classic
        const points = [
          ...magicSeeds.map((s) => ({
            x: s.x / meta.width, y: s.y / meta.height, label: s.label,
          })),
          // Auto "not" prompts from nearby other-class regions (label 0).
          ...autoNegPoints.map((p) => ({
            x: p.x / meta.width, y: p.y / meta.height, label: 0 as const,
          })),
        ];
        const box = magicBox
          ? {
              x0: magicBox.x / meta.width, y0: magicBox.y / meta.height,
              x1: (magicBox.x + magicBox.w) / meta.width, y1: (magicBox.y + magicBox.h) / meta.height,
            }
          : null;
        const res = await sam.segment(points, box, samDetail, samThreshold);
        if (cancelled) return;
        if (res) {
          const scale = res.width > 0 ? meta.width / res.width : 1;
          let mask = res.mask;
          // "Connected regions only": keep the component(s) at the positive prompts
          // (positive seeds, else the box center), dropping detached speckle.
          if (samConnectedOnly) {
            const sx = res.width / meta.width, sy = res.height / meta.height;
            const pos = magicSeeds
              .filter((s) => s.label === 1)
              .map((s) => ({ x: s.x * sx, y: s.y * sy }));
            if (pos.length === 0 && magicBox) {
              pos.push({ x: (magicBox.x + magicBox.w / 2) * sx, y: (magicBox.y + magicBox.h / 2) * sy });
            }
            mask = keepComponentsAtPoints(mask, res.width, res.height, pos);
          }
          setMagicPreview(
            maskToPolygons(mask, res.width, res.height, {
              smooth: magicSigma, scale, minRegion: 25,
            }),
          );
        } else {
          setMagicPreview([]);
        }
        setMagicLoading(false);
      })();
      return () => { cancelled = true; };
    }

    if (magicSeeds.length === 0) { setMagicPreview([]); return; }
    // Classic flood — serves both the classic magic wand and the Fill tool
    // (Fill = contiguous flood by `fillThreshold`). Synchronous, one run per frame.
    const isFill = tool === 'fill';
    const field = ensureMagicField();
    if (!field) { setMagicPreview([]); return; }
    setMagicLoading(true);
    const id = requestAnimationFrame(() => {
      const polys: number[][] = [];
      for (const seed of magicSeeds) {
        polys.push(
          ...magicSelect(field, seed.x, seed.y, {
            toleranceFrac: isFill ? fillThreshold : magicTolerance,
            mode: isFill ? 'contiguous' : magicMode,
            smooth: isFill ? 1 : magicSigma,
            edgeStop: isFill ? 0 : magicEdgeStop,
          }),
        );
      }
      setMagicPreview(polys);
      setMagicLoading(false);
    });
    return () => cancelAnimationFrame(id);
  }, [tool, magicSeeds, magicBox, autoNegPoints, samDetail, samThreshold, samConnectedOnly, samEncodeKey, makeSamSource, magicEngine, magicTolerance, magicMode, magicSigma, magicEdgeStop, fillThreshold, ensureMagicField, imageEl, meta, sam.ensureEncoded, sam.segment]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Commit the magic preview polygons as new shapes (one batched undo step). */
  const commitMagic = useCallback(() => {
    if (!sourceKey || activeClassId === null) return;
    // One batched add = one undo step for the whole magic selection.
    const shapes = magicPreview
      .filter((pts) => pts.length >= 6)
      .map((pts) => ({ id: uuidv4(), classId: activeClassId, kind: 'polygon' as const, points: pts }));
    if (shapes.length) commitShapes(shapes);
    resetMagic();
  }, [sourceKey, activeClassId, magicPreview, commitShapes, resetMagic]);

  // Preserve in-progress polygon/lasso/magic drafts across a transient hold-Space
  // pan (tool flips to 'pan' then back), but abandon them on a real tool switch.
  const prevToolRef = useRef(tool);
  useEffect(() => {
    const prev = prevToolRef.current;
    prevToolRef.current = tool;
    if (tool === 'pan' || prev === 'pan') return; // entering/leaving pan keeps the draft
    // Tracked clear: undo can reopen the abandoned draft (same slice/source context).
    clearDraft();
    resetMagnetic();
    resetMagic();
  }, [tool, clearDraft, resetMagnetic, resetMagic]);

  // A new image/slice always invalidates any in-progress draft. currentSlice lives in a
  // separate (untracked) store, so clear WITHOUT a history entry (pause) — otherwise undo
  // would produce a half-reversible step it can't fully restore.
  useEffect(() => {
    const temporal = useAnnotationStore.temporal.getState();
    temporal.pause();
    clearDraft();
    temporal.resume();
    resetMagnetic();
    resetMagic();
  }, [currentSlice, sourceKey, clearDraft, resetMagnetic, resetMagic]);

  // Keep the latest tool / preview / commit fn in refs so the single, stable Enter
  // handler always sees current values — no stale closure, no listener re-subscribe
  // race — and can approve the SHOWN selection even if a background re-segmentation
  // just flipped `magicLoading` true (which previously made Enter silently no-op).
  const magicPreviewRef = useRef(magicPreview);
  magicPreviewRef.current = magicPreview;
  const commitMagicRef = useRef(commitMagic);
  commitMagicRef.current = commitMagic;
  const toolRef = useRef(tool);
  toolRef.current = tool;
  const currentSliceRef = useRef(currentSlice);
  currentSliceRef.current = currentSlice;

  // Escape cancels the entire in-progress shape (polygon vertices, magnetic
  // trace, magic selection, or rect/ellipse drag) without committing anything.
  // Enter accepts the current magic-tool selection (same as the "Add" button).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Escape') {
        clearDraft();          // tracked: undo can reopen the abandoned draft
        setDragStart(null);
        setDragCurrent(null);
        setMarqueeStart(null);
        setMarqueeRect(null);
        resetMagnetic();
        resetMagic();
      } else if (
        e.key === 'Enter' &&
        (toolRef.current === 'magic' || toolRef.current === 'fill') &&
        magicPreviewRef.current.length > 0
      ) {
        e.preventDefault();
        commitMagicRef.current();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [clearDraft, resetMagnetic, resetMagic]);

  // Magnetic node-by-node undo/redo now falls out of the tracked store draft (each node
  // is one undo step; closing/reopening is automatic). The only thing to reconcile is the
  // DERIVED dijkstra map: whenever the committed magnetic seed changes from ANY cause —
  // placing a node, or an undo/redo restoring an earlier draft — recompute it so tracing
  // continues. useLayoutEffect so magneticPrevRef is ready before paint (no stale-trace flash).
  useLayoutEffect(() => {
    if (
      draft.tool !== 'magnetic' ||
      draft.sourceKey !== sourceKey ||
      draft.sliceKey !== String(currentSlice) ||
      !draft.magneticSeed
    ) {
      magneticPrevRef.current = null;
      return;
    }
    const cm = ensureCostMap();
    magneticPrevRef.current = cm
      ? dijkstra(cm, imageToGrid(cm, draft.magneticSeed.x, draft.magneticSeed.y))
      : null;
    // ensureCostMap is a stable ref-reader; imageEl/displayBase cover its inputs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft.magneticSeed, draft.tool, draft.sourceKey, draft.sliceKey, sourceKey, currentSlice, imageEl, displayBase]);

  // ---- Threshold brush ----------------------------------------------------
  // Paints only where the displayed intensity is inside the band, so a stroke
  // stops dead at a feature boundary. The result can't be expressed as a stroke +
  // radius, so it accumulates as a mask and commits as POLYGONS.

  /** Begin a threshold stroke at `pos`, priming the mask, gate, and preview canvas. */
  const startThresholdStroke = (pos: { x: number; y: number }, erase: boolean): void => {
    const field = ensureGateField();
    if (!field || !meta) return;
    const { gw, gh, scale } = field;
    const canvas = document.createElement('canvas');
    canvas.width = gw;
    canvas.height = gh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    thresholdStrokeRef.current = {
      mode: erase ? 'erase' : 'paint',
      gw, gh, scale,
      mask: new Uint8Array(gw * gh),
      gate: buildThresholdGate(field),
      last: pos,
      canvas,
      imageData: ctx.createImageData(gw, gh),
    };
    if (thresholdPreviewRef.current) {
      thresholdPreviewRef.current.image(canvas);
      thresholdPreviewRef.current.visible(true);
    }
    extendThresholdStroke(pos);
  };

  /** Stamp the segment from the last point to `pos` and repaint the preview.
   *  Only the segment's dirty rect is rewritten — repainting the whole grid every
   *  mousemove would be tens of millions of writes per frame at a 2x/4x full-res
   *  grid, which is exactly the size this tool is meant to be used at. */
  const extendThresholdStroke = (pos: { x: number; y: number }): void => {
    const st = thresholdStrokeRef.current;
    if (!st) return;
    const { gw, gh, scale, mask, gate } = st;
    const from = st.last;
    stampStroke(mask, gw, gh, [from.x, from.y, pos.x, pos.y], brushSize / scale, scale, 1, gate);
    st.last = pos;

    // Dirty rect of this segment in grid cells (the stamped capsule's bbox).
    const r = Math.max(0.5, brushSize / scale) + 1;
    const x0 = Math.max(0, Math.floor(Math.min(from.x, pos.x) / scale - r));
    const x1 = Math.min(gw - 1, Math.ceil(Math.max(from.x, pos.x) / scale + r));
    const y0 = Math.max(0, Math.floor(Math.min(from.y, pos.y) / scale - r));
    const y1 = Math.min(gh - 1, Math.ceil(Math.max(from.y, pos.y) / scale + r));
    if (x1 < x0 || y1 < y0) return;

    // Erase strokes preview white (matching the eraser's draft line); paint
    // strokes use the active class color.
    const d = st.imageData.data;
    const [cr, cg, cb] = st.mode === 'erase' ? [255, 255, 255] : hexToRgb(activeColor);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * gw + x;
        if (!mask[i]) continue;
        const o = i * 4;
        d[o] = cr; d[o + 1] = cg; d[o + 2] = cb; d[o + 3] = 255;
      }
    }
    st.canvas
      .getContext('2d')
      ?.putImageData(st.imageData, 0, 0, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    thresholdPreviewRef.current?.getLayer()?.batchDraw();
  };

  /** Flush the buffered threshold stroke: mask → polygons → one undo step. */
  const commitThresholdStroke = (): void => {
    const st = thresholdStrokeRef.current;
    thresholdStrokeRef.current = null;
    if (thresholdPreviewRef.current) {
      thresholdPreviewRef.current.visible(false);
      thresholdPreviewRef.current.image(undefined);
      thresholdPreviewRef.current.getLayer()?.batchDraw();
    }
    if (!st || !sourceKey || !meta || activeClassId === null) return;

    const { gw, gh, scale } = st;
    let any = false;
    for (let i = 0; i < st.mask.length; i++) if (st.mask[i]) { any = true; break; }
    if (!any) return;

    // Regularize before vectorizing. A per-pixel gate speckles, and every speck
    // becomes its own polygon: that is what made one stroke commit hundreds of
    // shapes and dominate clip cost. A morphological opening (erode then dilate)
    // drops isolated pixels and pinholes while leaving real regions intact, and
    // the small-component pass clears what survives. Better geometry AND a much
    // cheaper commit, from the same step.
    const mask = removeSmallComponents(
      dilate(erode(st.mask, gw, gh, 1), gw, gh, 1),
      gw,
      gh,
      THRESHOLD_MIN_REGION,
    );
    let survives = false;
    for (let i = 0; i < mask.length; i++) if (mask[i]) { survives = true; break; }
    // A thin stroke can be erased entirely by the opening; keep the raw mask
    // rather than silently discarding what the user just painted.
    const finalMask = survives ? mask : st.mask;

    const regions = maskToPolygonsWithHoles(finalMask, gw, gh, { minRegion: 4, scale })
      .filter((p) => p.points.length >= 6);
    if (regions.length === 0) return;

    if (st.mode === 'erase') {
      // Subtract the painted region from the shapes it actually overlaps, via the
      // same node-preserving boolean path the eraser uses. The raster overlap test
      // matters: `subtractFromShape` always rebuilds geometry, so running it on
      // untouched shapes would churn their ids and vertices for nothing.
      const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const visible = isShapeVisible;
      const inScope = (s: Shape) => eraseAllClasses || s.classId === activeClassId;
      // The stroke's own bounds come free from the regions we just vectorized, so
      // a shape nowhere near it is rejected before any rasterization happens.
      const strokeBounds = unionBBox(
        regions.map((p) => ({ id: '', classId: 0, kind: 'polygon' as const, points: p.points })),
      );
      const scratch = new Uint8Array(gw * gh);
      const overlapsStroke = (s: Shape) => {
        if (!bboxNear(shapeBBox(s), strokeBounds, scale + 1)) return false;
        scratch.fill(0);
        rasterizeShapes([s], gw, gh, scale, scratch);
        for (let i = 0; i < finalMask.length; i++) if (finalMask[i] && scratch[i]) return true;
        return false;
      };
      const stampMP = regionsToMultiPolygon(regions);
      if (stampMP.length === 0) return;

      const next: Shape[] = [];
      const replacedSelection: string[] = [];
      let changed = false;
      for (const s of sliceShapes) {
        if (!inScope(s) || !visible(s) || !overlapsStroke(s)) { next.push(s); continue; }
        const polys = subtractFromShape(s, stampMP, meta.width, meta.height);
        if (polys === null) { next.push(s); continue; }
        changed = true;
        next.push(...polys);
        if (selectedShapeIds.includes(s.id)) replacedSelection.push(...polys.map((p) => p.id));
      }
      if (changed) {
        setShapes(sourceKey, currentSlice, next);
        setSelectedShapeIds([
          ...selectedShapeIds.filter((id) => next.some((s) => s.id === id)),
          ...replacedSelection,
        ]);
      }
      return;
    }

    // Paint: commit as polygons through the shared path, so clip-to-other-classes
    // and merge-same-class apply and the whole stroke is a single undo step.
    commitShapes(
      regions.map((p) => ({
        id: uuidv4(),
        classId: activeClassId,
        kind: 'polygon' as const,
        points: p.points,
        ...(p.holes.length ? { holes: p.holes } : {}),
      })),
    );
  };

  /** Flush the buffered draft stroke to the Zustand store (one write per stroke). */
  const commitDraftStroke = () => {
    const draft = draftStrokeRef.current;
    if (!draft || !sourceKey) return;
    const { shapeId, mode, points, radius } = draft;
    draftStrokeRef.current = null;

    // Hide the draft line imperatively
    if (draftLineRef.current) {
      draftLineRef.current.visible(false);
      draftStrokeLayerRef.current?.batchDraw();
    }

    if (points.length < 2) return;
    // Duplicate single point so Konva renders it as a dot
    const finalPoints = points.length === 2 ? [...points, ...points] : points;
    if (mode === 'erase') {
      if (!meta) return;
      const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const visible = isShapeVisible;
      const inScope = (s: Shape) => eraseAllClasses || s.classId === activeClassId;
      const strokeHits = (s: Shape) => {
        // Radius-aware so grazing a shape's edge (disk overlaps, center outside)
        // still erases — matches the pixels the erase stamp actually removes.
        for (let i = 0; i + 1 < finalPoints.length; i += 2) {
          if (shapeNearPoint(s, finalPoints[i], finalPoints[i + 1], radius)) return true;
        }
        return false;
      };
      // Erase carves EVERY in-scope, visible shape the stroke passes over (plus the
      // shape it started on, if any) — so a drag across several regions erases all
      // of them, not just the first. "Erase all classes" widens the scope past the
      // active class.
      const targetIds = new Set(
        sliceShapes
          .filter((s) => s.id === shapeId || (visible(s) && inScope(s) && strokeHits(s)))
          .map((s) => s.id),
      );
      if (targetIds.size === 0) return;

      // Bake the erase directly into polygon geometry for every target — the
      // vertices always match the visible shape, splits produce independent
      // polygons, and undo is one clean shape-replacement step.
      const { gw, gh, scale } = fullResGridFor(meta.width, meta.height, workScale);
      // The erase stamp as polygon geometry, subtracted from each shape via a true
      // boolean difference so untouched vertices are PRESERVED (only the cut edge
      // gets new points). Falls back per-shape to a rasterize→re-vectorize round-trip
      // (which also bakes any legacy `erased` carve-outs) if the boolean op can't run.
      const stampMP = eraseStampToMultiPolygon(finalPoints, radius, meta.width, meta.height);
      const next: Shape[] = [];
      const replacedSelection: string[] = [];
      let clearedActiveBrush = false;
      for (const s of sliceShapes) {
        if (!targetIds.has(s.id)) { next.push(s); continue; }
        // Legacy vector shapes carrying `erased` carve-outs must go through the mask
        // path (it honors `erased`); everything else prefers node-preserving boolean.
        const hasLegacyErased = s.kind !== 'brush' && !!s.erased?.length;
        let polys: Shape[] | null = hasLegacyErased ? null : subtractFromShape(s, stampMP, meta.width, meta.height);
        if (polys === null) {
          const prospective: Shape = s.kind === 'brush'
            ? { ...s, strokes: [...s.strokes, { points: finalPoints, radius, mode: 'erase' as const }] }
            : { ...s, erased: [...(s.erased ?? []), { points: finalPoints, radius }] };
          const mask = rasterizeShapes([prospective], gw, gh, scale);
          polys = maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
            .filter((p) => p.points.length >= 6)
            .map((p) => ({
              id: uuidv4(), classId: s.classId, kind: 'polygon' as const,
              points: p.points, ...(p.holes.length ? { holes: p.holes } : {}),
            }));
        }
        next.push(...polys);
        if (selectedShapeIds.includes(s.id)) replacedSelection.push(...polys.map((p) => p.id));
        if (s.id === activeBrushShapeId) clearedActiveBrush = true;
      }
      setShapes(sourceKey, currentSlice, next);
      if (clearedActiveBrush) onNewBrushInstance(''); // the erased brush is now polygon(s)
      // Keep vertex editing live on any rebuilt polygon that was selected; drop ids
      // of shapes that were fully erased away.
      setSelectedShapeIds([
        ...selectedShapeIds.filter((id) => !targetIds.has(id)),
        ...replacedSelection,
      ]);
      return;
    }

    // Paint stroke with clip and/or same-class merge on: convert the brush instance
    // to a polygon, subtracting other-class pixels (clip) and unioning overlapping
    // same-class shapes (merge). Only converts when something actually changes;
    // otherwise the brush stays a brush.
    if (mode === 'paint' && meta && (clipToOtherClasses || mergeOverlappingSameClass)) {
      const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const brush = sliceShapes.find((s) => s.id === shapeId);
      if (brush && brush.kind === 'brush') {
        // Re-vectorize at full resolution so the round-trip is ~idempotent: existing
        // merged/clipped regions keep their shape instead of eroding or shifting a
        // little each time a stroke is added.
        const { gw, gh, scale } = fullResGridFor(meta.width, meta.height, workScale);
        const prospective = { ...brush, strokes: [...brush.strokes, { points: finalPoints, radius, mode: 'paint' as const }] };
        const mine = rasterizeShapes([prospective], gw, gh, scale);
        let clipChanged = false;

        // Clip detection (fast mask test): does the brush overlap other classes?
        // The actual clip is applied below via boolean difference so the brush tiles
        // flush against the neighbor (a mask carve left a ~1px unlabeled seam).
        // Bounds of the brush including this stroke. Used ONLY for the same-class
        // merge test below — the clip detection deliberately does not pre-filter
        // (see the note in lib/clipToClasses.ts about the reverted filter).
        const mineBounds = shapeBBox(prospective);
        const nearBrush = (s: Shape) => bboxNear(shapeBBox(s), mineBounds, scale + 1);
        const overlapScratch = new Uint8Array(gw * gh);

        if (clipToOtherClasses) {
          const others = sliceShapes.filter((s) => s.classId !== brush.classId);
          if (others.length > 0) {
            const otherMask = rasterizeUnion(others, gw, gh, scale);
            for (let i = 0; i < mine.length; i++) { if (mine[i] && otherMask[i]) { clipChanged = true; break; } }
          }
        }

        // Merge: which existing same-class shapes does the (clipped) brush overlap?
        const mergeTargets = mergeOverlappingSameClass
          ? sliceShapes.filter((s) => {
              if (s.id === shapeId || s.classId !== brush.classId) return false;
              if (!nearBrush(s)) return false;
              overlapScratch.fill(0);
              rasterizeShapes([s], gw, gh, scale, overlapScratch);
              for (let i = 0; i < mine.length; i++) if (mine[i] && overlapScratch[i]) return true;
              return false;
            })
          : [];

        if (clipChanged || mergeTargets.length) {
          // The new brush region as polygon(s) (not yet clipped).
          let brushPolys: Shape[] = maskToPolygonsWithHoles(mine, gw, gh, { minRegion: 4, scale })
            .filter((p) => p.points.length >= 6)
            .map((p) => ({
              id: uuidv4(), classId: brush.classId, kind: 'polygon' as const,
              points: p.points, ...(p.holes.length ? { holes: p.holes } : {}),
            }));

          // Clip via boolean difference so the brush tiles flush against other
          // classes (no ~1px unlabeled seam); existing shapes are untouched.
          //
          // `clipChanged` is decided by a mask test, so we KNOW there is overlap to
          // remove here. Every failure below must therefore fall back to the mask
          // clip rather than keeping the shape as-is: returning `bp` unchanged (the
          // old behaviour) leaves the brush overlapping its neighbour while the
          // toggle claims otherwise, which looks exactly like clipping being off.
          if (clipChanged) {
            const others = sliceShapes.filter((s) => s.classId !== brush.classId);
            const { mp: otherMP, ok } = unionShapesChecked(others, meta.width, meta.height);
            if (ok && otherMP.length) {
              // Boolean path per polygon; anything it can't do is batched into a
              // single mask clip (that helper caches its masks per call, so one
              // call for many shapes is far cheaper than one call each).
              const kept: Shape[] = [];
              const failed: Shape[] = [];
              for (const bp of brushPolys) {
                const sub = subtractFromShape(bp, otherMP, meta.width, meta.height);
                if (sub === null) failed.push(bp); else kept.push(...sub);
              }
              brushPolys = failed.length
                ? [...kept, ...clipShapesToOthersMask(failed, sliceShapes, meta.width, meta.height, workScale)]
                : kept;
            } else {
              brushPolys = clipShapesToOthersMask(brushPolys, sliceShapes, meta.width, meta.height, workScale);
            }
          }

          let resultPolys: Shape[] = brushPolys;
          if (mergeTargets.length) {
            // True boolean union so the existing same-class shapes keep their exact
            // vertices (only the seam changes); fall back to a mask union on failure.
            const merged = unionShapesToPolygons([...brushPolys, ...mergeTargets], brush.classId, meta.width, meta.height);
            if (merged) {
              resultPolys = merged;
            } else {
              const mask = rasterizeShapes([...brushPolys, ...mergeTargets], gw, gh, scale);
              resultPolys = maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
                .filter((p) => p.points.length >= 6)
                .map((p) => ({
                  id: uuidv4(), classId: brush.classId, kind: 'polygon' as const,
                  points: p.points, ...(p.holes.length ? { holes: p.holes } : {}),
                }));
            }
          }
          const removeIds = mergeTargets.map((s) => s.id);
          const kept = sliceShapes.filter((s) => s.id !== shapeId && !removeIds.includes(s.id));
          setShapes(sourceKey, currentSlice, [...kept, ...resultPolys]);
          onNewBrushInstance(''); // this brush instance is now polygon(s)
          return;
        }
      }
    }

    // Plain brush paint (no clip/merge conversion): if this stroke doesn't connect
    // to the target instance's existing strokes, start a NEW brush shape so painting
    // separate blobs yields separate, individually-selectable shapes.
    if (mode === 'paint') {
      const shapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const target = shapes.find((s) => s.id === shapeId);
      if (
        target && target.kind === 'brush' &&
        target.strokes.some((st) => st.mode === 'paint') &&
        !strokeTouchesBrush(finalPoints, radius, target.strokes)
      ) {
        const newId = uuidv4();
        addShape(sourceKey, currentSlice, {
          id: newId, classId: target.classId, kind: 'brush',
          strokes: [{ points: finalPoints, radius, mode: 'paint' }],
        });
        onNewBrushInstance(newId); // subsequent connected strokes extend this blob
        return;
      }
    }
    appendBrushStroke(sourceKey, currentSlice, shapeId, { points: finalPoints, radius, mode });
  };

  /** Tool-dispatched press: starts a marquee, seeds magic/magnetic, adds a polygon
   *  vertex, begins a rect/ellipse drag, or starts a buffered brush/erase stroke. */
  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPreviewing) return;

    // Select tool: pressing on empty canvas begins a marquee (rubber-band).
    // Whether it becomes a box-select or a deselect is decided on mouse-up.
    // Clicks on a shape are handled by that shape's own onClick (selectShape).
    if (tool === 'select') {
      if (e.target === e.target.getStage()) {
        const p = getPointerImagePos();
        if (p) {
          setMarqueeStart(p);
          setMarqueeRect(null);
          // Shift-drag accumulates onto the existing selection (decided on mouse-up).
          marqueeShiftRef.current = e.evt.shiftKey;
        }
      }
      return;
    }

    if (!sourceKey || !meta || activeClassId === null) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    if (tool === 'polygon') {
      addPolyNode(sourceKey, String(currentSlice), pos.x, pos.y);
    } else if (tool === 'magic') {
      // Seed the magic selection (the effect computes the preview).
      // SAM engine:
      //  • plain click → new positive point (start the object)
      //  • Shift-click → add a positive point (extend the object)
      //  • Alt/Option-click → add a negative point (carve a leak back out)
      // Classic wand keeps its original meaning (Alt removes the nearest seed).
      if (magicEngine === 'sam') {
        // Shift/Alt-click refine with points; a plain press starts a box drag
        // (resolved to a box prompt or a single point on mouse-up).
        if (e.evt.altKey) {
          setMagicSeeds((prev) => (prev.length === 0 && !magicBox ? prev : [...prev, { x: pos.x, y: pos.y, label: 0 }]));
        } else if (e.evt.shiftKey) {
          setMagicSeeds((prev) => [...prev, { x: pos.x, y: pos.y, label: 1 }]);
        } else {
          magicDragStartRef.current = { x: pos.x, y: pos.y };
          setMagicBoxDraft(null);
        }
      } else if (e.evt.altKey) {
        setMagicSeeds((prev) => {
          if (prev.length === 0) return prev;
          let bestI = 0, bestD = Infinity;
          prev.forEach((s, i) => {
            const d = (s.x - pos.x) ** 2 + (s.y - pos.y) ** 2;
            if (d < bestD) { bestD = d; bestI = i; }
          });
          return prev.filter((_, i) => i !== bestI);
        });
      } else if (e.evt.shiftKey) {
        setMagicSeeds((prev) => [...prev, { x: pos.x, y: pos.y, label: 1 }]);
      } else {
        setMagicSeeds([{ x: pos.x, y: pos.y, label: 1 }]);
      }
    } else if (tool === 'magnetic') {
      const cm = ensureCostMap();
      // Lock in the least-cost path from the previous seed to this click (or a straight
      // segment on the first click / no edge map). Each node is one tracked undo step;
      // the dijkstra map for the new seed is recomputed by the layout-effect above.
      const pathPoints = cm && magneticSeed && magneticPrevRef.current
        ? tracePath(cm, magneticPrevRef.current, imageToGrid(cm, pos.x, pos.y)).slice(2)
        : [pos.x, pos.y];
      addMagneticNode(sourceKey, String(currentSlice), pathPoints, pos);
      setMagneticPreview([]);
    } else if (tool === 'fill') {
      // Paint bucket: a plain click seeds a single contiguous flood (within
      // `fillThreshold`); Shift-click adds another region; Alt-click removes the
      // nearest seed. The region(s) preview, then Add/Enter commits (clip-aware).
      if (e.evt.altKey) {
        setMagicSeeds((prev) => {
          if (prev.length === 0) return prev;
          let bestI = 0, bestD = Infinity;
          prev.forEach((s, i) => {
            const d = (s.x - pos.x) ** 2 + (s.y - pos.y) ** 2;
            if (d < bestD) { bestD = d; bestI = i; }
          });
          return prev.filter((_, i) => i !== bestI);
        });
      } else if (e.evt.shiftKey) {
        setMagicSeeds((prev) => [...prev, { x: pos.x, y: pos.y, label: 1 }]);
      } else {
        // Plain click replaces the current selection with this single region.
        setMagicSeeds([{ x: pos.x, y: pos.y, label: 1 }]);
      }
    } else if (tool === 'rectangle' || tool === 'ellipse') {
      setDragStart(pos);
      setDragCurrent(pos);
    } else if (tool === 'brush') {
      setIsDrawing(true);

      // Determine which brush shape to append to
      const sliceShapes = byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const existingBrush = activeBrushShapeId
        ? sliceShapes.find((s) => s.id === activeBrushShapeId && s.kind === 'brush')
        : null;
      const canAppend = existingBrush && existingBrush.classId === activeClassId;

      let shapeId: string;
      if (canAppend && activeBrushShapeId) {
        shapeId = activeBrushShapeId;
      } else {
        shapeId = uuidv4();
        // Create container with empty strokes; stroke appended on mouseup
        addShape(sourceKey, currentSlice, {
          id: shapeId,
          classId: activeClassId,
          kind: 'brush',
          strokes: [],
        });
        onNewBrushInstance(shapeId);
      }

      draftStrokeRef.current = { shapeId, mode: 'paint', points: [pos.x, pos.y], radius: brushSize };

      // Prime the draft line for imperative updates
      if (draftLineRef.current) {
        draftLineRef.current.stroke(activeColor);
        draftLineRef.current.strokeWidth(brushSize * 2);
        draftLineRef.current.points([pos.x, pos.y]);
        draftLineRef.current.visible(true);
        draftStrokeLayerRef.current?.batchDraw();
      }
    } else if (tool === 'sampler') {
      // Magnetic lasso: seed the live-wire here, then snap each dragged segment
      // to the strongest edge between anchors. No store writes and no undo entry
      // — this measures, it does not annotate.
      setIsDrawing(true);
      const cm = ensureCostMap();
      samplerLassoRef.current = {
        committed: [pos.x, pos.y],
        anchor: pos,
        start: pos,
        cm,
        prev: cm ? dijkstra(cm, imageToGrid(cm, pos.x, pos.y)) : null,
      };
      if (draftLineRef.current) {
        draftLineRef.current.stroke('#38bdf8');
        draftLineRef.current.strokeWidth(2 / transform.scaleX);
        draftLineRef.current.points([pos.x, pos.y]);
        draftLineRef.current.closed(false);
        draftLineRef.current.visible(true);
        draftStrokeLayerRef.current?.batchDraw();
      }
    } else if (tool === 'threshold') {
      // Shift-click samples instead of painting: re-center the band on the pixel
      // under the cursor, keeping the configured sample width.
      if (e.evt.shiftKey) {
        // Sample from the GATE field so the picked value is in the same units as
        // the band — with a projection active those are score units, not greys.
        const field = ensureGateField();
        if (field) {
          const gx = Math.max(0, Math.min(field.gw - 1, Math.floor(pos.x / field.scale)));
          const gy = Math.max(0, Math.min(field.gh - 1, Math.floor(pos.y / field.scale)));
          const v = field.gray[gy * field.gw + gx];
          const half = thresholdSampleWidth / 2;
          setThresholdBand(v - half, v + half);
        }
        return;
      }
      setIsDrawing(true);
      startThresholdStroke(pos, e.evt.altKey);
    } else if (tool === 'eraser') {
      setIsDrawing(true);
      const sliceShapes = byImage[sourceKey]?.[String(currentSlice)] ?? [];
      // Erase carves only from the topmost shape of the ACTIVE class under the
      // cursor, so it never eats into another layer. Falls back to the active
      // brush instance if the click isn't over a shape. If nothing is under the
      // cursor we still start the stroke (so the preview shows and the user can
      // drag ONTO a shape) — the target is resolved from the whole stroke on
      // commit (see commitDraftStroke).
      const visible = isShapeVisible;
      // "Erase all classes" ignores the active-class restriction (any visible shape).
      const inScope = (s: Shape) => eraseAllClasses || s.classId === activeClassId;
      let target: Shape | undefined;
      for (let i = sliceShapes.length - 1; i >= 0; i--) {
        const s = sliceShapes[i];
        // Radius-aware: the brush disk grazing the shape counts, not just its center.
        if (inScope(s) && visible(s) && shapeNearPoint(s, pos.x, pos.y, brushSize)) { target = s; break; }
      }
      if (!target && activeBrushShapeId) {
        target = sliceShapes.find((s) => s.id === activeBrushShapeId && s.kind === 'brush');
      }
      draftStrokeRef.current = {
        shapeId: target?.id ?? '',
        mode: 'erase',
        points: [pos.x, pos.y],
        radius: brushSize,
        eraseTargetKind: target && target.kind !== 'brush' ? 'vector' : 'brush',
      };
      // Show a white dash to indicate erasing.
      if (draftLineRef.current) {
        draftLineRef.current.stroke('#ffffff');
        draftLineRef.current.strokeWidth(brushSize * 2);
        draftLineRef.current.points([pos.x, pos.y]);
        draftLineRef.current.visible(true);
        draftStrokeLayerRef.current?.batchDraw();
      }
    }
  };

  /** Drives all live previews while dragging: brush cursor, marquee, SAM box,
   *  magnetic path, rect/ellipse draft, and buffering brush/erase points. */
  const handleStageMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPreviewing) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    // Update brush cursor position imperatively — no React state, no re-render
    if (showBrushCursor && brushCursorRef.current) {
      brushCursorRef.current.position({ x: pos.x, y: pos.y });
      brushCursorLayerRef.current?.batchDraw();
    }
    if (!pointerInside) setPointerInside(true);

    // Marquee rubber-band: update the box while dragging on empty canvas.
    if (tool === 'select' && marqueeStart && e.evt.buttons === 1) {
      setMarqueeRect(normalizeRect(marqueeStart.x, marqueeStart.y, pos.x - marqueeStart.x, pos.y - marqueeStart.y));
      return;
    }

    if (!sourceKey || !meta) return;

    // SAM box prompt: live rubber-band while dragging.
    if (tool === 'magic' && magicEngine === 'sam' && magicDragStartRef.current && e.evt.buttons === 1) {
      const s = magicDragStartRef.current;
      setMagicBoxDraft(normalizeRect(s.x, s.y, pos.x - s.x, pos.y - s.y));
      return;
    }

    // Magnetic lasso: live least-cost path from the seed to the cursor.
    if (tool === 'magnetic' && magneticSeed) {
      const cm = magneticCostRef.current;
      if (cm && magneticPrevRef.current) {
        setMagneticPreview(tracePath(cm, magneticPrevRef.current, imageToGrid(cm, pos.x, pos.y)));
      } else {
        setMagneticPreview([magneticSeed.x, magneticSeed.y, pos.x, pos.y]);
      }
      return;
    }

    if ((tool === 'rectangle' || tool === 'ellipse') && dragStart && e.evt.buttons === 1) {
      setDragCurrent(pos);
      return;
    }

    // Sampler lasso: preview the snapped segment from the anchor to the cursor,
    // dropping a new anchor once the cursor has travelled far enough. Anchoring
    // periodically (rather than per pixel) is what keeps the live-wire honest —
    // one dijkstra per anchor, and the locked-in path stops re-flowing behind you.
    if (tool === 'sampler' && e.evt.buttons === 1 && samplerLassoRef.current) {
      const st = samplerLassoRef.current;
      const traced = samplerTraceTo(pos);
      if (draftLineRef.current) {
        draftLineRef.current.points([...st.committed, ...traced]);
        draftStrokeLayerRef.current?.batchDraw();
      }
      const dx = pos.x - st.anchor.x;
      const dy = pos.y - st.anchor.y;
      if (dx * dx + dy * dy >= SAMPLER_ANCHOR_STEP * SAMPLER_ANCHOR_STEP) {
        st.committed = [...st.committed, ...traced];
        samplerReseed(pos);
      }
      return;
    }

    // Threshold brush: stamp into the in-band mask and repaint the preview. Like
    // the brush, this writes nothing to the store until mouseup.
    if (tool === 'threshold' && e.evt.buttons === 1 && thresholdStrokeRef.current) {
      extendThresholdStroke(pos);
      return;
    }

    // Buffer brush/eraser points — no store writes here
    if ((tool === 'brush' || tool === 'eraser') && e.evt.buttons === 1 && draftStrokeRef.current) {
      draftStrokeRef.current.points.push(pos.x, pos.y);
      if (draftLineRef.current) {
        // Passing the array directly avoids an extra copy; Konva reads it synchronously
        draftLineRef.current.points(draftStrokeRef.current.points.slice());
        draftStrokeLayerRef.current?.batchDraw();
      }
    }
  };

  /** Finalizes the press: box-select/deselect marquee, resolve SAM box-vs-point,
   *  flush a brush/erase stroke, or commit a dragged rect/ellipse. */
  const handleStageMouseUp = () => {
    if (isPreviewing) return;

    // Finish a marquee. A real drag box-selects intersecting shapes; with Shift
    // those are added to the current selection (drag several boxes to build it
    // up). A bare non-Shift click clears; a bare Shift-click keeps the selection.
    if (tool === 'select' && marqueeStart) {
      const rect = marqueeRect;
      const additive = marqueeShiftRef.current;
      if (rect && rect.w > 3 && rect.h > 3 && sourceKey) {
        const ids = storeShapes
          .filter(isShapeVisible)
          .filter((s) => shapeIntersectsRect(s, rect))
          .map((s) => s.id);
        setSelectedShapeIds(additive ? Array.from(new Set([...selectedShapeIds, ...ids])) : ids);
      } else if (!additive) {
        setSelectedShapeIds([]);
      }
      setMarqueeStart(null);
      setMarqueeRect(null);
      marqueeShiftRef.current = false;
      return;
    }

    // SAM magic: resolve the press into a box prompt (real drag) or a point.
    if (tool === 'magic' && magicEngine === 'sam' && magicDragStartRef.current) {
      const start = magicDragStartRef.current;
      magicDragStartRef.current = null;
      const up = getPointerImagePos() ?? start;
      const box = normalizeRect(start.x, start.y, up.x - start.x, up.y - start.y);
      setMagicBoxDraft(null);
      if (box.w > 4 && box.h > 4) {
        setMagicBox(box);
        setMagicSeeds([]); // a box replaces any accumulated points
      } else {
        setMagicSeeds([{ x: start.x, y: start.y, label: 1 }]);
        setMagicBox(null);
      }
      return;
    }

    if (isDrawing) {
      commitDraftStroke();
      commitThresholdStroke();
      finishSamplerLasso();
      setIsDrawing(false);
    }
    if (!sourceKey || !meta || activeClassId === null) return;

    if ((tool === 'rectangle' || tool === 'ellipse') && dragStart && dragCurrent) {
      const dx = dragCurrent.x - dragStart.x;
      const dy = dragCurrent.y - dragStart.y;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        const id = uuidv4();
        if (tool === 'rectangle') {
          const { x, y, w, h } = normalizeRect(dragStart.x, dragStart.y, dx, dy);
          commitShapes([{ id, classId: activeClassId, kind: 'rectangle', x, y, w, h }]);
        } else {
          const { cx, cy, rx, ry } = normalizeEllipse(
            (dragStart.x + dragCurrent.x) / 2,
            (dragStart.y + dragCurrent.y) / 2,
            Math.abs(dx) / 2,
            Math.abs(dy) / 2
          );
          commitShapes([{ id, classId: activeClassId, kind: 'ellipse', cx, cy, rx, ry }]);
        }
      }
      setDragStart(null);
      setDragCurrent(null);
    }
  };

  /** Closes the in-progress polygon, or finishes & simplifies a magnetic trace, into a
   *  committed shape. The commit + draft-clear happen in ONE store step (commitDraftShapes)
   *  so undo reopens the shape to edit mode and redo re-closes it. */
  const handleStageDblClick = () => {
    if (isPreviewing) return;
    if (tool === 'polygon' && draftPoly.length >= 6 && sourceKey && activeClassId !== null) {
      const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const shape: Shape = { id: uuidv4(), classId: activeClassId, kind: 'polygon', points: draftPoly };
      commitDraftShapes(sourceKey, currentSlice, computeCommittedSlice([shape], sliceShapes));
    } else if (tool === 'magnetic' && sourceKey && activeClassId !== null) {
      // Commit the final hovered segment, then close the traced polygon.
      let pts = magneticCommitted;
      const cm = magneticCostRef.current;
      const pos = getPointerImagePos();
      if (cm && magneticPrevRef.current && pos) {
        pts = [...pts, ...tracePath(cm, magneticPrevRef.current, imageToGrid(cm, pos.x, pos.y)).slice(2)];
      }
      const simplified = simplifyPath(pts, 3);
      if (simplified.length >= 6) {
        const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
        const shape: Shape = { id: uuidv4(), classId: activeClassId, kind: 'polygon', points: simplified };
        commitDraftShapes(sourceKey, currentSlice, computeCommittedSlice([shape], sliceShapes));
      } else {
        clearDraft(); // trace too small to commit — discard (undo can restore it)
      }
      resetMagnetic();
    }
  };

  /** Hides the brush cursor and commits any in-progress stroke/drag on exit. */
  const handleStageMouseLeave = () => {
    setPointerInside(false); // hides the cursor preview (reactive visible prop)
    // Commit any in-progress stroke
    if (isDrawing) {
      commitDraftStroke();
      commitThresholdStroke();
      finishSamplerLasso();
      setIsDrawing(false);
    }
    setDragStart(null);
    setDragCurrent(null);
  };

  /** Re-shows the brush/eraser size cursor when the pointer re-enters the stage. */
  const handleStageMouseEnter = () => {
    setPointerInside(true);
    // Snap the preview to the current pointer so it doesn't flash at a stale spot.
    const pos = getPointerImagePos();
    if (showBrushCursor && pos && brushCursorRef.current) {
      brushCursorRef.current.position({ x: pos.x, y: pos.y });
      brushCursorLayerRef.current?.batchDraw();
    }
  };

  // When switching TO the brush/eraser while the pointer is already over the canvas
  // (no mouse-enter fires), snap the preview to the current pointer position so it
  // appears immediately rather than only after the next move.
  useEffect(() => {
    if (!showBrushCursor) return;
    const pos = getPointerImagePos();
    if (pos && brushCursorRef.current) {
      brushCursorRef.current.position({ x: pos.x, y: pos.y });
      brushCursorLayerRef.current?.batchDraw();
    }
  }, [showBrushCursor]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Zoom in/out toward the cursor, keeping the point under the pointer fixed. */
  // Zoom is applied at most once per animation frame. A trackpad or free-spin
  // wheel emits events far faster than the display refreshes, and each one used to
  // trigger a full React render (and potentially a layer re-cache). Accumulating
  // into a ref and flushing on rAF collapses a burst into a single update, with
  // the same final scale and focal point.
  const pendingZoomRef = useRef<{ steps: number; pointer: { x: number; y: number } } | null>(null);
  const zoomRafRef = useRef<number | null>(null);
  useEffect(() => () => { if (zoomRafRef.current != null) cancelAnimationFrame(zoomRafRef.current); }, []);

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const pending = pendingZoomRef.current;
    const steps = (pending?.steps ?? 0) + (e.evt.deltaY < 0 ? 1 : -1);
    pendingZoomRef.current = { steps, pointer };
    if (zoomRafRef.current != null) return;

    zoomRafRef.current = requestAnimationFrame(() => {
      zoomRafRef.current = null;
      const job = pendingZoomRef.current;
      pendingZoomRef.current = null;
      if (!job || job.steps === 0) return;
      setTransform((t) => {
        const newScale = t.scaleX * Math.pow(1.1, job.steps);
        const mousePointTo = {
          x: (job.pointer.x - t.x) / t.scaleX,
          y: (job.pointer.y - t.y) / t.scaleY,
        };
        return {
          scaleX: newScale, scaleY: newScale,
          x: job.pointer.x - mousePointTo.x * newScale,
          y: job.pointer.y - mousePointTo.y * newScale,
        };
      });
    });
  };

  /** Live (uncommitted) rect/ellipse preview while dragging it out. */
  const renderDraftShape = () => {
    if (!dragStart || !dragCurrent) return null;
    const dx = dragCurrent.x - dragStart.x;
    const dy = dragCurrent.y - dragStart.y;
    if (tool === 'rectangle') {
      const { x, y, w, h } = normalizeRect(dragStart.x, dragStart.y, dx, dy);
      return (
        <Rect
          x={x} y={y} width={w} height={h}
          fill={activeColor} stroke={activeColor} strokeWidth={1 / transform.scaleX}
          listening={false} perfectDrawEnabled={false}
        />
      );
    }
    if (tool === 'ellipse') {
      return (
        <Ellipse
          x={(dragStart.x + dragCurrent.x) / 2}
          y={(dragStart.y + dragCurrent.y) / 2}
          radiusX={Math.abs(dx) / 2}
          radiusY={Math.abs(dy) / 2}
          fill={activeColor} stroke={activeColor} strokeWidth={1 / transform.scaleX}
          listening={false} perfectDrawEnabled={false}
        />
      );
    }
    return null;
  };


  // ----- Interactive selection / move / vertex-edit (select tool only) -----

  /** Select a clicked shape; shift-click toggles it in the multi-selection. */
  const selectShape = (e: Konva.KonvaEventObject<MouseEvent>, id: string) => {
    e.cancelBubble = true;
    // Shift-click toggles the literally-clicked shape in/out of the selection.
    if (e.evt.shiftKey) {
      setSelectedShapeIds(
        selectedShapeIds.includes(id)
          ? selectedShapeIds.filter((sid) => sid !== id)
          : [...selectedShapeIds, id],
      );
      return;
    }
    // Plain click: prefer a shape of the ACTIVE class under the pointer, so an
    // overlapping other-class shape drawn on top doesn't block selecting yours.
    let targetId = id;
    const pos = getPointerImagePos();
    if (pos && activeClassId !== null) {
      const isVisible = (s: Shape) =>
        isShapeVisible(s);
      let topActive: string | null = null;
      for (const s of storeShapes) {
        // Later in draw order = rendered on top → keep the last (topmost) match.
        if (s.classId === activeClassId && isVisible(s) && shapeContainsPoint(s, pos.x, pos.y)) {
          topActive = s.id;
        }
      }
      if (topActive) targetId = topActive;
    }
    setSelectedShapeId(targetId);
  };

  /** Render a shape as an interactive, draggable hit target with a selection outline. */
  const renderInteractive = (shape: Shape) => {
    if (!sourceKey) return null;
    const selected = selectedShapeIds.includes(shape.id);
    // Move/resize/vertex editing require exactly one selected shape.
    const single = selectedId === shape.id;
    const baseColor = colorForClass(shape.classId);
    const outline = selected ? '#ffffff' : baseColor;
    const sw = (selected ? 2.5 : 1.5) / transform.scaleX;
    const dash = selected ? [6 / transform.scaleX, 4 / transform.scaleX] : undefined;
    // Near-transparent fill still registers hit detection without obscuring Layer 1.
    const hitFill = 'rgba(0,0,0,0.001)';
    const handlers = {
      onClick: (e: Konva.KonvaEventObject<MouseEvent>) => selectShape(e, shape.id),
      onTap: (e: Konva.KonvaEventObject<MouseEvent>) => selectShape(e, shape.id),
    };

    if (shape.kind === 'polygon') {
      const livePoints = editPoly?.id === shape.id ? editPoly.points : shape.points;
      const holes = (editPoly?.id === shape.id && editPoly.holes ? editPoly.holes : shape.holes) ?? [];
      const hasHoles = holes.length > 0;
      return (
        <Group
          key={shape.id}
          draggable={single}
          {...handlers}
          onDblClick={(e) => {
            // Double-click the body/edge of the selected polygon → insert a vertex
            // on the nearest edge (outer ring or a hole). Vertex circles handle
            // their own double-click (delete) and stop this from firing.
            if (!single) return;
            const pos = getPointerImagePos();
            if (!pos) return;
            e.cancelBubble = true;
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'polygon' ? insertVertexNearest(s, pos.x, pos.y) : s,
            );
            setEditPoly(null);
          }}
          onDragEnd={(e) => {
            // Only the Group itself moving = a whole-shape move; vertex-circle
            // drags set their own target and are handled below.
            if (e.target !== e.currentTarget) return;
            const dx = e.target.x();
            const dy = e.target.y();
            e.target.position({ x: 0, y: 0 });
            if (dx === 0 && dy === 0) return;
            const shift = (r: number[]) => r.map((v, i) => (i % 2 === 0 ? v + dx : v + dy));
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'polygon'
                ? { ...s, points: shift(s.points), holes: s.holes?.map(shift) }
                : s,
            );
          }}
        >
          {hasHoles ? (
            // Even-odd hit so the hole is click-through (the enclosed class stays
            // selectable) and the outline traces both the outer ring and holes.
            <KonvaShape
              stroke={outline}
              strokeWidth={sw}
              dash={dash}
              sceneFunc={(ctx: Konva.Context, node: Konva.Shape) => {
                buildRingsPath(ctx, [livePoints, ...holes]);
                const raw = (ctx as unknown as { _context: CanvasRenderingContext2D })._context;
                raw.fillStyle = 'rgba(0,0,0,0.001)';
                raw.fill('evenodd');
                ctx.strokeShape(node);
              }}
              hitFunc={(ctx: Konva.Context, node: Konva.Shape) => {
                buildRingsPath(ctx, [livePoints, ...holes]);
                const raw = (ctx as unknown as { _context: CanvasRenderingContext2D })._context;
                raw.fillStyle = (node as unknown as { colorKey: string }).colorKey;
                raw.fill('evenodd');
              }}
            />
          ) : (
            <Line points={livePoints} closed fill={hitFill} stroke={outline} strokeWidth={sw} dash={dash} />
          )}
          {single &&
            Array.from({ length: livePoints.length / 2 }, (_, vi) => (
              <Circle
                key={vi}
                x={livePoints[vi * 2]}
                y={livePoints[vi * 2 + 1]}
                radius={5 / transform.scaleX}
                fill="#ffffff"
                stroke="#1e293b"
                strokeWidth={1 / transform.scaleX}
                draggable
                onClick={(e) => { e.cancelBubble = true; }}
                onDblClick={(e) => {
                  // Delete this outer vertex (keep at least a triangle).
                  e.cancelBubble = true;
                  if (livePoints.length / 2 <= 3) return;
                  updateShape(sourceKey, currentSlice, shape.id, (s) => {
                    if (s.kind !== 'polygon') return s;
                    const np = s.points.slice();
                    np.splice(vi * 2, 2);
                    return { ...s, points: np };
                  });
                  setEditPoly(null);
                }}
                onDragMove={(e) => {
                  const np = [...livePoints];
                  np[vi * 2] = e.target.x();
                  np[vi * 2 + 1] = e.target.y();
                  setEditPoly({ id: shape.id, points: np });
                }}
                onDragEnd={(e) => {
                  const np = [...livePoints];
                  np[vi * 2] = e.target.x();
                  np[vi * 2 + 1] = e.target.y();
                  updateShape(sourceKey, currentSlice, shape.id, (s) =>
                    s.kind === 'polygon' ? { ...s, points: np } : s,
                  );
                  setEditPoly(null);
                }}
              />
            ))}
          {/* Hole (inner-ring) vertices — amber to distinguish from the outer ring. */}
          {single && holes.map((ring, hi) =>
            Array.from({ length: ring.length / 2 }, (_, vi) => (
              <Circle
                key={`h${hi}-${vi}`}
                x={ring[vi * 2]}
                y={ring[vi * 2 + 1]}
                radius={5 / transform.scaleX}
                fill="#fbbf24"
                stroke="#1e293b"
                strokeWidth={1 / transform.scaleX}
                draggable
                onClick={(e) => { e.cancelBubble = true; }}
                onDblClick={(e) => {
                  // Delete this hole vertex; if the hole would become degenerate
                  // (< 3 points), drop the whole hole.
                  e.cancelBubble = true;
                  updateShape(sourceKey, currentSlice, shape.id, (s) => {
                    if (s.kind !== 'polygon' || !s.holes) return s;
                    const hc = s.holes.map((r) => r.slice());
                    if (hc[hi].length / 2 <= 3) hc.splice(hi, 1);
                    else hc[hi].splice(vi * 2, 2);
                    return { ...s, holes: hc.length ? hc : undefined };
                  });
                  setEditPoly(null);
                }}
                onDragMove={(e) => {
                  // Live-update the dashed outline as the hole vertex moves.
                  const hc = holes.map((r) => r.slice());
                  hc[hi][vi * 2] = e.target.x();
                  hc[hi][vi * 2 + 1] = e.target.y();
                  setEditPoly({ id: shape.id, points: livePoints, holes: hc });
                }}
                onDragEnd={(e) => {
                  const nx = e.target.x(), ny = e.target.y();
                  updateShape(sourceKey, currentSlice, shape.id, (s) => {
                    if (s.kind !== 'polygon' || !s.holes) return s;
                    const hc = s.holes.map((r) => r.slice());
                    hc[hi][vi * 2] = nx;
                    hc[hi][vi * 2 + 1] = ny;
                    return { ...s, holes: hc };
                  });
                  setEditPoly(null);
                }}
              />
            )),
          )}
        </Group>
      );
    }

    if (shape.kind === 'rectangle') {
      return (
        <Rect
          key={shape.id}
          id={shape.id}
          x={shape.x} y={shape.y} width={shape.w} height={shape.h}
          fill={hitFill} stroke={outline} strokeWidth={sw} dash={dash}
          draggable={single}
          {...handlers}
          onDragEnd={(e) => {
            const nx = e.target.x();
            const ny = e.target.y();
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'rectangle' ? { ...s, x: nx, y: ny } : s,
            );
          }}
          onTransformEnd={(e) => {
            // Transformer applies a scale; bake it into width/height and reset.
            const node = e.target as Konva.Rect;
            const nw = Math.max(1, node.width() * node.scaleX());
            const nh = Math.max(1, node.height() * node.scaleY());
            node.scaleX(1); node.scaleY(1);
            node.width(nw); node.height(nh);
            const nx = node.x();
            const ny = node.y();
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'rectangle' ? { ...s, x: nx, y: ny, w: nw, h: nh } : s,
            );
          }}
        />
      );
    }

    if (shape.kind === 'ellipse') {
      return (
        <Ellipse
          key={shape.id}
          id={shape.id}
          x={shape.cx} y={shape.cy} radiusX={shape.rx} radiusY={shape.ry}
          fill={hitFill} stroke={outline} strokeWidth={sw} dash={dash}
          draggable={single}
          {...handlers}
          onDragEnd={(e) => {
            const nx = e.target.x();
            const ny = e.target.y();
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'ellipse' ? { ...s, cx: nx, cy: ny } : s,
            );
          }}
          onTransformEnd={(e) => {
            const node = e.target as Konva.Ellipse;
            const nrx = Math.max(1, node.radiusX() * node.scaleX());
            const nry = Math.max(1, node.radiusY() * node.scaleY());
            node.scaleX(1); node.scaleY(1);
            node.radiusX(nrx); node.radiusY(nry);
            const ncx = node.x();
            const ncy = node.y();
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'ellipse' ? { ...s, cx: ncx, cy: ncy, rx: nrx, ry: nry } : s,
            );
          }}
        />
      );
    }

    if (shape.kind === 'brush') {
      return (
        <Group
          key={shape.id}
          draggable={single}
          {...handlers}
          onDragEnd={(e) => {
            const dx = e.target.x();
            const dy = e.target.y();
            e.target.position({ x: 0, y: 0 });
            if (dx === 0 && dy === 0) return;
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'brush'
                ? {
                    ...s,
                    strokes: s.strokes.map((st) => ({
                      ...st,
                      points: st.points.map((v, i) => (i % 2 === 0 ? v + dx : v + dy)),
                    })),
                  }
                : s,
            );
          }}
        >
          {shape.strokes.map((st, i) => (
            <Line
              key={i}
              points={st.points}
              stroke={selected ? outline : hitFill}
              strokeWidth={selected ? sw : 1}
              hitStrokeWidth={st.radius * 2}
              dash={selected ? dash : undefined}
              lineCap="round"
              lineJoin="round"
              perfectDrawEnabled={false}
            />
          ))}
        </Group>
      );
    }
    return null;
  };

  const showInteractive = tool === 'select' && !isPreviewing;

  // Visible shapes with the single-selected one moved LAST, so its vertex handles
  // (incl. amber hole vertices) sit above any shape enclosed in its holes. Memoized
  // and built with a single partition rather than a comparator sort — this ran on
  // every render of the select tool, and a sort whose only job is to hoist one
  // element doesn't need to compare every pair.
  const interactiveShapes = useMemo(() => {
    if (!showInteractive) return EMPTY_SHAPES;
    const rest: Shape[] = [];
    let selected: Shape | null = null;
    for (const s of storeShapes) {
      if (!isShapeVisible(s)) continue;
      if (s.id === selectedId) selected = s;
      else rest.push(s);
    }
    return selected ? [...rest, selected] : rest;
  }, [showInteractive, storeShapes, isShapeVisible, selectedId]);
  // The single selected shape (resize/transform only applies to one).
  const selectedShape = sourceKey && selectedId
    ? storeShapes.find((s) => s.id === selectedId) ?? null
    : null;

  // Attach the resize Transformer to the selected rect/ellipse (by Konva id).
  const selectedResizable =
    selectedShape?.kind === 'rectangle' || selectedShape?.kind === 'ellipse';
  useEffect(() => {
    const tr = transformerRef.current;
    const stage = stageRef.current;
    if (!tr || !stage) return;
    if (showInteractive && selectedShape && selectedResizable) {
      const node = stage.findOne('#' + selectedShape.id);
      tr.nodes(node ? [node] : []);
    } else {
      tr.nodes([]);
    }
    tr.getLayer()?.batchDraw();
  }, [showInteractive, selectedShape, selectedResizable, displayShapes, transform]);

  /** Remove all currently selected shapes from the store and clear the selection. */
  const handleDeleteSelected = () => {
    if (!sourceKey || selectedShapeIds.length === 0) return;
    removeShapes(sourceKey, currentSlice, selectedShapeIds);
    setSelectedShapeId(null);
    setEditPoly(null);
  };

  /** Reassign every selected shape to *classId* (one undo step). */
  const handleReassignClass = (classId: number) => {
    if (!sourceKey || selectedShapeIds.length === 0) return;
    setClassForShapes(sourceKey, currentSlice, selectedShapeIds, classId);
  };

  // The class shared by all selected shapes, or null if they differ ("mixed").
  const selectedClassIds = new Set(
    storeShapes.filter((s) => selectedShapeIds.includes(s.id)).map((s) => s.classId),
  );
  const commonClassId = selectedClassIds.size === 1 ? [...selectedClassIds][0] : null;

  // ----- Selection editing: copy/paste, invert, region ops, brush thickness -----
  const selectedShapes = useMemo(
    () => storeShapes.filter((s) => selectedShapeIds.includes(s.id)),
    [storeShapes, selectedShapeIds],
  );
  const selectedBrush = selectedShape?.kind === 'brush' ? selectedShape : null;

  const [regionOp, setRegionOp] = useState<RegionOp | null>(null);
  const [regionParam, setRegionParam] = useState(4);

  // For the merge op, absorb overlapping same-class shapes that aren't explicitly
  // selected, so selecting one region merges the whole overlapping same-class cluster.
  const mergeShapes = useMemo(() => {
    if (regionOp !== 'merge' || !meta || selectedShapes.length === 0) return selectedShapes;
    return expandSameClassOverlap(selectedShapes, storeShapes, meta.width, meta.height);
  }, [regionOp, selectedShapes, storeShapes, meta]);

  // Live region-op preview polygons (per class), recomputed as op/param/selection change.
  const regionPreview = useMemo(() => {
    if (!regionOp || !meta) return [];
    const input = regionOp === 'merge' ? mergeShapes : selectedShapes;
    if (input.length === 0) return [];
    return computeRegionOps(input, meta.width, meta.height, regionOp, regionParam);
  }, [regionOp, regionParam, selectedShapes, mergeShapes, meta]);

  // Cancel a pending region op whenever the selection empties or the tool changes.
  useEffect(() => {
    if (selectedShapeIds.length === 0 || tool !== 'select') setRegionOp(null);
  }, [selectedShapeIds, tool]);

  const handleCopy = useCallback(() => {
    if (selectedShapes.length) clipboard.copy(selectedShapes);
  }, [selectedShapes, clipboard]);

  const handlePaste = useCallback(() => {
    if (!sourceKey || clipboard.shapes.length === 0) return;
    const pasted = clipboard.shapes.map((s) => offsetShape({ ...structuredClone(s), id: uuidv4() }, 12, 12));
    const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(currentSlice)] ?? [];
    const toAdd = clipToOtherClasses && meta && pasted.some((s) => hasOtherClass(sliceShapes, s.classId))
      ? clipShapesToOthers(pasted, sliceShapes, meta.width, meta.height)
      : pasted;
    if (toAdd.length === 0) return;
    addShapes(sourceKey, currentSlice, toAdd);
    setSelectedShapeIds(toAdd.map((s) => s.id));
  }, [sourceKey, clipboard.shapes, addShapes, currentSlice, setSelectedShapeIds, clipToOtherClasses, meta]);

  /** Replace the single selected shape with its complement (frame minus shape). */
  const handleInvert = useCallback(() => {
    if (!sourceKey || !meta || !selectedShape) return;
    const { gw, gh, scale } = gridFor(meta.width, meta.height);
    const mask = rasterizeShapes([selectedShape], gw, gh, scale);
    const holes = maskToPolygons(mask, gw, gh, { minRegion: 4, scale });
    const outer = [0, 0, meta.width, 0, meta.width, meta.height, 0, meta.height];
    updateShape(sourceKey, currentSlice, selectedShape.id, (s) => ({
      id: s.id, classId: s.classId, kind: 'polygon', points: outer, holes,
    }));
  }, [sourceKey, meta, selectedShape, updateShape, currentSlice]);

  /** Commit the region-op preview: replace the operated shapes with the result.
   *  Merge also consumes the overlapping same-class shapes it absorbed. */
  const handleApplyRegion = useCallback(() => {
    if (!sourceKey || regionPreview.length === 0) return;
    // Merge operates on the expanded (overlap-absorbed) set; others on the selection.
    const consumedIds = new Set(
      (regionOp === 'merge' ? mergeShapes : selectedShapes).map((s) => s.id),
    );
    const kept = storeShapes.filter((s) => !consumedIds.has(s.id));
    const created: Shape[] = regionPreview.map((r) => ({
      id: uuidv4(), classId: r.classId, kind: 'polygon' as const, points: r.points,
      ...(r.holes && r.holes.length ? { holes: r.holes } : {}),
    }));
    const finalCreated = clipToOtherClasses && meta
      ? clipShapesToOthers(created, kept, meta.width, meta.height)
      : created;
    setShapes(sourceKey, currentSlice, [...kept, ...finalCreated]);
    setSelectedShapeIds(finalCreated.map((s) => s.id));
    setRegionOp(null);
  }, [sourceKey, regionPreview, regionOp, mergeShapes, selectedShapes, storeShapes, setShapes, currentSlice, setSelectedShapeIds, clipToOtherClasses, meta]);

  /** Set every stroke's radius on the selected brush shape (post-draw re-thickness). */
  const handleBrushThickness = useCallback((radius: number) => {
    if (!sourceKey || !selectedBrush) return;
    updateShape(sourceKey, currentSlice, selectedBrush.id, (s) =>
      s.kind === 'brush' ? { ...s, strokes: s.strokes.map((st) => ({ ...st, radius })) } : s,
    );
  }, [sourceKey, selectedBrush, updateShape, currentSlice]);

  // Keyboard: copy/paste/invert (select tool) + Enter to apply a region op.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (tool !== 'select') return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      const mod = e.metaKey || e.ctrlKey;
      const k = e.key.toLowerCase();
      if (mod && k === 'a') {
        // Select all shapes on the slice, scoped to the active class or all classes.
        e.preventDefault();
        const visible = isShapeVisible;
        const ids = storeShapes
          .filter((s) => (selectScope === 'all' || s.classId === activeClassId) && visible(s))
          .map((s) => s.id);
        setSelectedShapeIds(ids);
      }
      else if (mod && k === 'c' && selectedShapes.length) { e.preventDefault(); handleCopy(); }
      else if (mod && k === 'v' && clipboard.shapes.length) { e.preventDefault(); handlePaste(); }
      else if (!mod && k === 'i' && selectedShape) { e.preventDefault(); handleInvert(); }
      else if (e.key === 'Enter' && regionOp && regionPreview.length) { e.preventDefault(); handleApplyRegion(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tool, selectedShapes, clipboard.shapes, selectedShape, regionOp, regionPreview,
      handleCopy, handlePaste, handleInvert, handleApplyRegion,
      storeShapes, classes, activeClassId, selectScope, setSelectedShapeIds]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-gray-900 overflow-hidden"
      style={{ cursor: showBrushCursor ? 'none' : (tool === 'magnetic' || tool === 'magic' || tool === 'fill' || tool === 'sampler') ? 'crosshair' : undefined }}
    >
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        draggable={tool === 'pan'}
        onMouseDown={handleStageMouseDown}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
        onMouseLeave={handleStageMouseLeave}
        onMouseEnter={handleStageMouseEnter}
        onDblClick={handleStageDblClick}
        onWheel={handleWheel}
        x={transform.x}
        y={transform.y}
        scaleX={transform.scaleX}
        scaleY={transform.scaleY}
        onDragEnd={(e) => {
          // Drag events bubble: a vertex/shape drag also fires this. Only the
          // stage itself being dragged (pan) should update the pan transform —
          // otherwise the image jumps to the dragged node's coordinates.
          if (e.target !== e.target.getStage()) return;
          setTransform((t) => ({ ...t, x: e.target.x(), y: e.target.y() }));
        }}
      >
        {/* Layer 0: image — preprocessed base (CLAHE/Sharpen baked); linear
            brightness/contrast/levels/gamma/colormap applied via the GPU SVG filter below. */}
        <Layer ref={imageLayerRef}>
          {imageEl && meta && (
            <KonvaImage
              ref={imageRef}
              image={displayBase ?? imageEl}
              width={meta.width}
              height={meta.height}
              listening={false}
            />
          )}
        </Layer>

        {/* Threshold band overlay (ImageJ-style): every pixel the threshold brush
            is currently allowed to paint, in translucent red. Sits above the image
            but below the annotations so existing regions stay readable. */}
        {showThresholdOverlay && meta && (
          <Layer listening={false} opacity={0.35} {...imageClip}>
            <KonvaImage
              ref={overlayImageRef}
              image={overlayCanvasRef.current ?? undefined}
              width={meta.width}
              height={meta.height}
              listening={false}
              perfectDrawEnabled={false}
            />
          </Layer>
        )}

        {/* Layer 1: committed shapes — cached + opacity applied once at the
            layer so overlapping same-class shapes render a uniform class color.
            Clipped to the image frame so strokes never render past the edges.
            Memoized (see ShapesLayer) so display-slider ticks don't reconcile it. */}
        {meta && (
          <ShapesLayer
            layerRef={shapesLayerRef}
            shapes={displayShapes}
            classMap={renderClassMap}
            fillOpacity={fillOpacity}
            scaleX={transform.scaleX}
            selectedShapeIds={selectedShapeIds}
            activeBrushShapeId={activeBrushShapeId}
            activeClassId={activeClassId}
            imageWidth={meta.width}
            imageHeight={meta.height}
          />
        )}

        {/* Interactive layer: hit targets for select/move/edit (select tool only). */}
        {showInteractive && (
          <Layer>
            {interactiveShapes.map(renderInteractive)}
            <Transformer
              ref={transformerRef}
              rotateEnabled={false}
              flipEnabled={false}
              ignoreStroke
              anchorSize={8}
              anchorStroke="#38bdf8"
              anchorFill="#ffffff"
              borderStroke="#38bdf8"
              boundBoxFunc={(oldBox, newBox) =>
                newBox.width < 5 || newBox.height < 5 ? oldBox : newBox
              }
            />
            {marqueeRect && (
              <Rect
                x={marqueeRect.x}
                y={marqueeRect.y}
                width={marqueeRect.w}
                height={marqueeRect.h}
                fill="rgba(56,189,248,0.12)"
                stroke="#38bdf8"
                strokeWidth={1 / transform.scaleX}
                dash={[4 / transform.scaleX, 3 / transform.scaleX]}
                listening={false}
              />
            )}
          </Layer>
        )}

        {/* Layer 2: drag preview (dimmed to match annotation fill opacity) */}
        <Layer opacity={fillOpacity}>
          {renderDraftShape()}

          {/* SAM box prompt: committed box (solid) + live drag (dashed). */}
          {(tool === 'magic' || tool === 'pan') && (magicBox || magicBoxDraft) && (() => {
            const b = magicBoxDraft ?? magicBox!;
            return (
              <Rect
                x={b.x} y={b.y} width={b.w} height={b.h}
                stroke="#38bdf8"
                strokeWidth={1.5 / transform.scaleX}
                dash={magicBoxDraft ? [6 / transform.scaleX, 4 / transform.scaleX] : undefined}
                listening={false}
                perfectDrawEnabled={false}
              />
            );
          })()}

          {/* Magic-wand preview regions (kept visible while panning). */}
          {(tool === 'magic' || tool === 'fill' || tool === 'pan') &&
            magicPreview.map((pts, i) => (
              <Line
                key={`magic-${i}`}
                points={pts}
                closed
                fill={activeColor}
                stroke={activeColor}
                strokeWidth={1.5 / transform.scaleX}
                dash={[5 / transform.scaleX, 3 / transform.scaleX]}
                listening={false}
                perfectDrawEnabled={false}
              />
            ))}
          {/* Markers for each accumulated magic prompt: white dot = positive
              (include), red ring = negative (exclude, SAM only). */}
          {(tool === 'magic' || tool === 'fill' || tool === 'pan') &&
            magicSeeds.map((s, i) => (
              <Circle
                key={`magic-seed-${i}`}
                x={s.x}
                y={s.y}
                radius={3.5 / transform.scaleX}
                fill={s.label === 0 ? '#ef4444' : '#ffffff'}
                stroke={s.label === 0 ? '#ffffff' : activeColor}
                strokeWidth={1.5 / transform.scaleX}
                listening={false}
              />
            ))}
          {/* Auto "not" anchors from other-class regions: faint hollow red rings,
              distinct from the user's solid-red manual negatives. */}
          {(tool === 'magic' || tool === 'pan') && magicEngine === 'sam' && samAvoidLabeled &&
            autoNegPoints.map((p, i) => (
              <Circle
                key={`auto-neg-${i}`}
                x={p.x}
                y={p.y}
                radius={4 / transform.scaleX}
                stroke="#ef4444"
                strokeWidth={1.25 / transform.scaleX}
                fillEnabled={false}
                opacity={0.65}
                listening={false}
                perfectDrawEnabled={false}
              />
            ))}

          {/* Region-op preview (select tool): dashed outline of the pending result.
              Renders holes (even-odd) so merged interior gaps show as unfilled. */}
          {regionOp && regionPreview.map((r, i) => (
            r.holes && r.holes.length ? (
              <KonvaShape
                key={`region-${i}`}
                stroke={colorForClass(r.classId)}
                strokeWidth={1.5 / transform.scaleX}
                dash={[5 / transform.scaleX, 3 / transform.scaleX]}
                opacity={0.6}
                listening={false}
                perfectDrawEnabled={false}
                sceneFunc={(ctx: Konva.Context, node: Konva.Shape) => {
                  buildRingsPath(ctx, [r.points, ...r.holes!]);
                  const raw = (ctx as unknown as { _context: CanvasRenderingContext2D })._context;
                  raw.fillStyle = colorForClass(r.classId);
                  raw.fill('evenodd');
                  ctx.strokeShape(node);
                }}
              />
            ) : (
              <Line
                key={`region-${i}`}
                points={r.points}
                closed
                fill={colorForClass(r.classId)}
                stroke={colorForClass(r.classId)}
                strokeWidth={1.5 / transform.scaleX}
                dash={[5 / transform.scaleX, 3 / transform.scaleX]}
                opacity={0.6}
                listening={false}
                perfectDrawEnabled={false}
              />
            )
          ))}
        </Layer>

        {/* Layer 2b: in-progress vector guides (polygon draft + magnetic lasso path).
            Rendered at FULL opacity — these are guide lines, not fills, so they must
            stay crisp regardless of the annotation fill opacity. */}
        <Layer listening={false}>
          {draftPoly.length >= 2 && (
            <>
              <Line
                points={draftPoly}
                stroke={activeColor}
                strokeWidth={2.5 / transform.scaleX}
                dash={[5 / transform.scaleX, 3 / transform.scaleX]}
                lineCap="round"
                lineJoin="round"
                listening={false}
                perfectDrawEnabled={false}
              />
              {draftPoly.map((_, i) =>
                i % 2 === 0 ? (
                  <Circle
                    key={i}
                    x={draftPoly[i]} y={draftPoly[i + 1]}
                    radius={4 / transform.scaleX}
                    fill={activeColor}
                    listening={false}
                  />
                ) : null
              )}
            </>
          )}

          {/* Magnetic lasso: committed path (solid) + live edge-traced preview (dashed).
              The committed path is context-gated in the store draft, so it stays visible
              while panning and after an undo/redo restores it — regardless of current tool. */}
          {magneticCommitted.length >= 2 && (
            <Line
              points={magneticCommitted}
              stroke={activeColor}
              strokeWidth={2.5 / transform.scaleX}
              lineCap="round"
              lineJoin="round"
              listening={false}
              perfectDrawEnabled={false}
            />
          )}
          {tool === 'magnetic' && magneticPreview.length >= 2 && (
            <Line
              points={magneticPreview}
              stroke={activeColor}
              strokeWidth={2.5 / transform.scaleX}
              dash={[5 / transform.scaleX, 3 / transform.scaleX]}
              lineCap="round"
              lineJoin="round"
              listening={false}
              perfectDrawEnabled={false}
            />
          )}
          {magneticSeed && (
            <Circle
              x={magneticSeed.x}
              y={magneticSeed.y}
              radius={4 / transform.scaleX}
              fill={activeColor}
              listening={false}
            />
          )}
        </Layer>

        {/* Layer 3: in-progress brush stroke (imperatively updated, no React re-renders per move) */}
        <Layer ref={draftStrokeLayerRef} listening={false} opacity={fillOpacity} {...imageClip}>
          <Line
            ref={draftLineRef}
            points={[]}
            visible={false}
            lineCap="round"
            lineJoin="round"
            perfectDrawEnabled={false}
            listening={false}
          />
          {/* Threshold-brush stroke preview: the in-band mask painted so far,
              drawn at image size (its canvas is at the working resolution). */}
          {meta && (
            <KonvaImage
              ref={thresholdPreviewRef}
              image={undefined}
              width={meta.width}
              height={meta.height}
              visible={false}
              listening={false}
              perfectDrawEnabled={false}
            />
          )}
        </Layer>

        {/* Layer 4: brush/eraser size cursor preview (position updated imperatively) */}
        <Layer ref={brushCursorLayerRef} listening={false}>
          <Circle
            ref={brushCursorRef}
            radius={brushSize}
            visible={showBrushCursor && pointerInside}
            // Translucent fill (via rgba alpha) with a mostly-opaque, same-color
            // border (node opacity ≈ 1) so the outline stays easy to see.
            fill={underlyingTool === 'eraser' ? 'rgba(255,255,255,0.15)' : withAlpha(activeColor, 0.2)}
            opacity={0.95}
            stroke={underlyingTool === 'eraser' ? '#f8fafc' : activeColor}
            strokeWidth={2.5 / transform.scaleX}
            dash={
              underlyingTool === 'eraser'
                ? [5 / transform.scaleX, 4 / transform.scaleX]
                : undefined
            }
            perfectDrawEnabled={false}
          />
        </Layer>

        {/* Focus highlight: flashes the region of an Insights QA flag. */}
        {focusHighlight && (
          <Layer listening={false}>
            <Rect
              x={focusHighlight.x}
              y={focusHighlight.y}
              width={focusHighlight.w}
              height={focusHighlight.h}
              stroke="#f43f5e"
              strokeWidth={3 / transform.scaleX}
              dash={[8 / transform.scaleX, 5 / transform.scaleX]}
              shadowColor="#f43f5e"
              shadowBlur={8 / transform.scaleX}
              perfectDrawEnabled={false}
            />
          </Layer>
        )}
      </Stage>

      {/* Selection toolbar (select tool) */}
      {showInteractive && selectedShapeIds.length > 0 && (
        <div className="absolute top-2 left-2 flex flex-col gap-1.5 bg-slate-800/90 text-slate-100 text-xs px-3 py-1.5 rounded-md shadow-lg max-w-[560px]">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-slate-300">
              {selectedShapeIds.length > 1
                ? `${selectedShapeIds.length} selected`
                : selectedShape?.kind === 'polygon'
                  ? 'Drag a vertex to edit, or drag to move'
                  : 'Drag to move or resize'}
            </span>
            {/* Reassign the selected shape(s) to a different class. */}
            <label className="flex items-center gap-1 text-slate-300">
              Class:
              <select
                value={commonClassId ?? ''}
                onChange={(e) => handleReassignClass(Number(e.target.value))}
                className="bg-slate-700 text-slate-100 text-xs rounded px-1 py-0.5 border border-slate-600 focus:outline-none focus:border-sky-400"
                title="Change class of the selection"
              >
                {commonClassId === null && <option value="" disabled>(mixed)</option>}
                {classes.map((c) => (
                  <option key={c.classId} value={c.classId}>{c.label}</option>
                ))}
              </select>
            </label>
            <button type="button" onClick={handleCopy}
              className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 text-white font-medium" title="Copy (⌘/Ctrl-C)">
              Copy
            </button>
            <button type="button" onClick={handlePaste} disabled={clipboard.shapes.length === 0}
              className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 disabled:opacity-40 text-white font-medium" title="Paste (⌘/Ctrl-V)">
              Paste
            </button>
            <button type="button" onClick={handleInvert} disabled={!selectedShape}
              className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 disabled:opacity-40 text-white font-medium"
              title="Invert the selected shape (I)">
              Invert
            </button>
            <button type="button" onClick={handleDeleteSelected}
              className="flex items-center gap-1 px-2 py-0.5 rounded bg-red-600 hover:bg-red-500 text-white font-medium" title="Delete selected (Del)">
              <Trash size={13} />
              Delete{selectedShapeIds.length > 1 ? ` (${selectedShapeIds.length})` : ''}
            </button>
          </div>

          {/* Brush thickness (single brush shape) */}
          {selectedBrush && (
            <label className="flex items-center gap-2 text-slate-300">
              Thickness
              <input type="range" min={1} max={200} value={selectedBrush.strokes[0]?.radius ?? brushSize}
                onChange={(e) => handleBrushThickness(Number(e.target.value))}
                className="flex-1 accent-sky-400" />
              <span className="w-7 text-right tabular-nums">{selectedBrush.strokes[0]?.radius ?? brushSize}</span>
            </label>
          )}

          {/* Region ops on the selection (mask-level), with a magic-wand-style confirm. */}
          <div className="flex items-center gap-1 flex-wrap">
            <span className="text-slate-400">Region:</span>
            {(['merge', 'grow', 'shrink', 'islands'] as RegionOp[]).map((op) => (
              <button key={op} type="button"
                onClick={() => { setRegionOp(op); if (op === 'islands') setRegionParam(50); else if (op !== 'merge') setRegionParam(4); }}
                className={`px-2 py-0.5 rounded font-medium ${regionOp === op ? 'bg-sky-600 text-white' : 'bg-slate-600 hover:bg-slate-500 text-white'}`}>
                {op === 'islands' ? 'Remove islands' : op[0].toUpperCase() + op.slice(1)}
              </button>
            ))}
            {regionOp && regionOp !== 'merge' && (
              <label className="flex items-center gap-1 text-slate-300">
                {regionOp === 'islands' ? 'min px²' : 'px'}
                <input type="range"
                  min={1} max={regionOp === 'islands' ? 2000 : 30} step={1}
                  value={regionParam}
                  onChange={(e) => setRegionParam(Number(e.target.value))}
                  className="w-24 accent-sky-400" />
                <span className="w-9 text-right tabular-nums">{regionParam}</span>
              </label>
            )}
            {regionOp && (
              <>
                <button type="button" onClick={handleApplyRegion} disabled={regionPreview.length === 0}
                  className="px-2 py-0.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-medium" title="Apply (Enter)">
                  Apply
                </button>
                <button type="button" onClick={() => setRegionOp(null)}
                  className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 text-white font-medium">
                  Cancel
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {showInteractive && selectedShapeIds.length === 0 && (
        <div className="absolute top-2 left-2 bg-slate-800/80 text-slate-300 text-xs px-3 py-1.5 rounded-md pointer-events-none">
          Click a shape · drag a box to select many · Shift-drag to add more · Shift-click to toggle
        </div>
      )}

      {tool === 'magnetic' && !isPreviewing && (
        <div className="absolute top-2 left-2 bg-slate-800/85 text-slate-200 text-xs px-3 py-1.5 rounded-md pointer-events-none">
          Click along an edge to trace it · double-click to finish
        </div>
      )}

      {/* Magic: hint before a click, Add/Cancel panel after. */}
      {tool === 'magic' && !isPreviewing && magicSeeds.length === 0 && !magicBox && (
        <div className="absolute top-2 left-2 bg-slate-800/85 text-slate-200 text-xs px-3 py-1.5 rounded-md pointer-events-none">
          {magicEngine === 'sam'
            ? sam.status === 'loading-model'
              ? 'Loading SAM model…'
              : sam.status === 'encoding'
                ? 'Encoding slice…'
                : `Drag a box or click an object · Shift-click to add to it · ${REMOVE_KEY_LABEL}-click to exclude a region`
            : `Click a region · Shift-click to add more · ${REMOVE_KEY_LABEL}-click to remove one`}
        </div>
      )}
      {tool === 'magic' && !isPreviewing && (magicSeeds.length > 0 || magicBox) && (
        <div className="absolute top-2 left-2 flex items-center gap-2 bg-slate-800/90 text-slate-100 text-xs px-3 py-1.5 rounded-md shadow-lg">
          <span className="text-slate-300">
            {magicEngine === 'sam' && sam.status === 'loading-model'
              ? 'Loading SAM model…'
              : magicEngine === 'sam' && sam.status === 'encoding'
                ? 'Encoding slice…'
                : magicLoading
                  ? magicEngine === 'sam' ? 'Segmenting…' : 'Selecting…'
                  : magicPreview.length > 0
                    ? magicEngine === 'sam'
                      ? `${magicPreview.length} region${magicPreview.length === 1 ? '' : 's'} · Shift-click adds · ${REMOVE_KEY_LABEL}-click marks "not" (red) to exclude`
                      : `${magicPreview.length} region${magicPreview.length === 1 ? '' : 's'} · shift-add · ${REMOVE_KEY_LABEL.toLowerCase()}-remove`
                    : magicEngine === 'sam'
                      ? 'No object — try another point'
                      : 'No match — raise tolerance'}
          </span>
          <button
            type="button"
            onClick={commitMagic}
            disabled={magicLoading || magicPreview.length === 0}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-medium"
          >
            Add
          </button>
          {magicSeeds.length > 1 && (
            <button
              type="button"
              onClick={() => setMagicSeeds((prev) => prev.slice(0, -1))}
              className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 text-white font-medium"
              title="Remove the last clicked region"
            >
              Undo last
            </button>
          )}
          <button
            type="button"
            onClick={resetMagic}
            className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 text-white font-medium"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Fill: hint before a click, Add/Cancel panel after (preview + confirm). */}
      {tool === 'fill' && !isPreviewing && magicSeeds.length === 0 && (
        <div className="absolute top-2 left-2 bg-slate-800/85 text-slate-200 text-xs px-3 py-1.5 rounded-md pointer-events-none">
          {`Click a region to flood-fill it · Shift-click adds more · ${REMOVE_KEY_LABEL}-click removes one`}
        </div>
      )}
      {tool === 'fill' && !isPreviewing && magicSeeds.length > 0 && (
        <div className="absolute top-2 left-2 flex items-center gap-2 bg-slate-800/90 text-slate-100 text-xs px-3 py-1.5 rounded-md shadow-lg">
          <span className="text-slate-300">
            {magicLoading
              ? 'Filling…'
              : magicPreview.length > 0
                ? `${magicPreview.length} region${magicPreview.length === 1 ? '' : 's'} · Shift-click adds · ${REMOVE_KEY_LABEL}-click removes`
                : 'No match — raise the fill threshold'}
          </span>
          <button
            type="button"
            onClick={commitMagic}
            disabled={magicLoading || magicPreview.length === 0}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-medium"
          >
            Add
          </button>
          {magicSeeds.length > 1 && (
            <button
              type="button"
              onClick={() => setMagicSeeds((prev) => prev.slice(0, -1))}
              className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 text-white font-medium"
              title="Remove the last added region"
            >
              Undo last
            </button>
          )}
          <button
            type="button"
            onClick={resetMagic}
            className="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 text-white font-medium"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Zoom overlay */}
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-0.5 rounded pointer-events-none">
        {Math.round(transform.scaleX * 100)}%
      </div>

      {/* GPU display-adjust filter (brightness/contrast/levels as one linear
          transform). Referenced by the image layer canvas via CSS filter:url(). */}
      <svg width="0" height="0" aria-hidden="true" style={{ position: 'absolute' }}>
        <filter id={displayFilterId} colorInterpolationFilters="sRGB">
          {/* 1. Brightness/contrast/levels as one linear per-channel transform. */}
          <feComponentTransfer>
            <feFuncR type="linear" slope={displayAffine.slope} intercept={displayAffine.intercept} />
            <feFuncG type="linear" slope={displayAffine.slope} intercept={displayAffine.intercept} />
            <feFuncB type="linear" slope={displayAffine.slope} intercept={displayAffine.intercept} />
          </feComponentTransfer>
          {/* 2. Gamma (skip when 1). */}
          {gamma !== 1 && (
            <feComponentTransfer>
              <feFuncR type="gamma" exponent={gamma} />
              <feFuncG type="gamma" exponent={gamma} />
              <feFuncB type="gamma" exponent={gamma} />
            </feComponentTransfer>
          )}
          {/* 3. Colormap LUT — grayscale in (R=G=B), false-color out. */}
          {cmapTables && (
            <feComponentTransfer>
              <feFuncR type="table" tableValues={cmapTables.r.join(' ')} />
              <feFuncG type="table" tableValues={cmapTables.g.join(' ')} />
              <feFuncB type="table" tableValues={cmapTables.b.join(' ')} />
            </feComponentTransfer>
          )}
        </filter>
      </svg>
    </div>
  );
}
