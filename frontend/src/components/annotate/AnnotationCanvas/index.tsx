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
import { useRef, useState, useEffect, useCallback, useMemo, useId } from 'react';
import {
  Stage, Layer, Image as KonvaImage, Line, Rect, Ellipse, Group, Circle, Transformer,
  Shape as KonvaShape,
} from 'react-konva';
import { Trash } from '@phosphor-icons/react';
import type Konva from 'konva';
import { v4 as uuidv4 } from 'uuid';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore, type Shape, type PolygonShape, type BrushStroke, type EraseStroke } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { toImage, normalizeRect, normalizeEllipse } from '@/lib/geometry';
import { buildCostMap, dijkstra, tracePath, imageToGrid, simplifyPath, type CostMap } from '@/lib/livewire';
import { useImageSlice } from '@/hooks/useImageSlice';
import { buildSourceKey } from '@/lib/sourceKey';
import { buildField, magicSelect, maskToPolygons, maskToPolygonsWithHoles, type GrayField } from '@/lib/magicwand';
import { useSam } from '@/hooks/useSam';
import { renderAdjusted, renderPreprocessOnly } from '@/lib/sam/adjust';
import { gridFor, rasterizeShapes, rasterizeUnion } from '@/lib/rasterize';
import { computeRegionOps, type RegionOp } from '@/lib/regionOps';
import { clipShapesToOthers, hasOtherClass } from '@/lib/clipToClasses';
import { mergeNewWithSameClass, expandSameClassOverlap } from '@/lib/mergeSameClass';
import { useClipboardStore } from '@/stores/clipboardStore';
import { colormapTables, type ColormapName } from '@/lib/colormaps';

// macOS labels the Alt key "Option" (⌥). e.altKey is true for it either way,
// so only the on-screen label needs to differ.
const IS_MAC = typeof navigator !== 'undefined' && /mac/i.test(navigator.userAgent);
const REMOVE_KEY_LABEL = IS_MAC ? 'Option' : 'Alt';

/** Shared stable empty shape list — see `storeShapes` for why identity matters. */
const EMPTY_SHAPES: Shape[] = [];

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
  /** Emits the current slice's 256-bin luminance histogram when it loads. */
  onHistogram?: (bins: number[]) => void;
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

interface BBox { x: number; y: number; w: number; h: number; }

/** True if two AABBs overlap. */
function bboxIntersects(a: BBox, b: BBox): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
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
    for (let i = 0; i < p.length; i += 2) if (inRect(p[i], p[i + 1])) return true;
    for (const [x, y] of corners) if (pointInPolygon(x, y, p)) return true;
    for (let i = 0; i < p.length; i += 2) {
      const ax = p[i], ay = p[i + 1];
      const bx = p[(i + 2) % p.length], by = p[(i + 3) % p.length];
      for (let k = 0; k < 4; k++) {
        const [c1x, c1y] = corners[k];
        const [c2x, c2y] = corners[(k + 1) % 4];
        if (segIntersects(ax, ay, bx, by, c1x, c1y, c2x, c2y)) return true;
      }
    }
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
  onHistogram,
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

  const { kind, source, serverUri, meta, currentSlice, renderOpts } = useDatasetStore();
  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;
  const { byImage, addShape, addShapes, appendBrushStroke, appendEraseStroke, updateShape, removeShapes, setShapes, setClassForShapes } = useAnnotationStore();
  const clipboard = useClipboardStore();
  const { tool, brushSize, fillOpacity, selectedShapeIds, setSelectedShapeId, setSelectedShapeIds } = useToolStore();
  // Single-selection id — drives move/resize/vertex editing (those need exactly one).
  const selectedId = selectedShapeIds.length === 1 ? selectedShapeIds[0] : null;
  const magicTolerance = useToolStore((s) => s.magicTolerance);
  const magicMode = useToolStore((s) => s.magicMode);
  const magicSigma = useToolStore((s) => s.magicSigma);
  const magicEdgeStop = useToolStore((s) => s.magicEdgeStop);
  const magicEngine = useToolStore((s) => s.magicEngine);
  const setMagicEngine = useToolStore((s) => s.setMagicEngine);
  const samDetail = useToolStore((s) => s.samDetail);
  const samThreshold = useToolStore((s) => s.samThreshold);
  const samAvoidLabeled = useToolStore((s) => s.samAvoidLabeled);
  const fitRequestId = useToolStore((s) => s.fitRequestId);
  const clipToOtherClasses = useToolStore((s) => s.clipToOtherClasses);
  const mergeOverlappingSameClass = useToolStore((s) => s.mergeOverlappingSameClass);
  const fillThreshold = useToolStore((s) => s.fillThreshold);
  const { classes } = useClassStore();

  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [transform, setTransform] = useState({ scaleX: 1, scaleY: 1, x: 0, y: 0 });

  // Draft polygon vertices (click-vertex mode)
  const [draftPoly, setDraftPoly] = useState<number[]>([]);
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

  // Magnetic lasso (livewire) state. Committed = locked path; preview = live
  // least-cost path from the current seed to the cursor.
  const [magneticCommitted, setMagneticCommitted] = useState<number[]>([]);
  const [magneticPreview, setMagneticPreview] = useState<number[]>([]);
  const magneticCostRef = useRef<CostMap | null>(null);
  const magneticBuiltForRef = useRef<CanvasImageSource | null>(null);
  const magneticPrevRef = useRef<Int32Array | null>(null);
  const magneticSeedRef = useRef<{ x: number; y: number } | null>(null);
  // True while a brush/eraser stroke is actively being drawn — suppresses the
  // (expensive) layer re-cache so we don't rebuild the shapes bitmap mid-stroke.
  const [isDrawing, setIsDrawing] = useState(false);

  const { data: sliceUrl } = useImageSlice(source, kind, currentSlice, renderOpts, serverUri);
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
  const preprocess = useMemo(() => ({ clahe, sharpen }), [clahe, sharpen]);
  const preprocessKey = `${clahe ? 1 : 0}${sharpen ? 1 : 0}`;
  const displayBase = useMemo<CanvasImageSource | null>(() => {
    if (!imageEl || !meta) return imageEl;
    return (clahe || sharpen)
      ? renderPreprocessOnly(imageEl, meta.width, meta.height, preprocess)
      : imageEl;
  }, [imageEl, meta, clahe, sharpen, preprocess]);

  // SAM sees the preprocessed + brightness/contrast/levels-adjusted image
  // (windowing a low-contrast slice greatly helps), so the encode is keyed on
  // those — adjusting them re-encodes. Source building is deferred to a real encode.
  const samEncodeKey = imageEl && meta
    ? `${sourceKey}|${currentSlice}|b${brightness}|c${contrast}|l${levelsLo}-${levelsHi}|pp${preprocessKey}`
    : null;
  /** Lazily render the display-adjusted slice (preprocess → brightness/contrast/levels) that SAM encodes. */
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
    const b255 = brightness * 255;
    const adjust = Math.pow((contrast + 100) / 100, 2);
    const range = Math.max(1, levelsHi - levelsLo);
    const A = adjust * (255 / range);
    const Bconst = (255 / range) * (adjust * b255 + 127.5 * (1 - adjust)) - (255 * levelsLo) / range;
    // The filter is a no-op only when brightness/contrast/levels AND gamma AND
    // colormap are all identity — otherwise it must stay applied.
    const identity =
      brightness === 0 && contrast === 0 && levelsLo <= 0 && levelsHi >= 255 &&
      gamma === 1 && colormap === 'gray';
    return { slope: A, intercept: Bconst / 255, identity };
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
  // the levels control. Runs once per loaded image.
  useEffect(() => {
    if (!imageEl || !onHistogram) return;
    const maxDim = 512;
    const scale = Math.max(1, Math.ceil(Math.max(imageEl.width, imageEl.height) / maxDim));
    const w = Math.max(1, Math.floor(imageEl.width / scale));
    const h = Math.max(1, Math.floor(imageEl.height / scale));
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    ctx.drawImage(imageEl, 0, 0, w, h);
    let data: Uint8ClampedArray;
    try { data = ctx.getImageData(0, 0, w, h).data; } catch { return; }
    const bins = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
      const lum = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) | 0;
      bins[lum < 0 ? 0 : lum > 255 ? 255 : lum]++;
    }
    onHistogram(bins);
  }, [imageEl, onHistogram]);

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
  useEffect(() => {
    const layer = shapesLayerRef.current;
    if (!layer) return;
    if (isDrawing) return;
    layer.clearCache();
    if (displayShapes.length > 0) {
      const pr = Math.min(Math.max(transform.scaleX, 1), 4);
      layer.cache({ pixelRatio: pr });
    }
    layer.batchDraw();
  }, [displayShapes, renderClasses, fillOpacity, meta, transform.scaleX, isDrawing]);

  const colorForClass = (classId: number) =>
    renderClasses.find((c) => c.classId === classId)?.color ?? '#ff0000';

  /** Commit new shapes, clipping them against other classes when the toggle is on
   *  (neighbor classes act as a hard boundary), and merging with overlapping
   *  same-class shapes when that toggle is on. One undo step. */
  const commitShapes = useCallback((newShapes: Shape[]) => {
    if (!sourceKey || newShapes.length === 0) return;
    const slice = useDatasetStore.getState().currentSlice;
    const sliceShapes = useAnnotationStore.getState().byImage[sourceKey]?.[String(slice)] ?? [];
    let toAdd = newShapes;
    if (clipToOtherClasses && meta && newShapes.some((s) => hasOtherClass(sliceShapes, s.classId))) {
      toAdd = clipShapesToOthers(newShapes, sliceShapes, meta.width, meta.height);
    }
    // Auto-merge with overlapping same-class shapes (replaces those + the new shape
    // with one unioned polygon). Runs after clipping so other-class bounds still hold.
    if (mergeOverlappingSameClass && meta && toAdd.length) {
      const { add, removeIds } = mergeNewWithSameClass(toAdd, sliceShapes, meta.width, meta.height);
      if (removeIds.length) {
        // Remove the consumed originals + add the merged result in one undo step.
        const kept = sliceShapes.filter((s) => !removeIds.includes(s.id));
        setShapes(sourceKey, slice, [...kept, ...add]);
        return;
      }
      toAdd = add;
    }
    if (toAdd.length) addShapes(sourceKey, slice, toAdd);
  }, [sourceKey, clipToOtherClasses, mergeOverlappingSameClass, meta, addShapes, setShapes]);

  const activeColor =
    activeClassId !== null ? colorForClass(activeClassId) : '#4090ff';

  const showBrushCursor = (tool === 'brush' || tool === 'eraser') && !!meta && !isPreviewing;

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
    const cm = buildCostMap(base, meta.width, meta.height);
    magneticCostRef.current = cm;
    magneticBuiltForRef.current = base;
    return cm;
  };

  /** Clear the magnetic-lasso trace and its dijkstra/seed refs. */
  const resetMagnetic = useCallback(() => {
    magneticPrevRef.current = null;
    magneticSeedRef.current = null;
    setMagneticCommitted([]);
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
  const ensureMagicField = useCallback((): GrayField | null => {
    if (!imageEl || !meta) return null;
    const key = `${sourceKey}|${currentSlice}|b${brightness}|c${contrast}|l${levelsLo}-${levelsHi}|pp${preprocessKey}`;
    if (magicFieldRef.current && magicFieldForRef.current === key) return magicFieldRef.current;
    const src = renderAdjusted(imageEl, meta.width, meta.height, brightness, contrast, levelsLo, levelsHi, preprocess);
    const f = buildField(src, meta.width, meta.height);
    magicFieldRef.current = f;
    magicFieldForRef.current = key;
    return f;
  }, [imageEl, meta, sourceKey, currentSlice, brightness, contrast, levelsLo, levelsHi, preprocessKey, preprocess]);

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

    const isClassVisible = (s: Shape) =>
      classes.find((c) => c.classId === s.classId)?.isVisible !== false;

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
          setMagicPreview(
            maskToPolygons(res.mask, res.width, res.height, {
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
  }, [tool, magicSeeds, magicBox, autoNegPoints, samDetail, samThreshold, samEncodeKey, makeSamSource, magicEngine, magicTolerance, magicMode, magicSigma, magicEdgeStop, fillThreshold, ensureMagicField, imageEl, meta, sam.ensureEncoded, sam.segment]); // eslint-disable-line react-hooks/exhaustive-deps

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
    resetMagnetic();
    resetMagic();
    setDraftPoly([]);
  }, [tool, resetMagnetic, resetMagic]);

  // A new image/slice always invalidates any in-progress draft.
  useEffect(() => {
    resetMagnetic();
    resetMagic();
    setDraftPoly([]);
  }, [currentSlice, sourceKey, resetMagnetic, resetMagic]);

  // Escape cancels the entire in-progress shape (polygon vertices, magnetic
  // trace, magic selection, or rect/ellipse drag) without committing anything.
  // Enter accepts the current magic-tool selection (same as the "Add" button).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Escape') {
        setDraftPoly([]);
        setDragStart(null);
        setDragCurrent(null);
        setMarqueeStart(null);
        setMarqueeRect(null);
        resetMagnetic();
        resetMagic();
      } else if (e.key === 'Enter' && (tool === 'magic' || tool === 'fill') && !magicLoading && magicPreview.length > 0) {
        e.preventDefault();
        commitMagic();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [resetMagnetic, resetMagic, tool, magicLoading, magicPreview, commitMagic]);

  /** Flush the buffered draft stroke to the Zustand store (one write per stroke). */
  const commitDraftStroke = () => {
    const draft = draftStrokeRef.current;
    if (!draft || !sourceKey) return;
    const { shapeId, mode, points, radius, eraseTargetKind } = draft;
    draftStrokeRef.current = null;

    // Hide the draft line imperatively
    if (draftLineRef.current) {
      draftLineRef.current.visible(false);
      draftStrokeLayerRef.current?.batchDraw();
    }

    if (points.length < 2) return;
    // Duplicate single point so Konva renders it as a dot
    const finalPoints = points.length === 2 ? [...points, ...points] : points;
    if (mode === 'erase' && eraseTargetKind === 'vector') {
      appendEraseStroke(sourceKey, currentSlice, shapeId, { points: finalPoints, radius });
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
        const { gw, gh, scale } = gridFor(meta.width, meta.height);
        const prospective = { ...brush, strokes: [...brush.strokes, { points: finalPoints, radius, mode: 'paint' as const }] };
        const mine = rasterizeShapes([prospective], gw, gh, scale);
        let converted = false;
        const removeIds: string[] = [];

        // Clip: carve out pixels already labeled by other classes.
        if (clipToOtherClasses) {
          const others = sliceShapes.filter((s) => s.classId !== brush.classId);
          if (others.length > 0) {
            const otherMask = rasterizeUnion(others, gw, gh, scale);
            let overlap = false;
            for (let i = 0; i < mine.length; i++) { if (mine[i] && otherMask[i]) { overlap = true; break; } }
            if (overlap) {
              for (let i = 0; i < mine.length; i++) if (otherMask[i]) mine[i] = 0;
              converted = true;
            }
          }
        }

        // Merge: union with overlapping same-class shapes (they get consumed).
        if (mergeOverlappingSameClass) {
          const sameClass = sliceShapes.filter((s) => s.id !== shapeId && s.classId === brush.classId);
          for (const s of sameClass) {
            const sm = rasterizeShapes([s], gw, gh, scale);
            let hit = false;
            for (let i = 0; i < mine.length; i++) { if (mine[i] && sm[i]) { hit = true; break; } }
            if (!hit) continue;
            for (let i = 0; i < mine.length; i++) if (sm[i]) mine[i] = 1;
            removeIds.push(s.id);
            converted = true;
          }
        }

        if (converted) {
          const polys = maskToPolygonsWithHoles(mine, gw, gh, { minRegion: 4, scale })
            .filter((p) => p.points.length >= 6)
            .map((p) => ({
              id: uuidv4(), classId: brush.classId, kind: 'polygon' as const,
              points: p.points, ...(p.holes.length ? { holes: p.holes } : {}),
            }));
          const kept = sliceShapes.filter((s) => s.id !== shapeId && !removeIds.includes(s.id));
          setShapes(sourceKey, currentSlice, [...kept, ...polys]);
          onNewBrushInstance(''); // this brush instance is now polygon(s)
          return;
        }
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
      setDraftPoly((prev) => [...prev, pos.x, pos.y]);
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
      if (cm && magneticSeedRef.current && magneticPrevRef.current) {
        // Lock in the least-cost path from the previous seed to this click.
        const path = tracePath(cm, magneticPrevRef.current, imageToGrid(cm, pos.x, pos.y));
        setMagneticCommitted((prev) => [...prev, ...path.slice(2)]);
      } else {
        // First click (or no edge map) — straight-segment fallback.
        setMagneticCommitted((prev) => [...prev, pos.x, pos.y]);
      }
      // Re-seed at the click point.
      magneticSeedRef.current = pos;
      magneticPrevRef.current = cm ? dijkstra(cm, imageToGrid(cm, pos.x, pos.y)) : null;
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
    } else if (tool === 'eraser') {
      setIsDrawing(true);
      const sliceShapes = byImage[sourceKey]?.[String(currentSlice)] ?? [];
      // Erase carves only from the topmost shape of the ACTIVE class under the
      // cursor, so it never eats into another layer. Falls back to the active
      // brush instance if the click isn't over a shape.
      const visible = (s: Shape) =>
        classes.find((c) => c.classId === s.classId)?.isVisible !== false;
      let target: Shape | undefined;
      for (let i = sliceShapes.length - 1; i >= 0; i--) {
        const s = sliceShapes[i];
        if (s.classId === activeClassId && visible(s) && shapeContainsPoint(s, pos.x, pos.y)) { target = s; break; }
      }
      if (!target && activeBrushShapeId) {
        target = sliceShapes.find((s) => s.id === activeBrushShapeId && s.kind === 'brush');
      }
      if (target) {
        draftStrokeRef.current = {
          shapeId: target.id,
          mode: 'erase',
          points: [pos.x, pos.y],
          radius: brushSize,
          eraseTargetKind: target.kind === 'brush' ? 'brush' : 'vector',
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
    if (tool === 'magnetic' && magneticSeedRef.current) {
      const cm = magneticCostRef.current;
      if (cm && magneticPrevRef.current) {
        setMagneticPreview(tracePath(cm, magneticPrevRef.current, imageToGrid(cm, pos.x, pos.y)));
      } else {
        const seed = magneticSeedRef.current;
        setMagneticPreview([seed.x, seed.y, pos.x, pos.y]);
      }
      return;
    }

    if ((tool === 'rectangle' || tool === 'ellipse') && dragStart && e.evt.buttons === 1) {
      setDragCurrent(pos);
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
          .filter((s) => classes.find((c) => c.classId === s.classId)?.isVisible !== false)
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

  /** Closes the in-progress polygon, or finishes & simplifies a magnetic trace, into a committed shape. */
  const handleStageDblClick = () => {
    if (isPreviewing) return;
    if (tool === 'polygon' && draftPoly.length >= 6 && sourceKey && activeClassId !== null) {
      const id = uuidv4();
      commitShapes([{ id, classId: activeClassId, kind: 'polygon', points: draftPoly }]);
      setDraftPoly([]);
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
        commitShapes([{ id: uuidv4(), classId: activeClassId, kind: 'polygon', points: simplified }]);
      }
      resetMagnetic();
    }
  };

  /** Hides the brush cursor and commits any in-progress stroke/drag on exit. */
  const handleStageMouseLeave = () => {
    // Hide brush cursor
    if (brushCursorRef.current) {
      brushCursorRef.current.visible(false);
      brushCursorLayerRef.current?.batchDraw();
    }
    // Commit any in-progress stroke
    if (isDrawing) {
      commitDraftStroke();
      setIsDrawing(false);
    }
    setDragStart(null);
    setDragCurrent(null);
  };

  /** Re-shows the brush/eraser size cursor when the pointer re-enters the stage. */
  const handleStageMouseEnter = () => {
    if (showBrushCursor && brushCursorRef.current) {
      brushCursorRef.current.visible(true);
      brushCursorLayerRef.current?.batchDraw();
    }
  };

  // Hide brush cursor when switching away from brush/eraser tools
  useEffect(() => {
    if (!showBrushCursor && brushCursorRef.current) {
      brushCursorRef.current.visible(false);
      brushCursorLayerRef.current?.batchDraw();
    }
  }, [showBrushCursor]);

  /** Zoom in/out toward the cursor, keeping the point under the pointer fixed. */
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const scaleBy = 1.1;
    const oldScale = transform.scaleX;
    const pointer = stage.getPointerPosition()!;
    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const mousePointTo = { x: (pointer.x - transform.x) / oldScale, y: (pointer.y - transform.y) / oldScale };
    setTransform({
      scaleX: newScale, scaleY: newScale,
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
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

  /** Destination-out lines that carve erase strokes out of a vector shape. */
  const renderErased = (erased?: EraseStroke[]) =>
    (erased ?? []).map((st, i) => (
      <Line
        key={`erase-${i}`}
        points={st.points}
        stroke="black"
        strokeWidth={st.radius * 2}
        lineCap="round"
        lineJoin="round"
        globalCompositeOperation="destination-out"
        perfectDrawEnabled={false}
        listening={false}
      />
    ));

  /** Render a polygon that may have holes via an even-odd fill (outer path minus
   *  hole subpaths). Even-odd — not destination-out — so a hole reveals whatever
   *  is *beneath* it (e.g. another class) instead of erasing it off the layer. */
  const renderPolygonWithHoles = (points: number[], holes: number[][], color: string, strokeW: number) => (
    <KonvaShape
      stroke={color}
      strokeWidth={strokeW}
      // width/height give the shape a real self-rect so the CACHED committed layer
      // sizes its cache canvas to include it (otherwise it's clipped away when the
      // only other in-bounds shape — the enclosed class — is hidden).
      width={meta?.width ?? 0}
      height={meta?.height ?? 0}
      perfectDrawEnabled={false}
      listening={false}
      sceneFunc={(ctx: Konva.Context, node: Konva.Shape) => {
        buildRingsPath(ctx, [points, ...holes]);
        const raw = (ctx as unknown as { _context: CanvasRenderingContext2D })._context;
        raw.fillStyle = color;
        raw.fill('evenodd');
        ctx.strokeShape(node);
      }}
    />
  );

  /** Render a committed shape (any kind) on the cached display layer, with the
   *  active brush instance recolored to the active class and erase strokes carved out. */
  const renderShape = (shape: Shape) => {
    const color =
      shape.id === activeBrushShapeId && activeClassId !== null
        ? colorForClass(activeClassId)
        : colorForClass(shape.classId);
    const isSelected = selectedShapeIds.includes(shape.id);
    const strokeW = (isSelected ? 2 : 1) / transform.scaleX;

    if (shape.kind === 'polygon') {
      return (
        <Group key={shape.id}>
          {shape.holes?.length
            ? renderPolygonWithHoles(shape.points, shape.holes, color, strokeW)
            : (
              <Line
                points={shape.points}
                closed
                fill={color}
                stroke={color}
                strokeWidth={strokeW}
                perfectDrawEnabled={false}
              />
            )}
          {renderErased(shape.erased)}
        </Group>
      );
    }
    if (shape.kind === 'rectangle') {
      return (
        <Group key={shape.id}>
          <Rect
            x={shape.x} y={shape.y} width={shape.w} height={shape.h}
            fill={color}
            stroke={color} strokeWidth={strokeW}
            perfectDrawEnabled={false}
          />
          {renderErased(shape.erased)}
        </Group>
      );
    }
    if (shape.kind === 'ellipse') {
      return (
        <Group key={shape.id}>
          <Ellipse
            x={shape.cx} y={shape.cy} radiusX={shape.rx} radiusY={shape.ry}
            fill={color}
            stroke={color} strokeWidth={strokeW}
            perfectDrawEnabled={false}
          />
          {renderErased(shape.erased)}
        </Group>
      );
    }
    if (shape.kind === 'brush') {
      return (
        <Group key={shape.id}>
          {shape.strokes.map((stroke, i) => {
            if (stroke.mode === 'erase') {
              return (
                <Line
                  key={i}
                  points={stroke.points}
                  stroke="black"
                  strokeWidth={stroke.radius * 2}
                  lineCap="round"
                  lineJoin="round"
                  globalCompositeOperation="destination-out"
                  perfectDrawEnabled={false}
                  listening={false}
                />
              );
            }
            return (
              <Line
                key={i}
                points={stroke.points}
                stroke={color}
                strokeWidth={stroke.radius * 2}
                lineCap="round"
                lineJoin="round"
                perfectDrawEnabled={false}
                listening={false}
              />
            );
          })}
        </Group>
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
        classes.find((c) => c.classId === s.classId)?.isVisible !== false;
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
      if (mod && k === 'c' && selectedShapes.length) { e.preventDefault(); handleCopy(); }
      else if (mod && k === 'v' && clipboard.shapes.length) { e.preventDefault(); handlePaste(); }
      else if (!mod && k === 'i' && selectedShape) { e.preventDefault(); handleInvert(); }
      else if (e.key === 'Enter' && regionOp && regionPreview.length) { e.preventDefault(); handleApplyRegion(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tool, selectedShapes, clipboard.shapes, selectedShape, regionOp, regionPreview,
      handleCopy, handlePaste, handleInvert, handleApplyRegion]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-gray-900 overflow-hidden"
      style={{ cursor: showBrushCursor ? 'none' : (tool === 'magnetic' || tool === 'magic' || tool === 'fill') ? 'crosshair' : undefined }}
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

        {/* Layer 1: committed shapes — cached + opacity applied once at the
            layer so overlapping same-class shapes render a uniform class color.
            Clipped to the image frame so strokes never render past the edges. */}
        <Layer ref={shapesLayerRef} listening={false} opacity={fillOpacity} {...imageClip}>
          {displayShapes
            .filter((s) => renderClasses.find((c) => c.classId === s.classId)?.isVisible !== false)
            .map(renderShape)}
        </Layer>

        {/* Interactive layer: hit targets for select/move/edit (select tool only). */}
        {showInteractive && (
          <Layer>
            {storeShapes
              .filter((s) => classes.find((c) => c.classId === s.classId)?.isVisible !== false)
              // Render the single-selected shape LAST so its vertex handles (incl.
              // amber hole vertices) sit above any shape enclosed in its holes.
              .sort((a, b) => (a.id === selectedId ? 1 : 0) - (b.id === selectedId ? 1 : 0))
              .map(renderInteractive)}
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

        {/* Layer 2: draft polygon + drag preview */}
        <Layer opacity={fillOpacity}>
          {draftPoly.length >= 2 && (
            <>
              <Line
                points={draftPoly}
                stroke={activeColor}
                strokeWidth={1 / transform.scaleX}
                dash={[4 / transform.scaleX, 2 / transform.scaleX]}
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
          {renderDraftShape()}

          {/* Magnetic lasso: committed path (solid) + live edge-traced preview (dashed).
              Kept visible while panning (tool flips to 'pan') so the trace survives. */}
          {(tool === 'magnetic' || tool === 'pan') && magneticCommitted.length >= 2 && (
            <Line
              points={magneticCommitted}
              stroke={activeColor}
              strokeWidth={1.5 / transform.scaleX}
              listening={false}
              perfectDrawEnabled={false}
            />
          )}
          {tool === 'magnetic' && magneticPreview.length >= 2 && (
            <Line
              points={magneticPreview}
              stroke={activeColor}
              strokeWidth={1.5 / transform.scaleX}
              dash={[5 / transform.scaleX, 3 / transform.scaleX]}
              listening={false}
              perfectDrawEnabled={false}
            />
          )}
          {(tool === 'magnetic' || tool === 'pan') && magneticSeedRef.current && (
            <Circle
              x={magneticSeedRef.current.x}
              y={magneticSeedRef.current.y}
              radius={4 / transform.scaleX}
              fill={activeColor}
              listening={false}
            />
          )}

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
        </Layer>

        {/* Layer 4: brush/eraser size cursor preview (position updated imperatively) */}
        <Layer ref={brushCursorLayerRef} listening={false}>
          <Circle
            ref={brushCursorRef}
            radius={brushSize}
            visible={false}
            fill={tool === 'eraser' ? '#ffffff' : activeColor}
            opacity={tool === 'eraser' ? 0.2 : 0.25}
            stroke={tool === 'eraser' ? '#e2e8f0' : activeColor}
            strokeWidth={2 / transform.scaleX}
            dash={
              tool === 'eraser'
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
