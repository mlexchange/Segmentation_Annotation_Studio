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
import { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import {
  Stage, Layer, Image as KonvaImage, Line, Rect, Ellipse, Group, Circle, Transformer,
} from 'react-konva';
import { Trash } from '@phosphor-icons/react';
import type Konva from 'konva';
import { v4 as uuidv4 } from 'uuid';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore, type Shape, type BrushStroke, type EraseStroke } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { toImage, normalizeRect, normalizeEllipse } from '@/lib/geometry';
import { buildCostMap, dijkstra, tracePath, imageToGrid, simplifyPath, type CostMap } from '@/lib/livewire';
import { useImageSlice } from '@/hooks/useImageSlice';
import { buildSourceKey } from '@/lib/sourceKey';
import { buildField, magicSelect, maskToPolygons, type GrayField } from '@/lib/magicwand';
import { useSam } from '@/hooks/useSam';
import { renderAdjusted } from '@/lib/sam/adjust';

// macOS labels the Alt key "Option" (⌥). e.altKey is true for it either way,
// so only the on-screen label needs to differ.
const IS_MAC = typeof navigator !== 'undefined' && /mac/i.test(navigator.userAgent);
const REMOVE_KEY_LABEL = IS_MAC ? 'Option' : 'Alt';

interface AnnotationCanvasProps {
  brightness: number;
  contrast: number;
  activeClassId: number | null;
  activeBrushShapeId: string | null;
  onNewBrushInstance: (id: string) => void;
  /** When non-null, the canvas renders these shapes read-only (version preview). */
  previewShapes?: Shape[] | null;
  /** Class definitions used for preview shape colors (the version's own classes). */
  previewClasses?: AnnotationClass[] | null;
}

// ---- Point-in-shape hit testing (used by the eraser to pick a target) ----

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
  if (shape.kind === 'polygon') return pointInPolygon(x, y, shape.points);
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
  activeClassId,
  activeBrushShapeId,
  onNewBrushInstance,
  previewShapes = null,
  previewClasses = null,
}: AnnotationCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<Konva.Image>(null);
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
  const { byImage, addShape, addShapes, appendBrushStroke, appendEraseStroke, updateShape, removeShapes, setClassForShapes } = useAnnotationStore();
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
  const { classes } = useClassStore();

  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [transform, setTransform] = useState({ scaleX: 1, scaleY: 1, x: 0, y: 0 });

  // Draft polygon vertices (click-vertex mode)
  const [draftPoly, setDraftPoly] = useState<number[]>([]);
  // Draft rect/ellipse start
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);
  // Live polygon vertex edit (select tool): { id, points } while dragging a handle.
  const [editPoly, setEditPoly] = useState<{ id: string; points: number[] } | null>(null);

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
  const magneticBuiltForRef = useRef<HTMLImageElement | null>(null);
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

  // SAM sees the brightness/contrast-adjusted image (windowing a low-contrast
  // slice greatly helps), so the encode is keyed on those — adjusting them
  // re-encodes. Building the source is deferred so it only runs on a real encode.
  const samEncodeKey = imageEl && meta
    ? `${sourceKey}|${currentSlice}|b${brightness}|c${contrast}`
    : null;
  /** Lazily render the brightness/contrast-adjusted slice that SAM encodes. */
  const makeSamSource = useCallback(
    () => renderAdjusted(imageEl!, meta!.width, meta!.height, brightness, contrast),
    [imageEl, meta, brightness, contrast],
  );

  // Proactively encode the slice when SAM is active so the first click is fast.
  // Deliberately NOT keyed on sam.status — ensureEncoded is idempotent and
  // keying on status would re-fire every encoding→ready transition.
  useEffect(() => {
    if (samActive && samEncodeKey) {
      sam.ensureEncoded(samEncodeKey, makeSamSource).catch(() => { /* fallback handled by sam.error */ });
    }
  }, [samActive, samEncodeKey, makeSamSource, sam.ensureEncoded]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!imageRef.current) return;
    imageRef.current.cache();
    imageRef.current.filters([
      (window as any).Konva?.Filters?.Brighten ?? (() => {}),
      (window as any).Konva?.Filters?.Contrast ?? (() => {}),
    ]);
    imageRef.current.brightness(brightness);
    imageRef.current.contrast(contrast);
    imageRef.current.getLayer()?.batchDraw();
  }, [brightness, contrast, imageEl]);

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

  const isPreviewing = previewShapes !== null;
  const storeShapes = sourceKey ? (byImage[sourceKey]?.[String(currentSlice)] ?? []) : [];
  // Shapes actually drawn on Layer 1: previewed version when previewing, else the live store.
  const displayShapes = isPreviewing ? previewShapes! : storeShapes;
  // Classes used for color/visibility lookups when rendering Layer 1.
  const renderClasses = isPreviewing && previewClasses ? previewClasses : classes;

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

  // Build (and cache, per rendered image) the live-wire edge cost map.
  const ensureCostMap = (): CostMap | null => {
    if (!imageEl || !meta) return null;
    if (magneticCostRef.current && magneticBuiltForRef.current === imageEl) {
      return magneticCostRef.current;
    }
    const cm = buildCostMap(imageEl, meta.width, meta.height);
    magneticCostRef.current = cm;
    magneticBuiltForRef.current = imageEl;
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

  // Build (and cache, per rendered image) the grayscale field used by the wand.
  const magicFieldRef = useRef<GrayField | null>(null);
  const magicFieldForRef = useRef<HTMLImageElement | null>(null);
  const ensureMagicField = useCallback((): GrayField | null => {
    if (!imageEl || !meta) return null;
    if (magicFieldRef.current && magicFieldForRef.current === imageEl) return magicFieldRef.current;
    const f = buildField(imageEl, meta.width, meta.height);
    magicFieldRef.current = f;
    magicFieldForRef.current = imageEl;
    return f;
  }, [imageEl, meta]);

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
    if (magicEngine === 'sam') {
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
    // Classic wand — synchronous, coalesced to one run per frame.
    const field = ensureMagicField();
    if (!field) { setMagicPreview([]); return; }
    setMagicLoading(true);
    const id = requestAnimationFrame(() => {
      const polys: number[][] = [];
      for (const seed of magicSeeds) {
        polys.push(
          ...magicSelect(field, seed.x, seed.y, {
            toleranceFrac: magicTolerance,
            mode: magicMode,
            smooth: magicSigma,
            edgeStop: magicEdgeStop,
          }),
        );
      }
      setMagicPreview(polys);
      setMagicLoading(false);
    });
    return () => cancelAnimationFrame(id);
  }, [magicSeeds, magicBox, autoNegPoints, samDetail, samThreshold, samEncodeKey, makeSamSource, magicEngine, magicTolerance, magicMode, magicSigma, magicEdgeStop, ensureMagicField, imageEl, meta, sam.ensureEncoded, sam.segment]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Commit the magic preview polygons as new shapes (one batched undo step). */
  const commitMagic = useCallback(() => {
    if (!sourceKey || activeClassId === null) return;
    // One batched add = one undo step for the whole magic selection.
    const shapes = magicPreview
      .filter((pts) => pts.length >= 6)
      .map((pts) => ({ id: uuidv4(), classId: activeClassId, kind: 'polygon' as const, points: pts }));
    if (shapes.length) addShapes(sourceKey, currentSlice, shapes);
    resetMagic();
  }, [sourceKey, activeClassId, magicPreview, addShapes, currentSlice, resetMagic]);

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
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      setDraftPoly([]);
      setDragStart(null);
      setDragCurrent(null);
      setMarqueeStart(null);
      setMarqueeRect(null);
      resetMagnetic();
      resetMagic();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [resetMagnetic, resetMagic]);

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
    } else {
      appendBrushStroke(sourceKey, currentSlice, shapeId, { points: finalPoints, radius, mode });
    }
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
      // Erase carves from the topmost shape under the cursor (any kind). Falls
      // back to the active brush instance if the click isn't over a shape.
      const visible = (s: Shape) =>
        classes.find((c) => c.classId === s.classId)?.isVisible !== false;
      let target: Shape | undefined;
      for (let i = sliceShapes.length - 1; i >= 0; i--) {
        const s = sliceShapes[i];
        if (visible(s) && shapeContainsPoint(s, pos.x, pos.y)) { target = s; break; }
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
          addShape(sourceKey, currentSlice, { id, classId: activeClassId, kind: 'rectangle', x, y, w, h });
        } else {
          const { cx, cy, rx, ry } = normalizeEllipse(
            (dragStart.x + dragCurrent.x) / 2,
            (dragStart.y + dragCurrent.y) / 2,
            Math.abs(dx) / 2,
            Math.abs(dy) / 2
          );
          addShape(sourceKey, currentSlice, { id, classId: activeClassId, kind: 'ellipse', cx, cy, rx, ry });
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
      addShape(sourceKey, currentSlice, { id, classId: activeClassId, kind: 'polygon', points: draftPoly });
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
        addShape(sourceKey, currentSlice, { id: uuidv4(), classId: activeClassId, kind: 'polygon', points: simplified });
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
          <Line
            points={shape.points}
            closed
            fill={color}
            stroke={color}
            strokeWidth={strokeW}
            perfectDrawEnabled={false}
          />
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
    // Shift-click toggles the shape in/out of the current multi-selection.
    if (e.evt.shiftKey) {
      setSelectedShapeIds(
        selectedShapeIds.includes(id)
          ? selectedShapeIds.filter((sid) => sid !== id)
          : [...selectedShapeIds, id],
      );
    } else {
      setSelectedShapeId(id);
    }
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
      return (
        <Group
          key={shape.id}
          draggable={single}
          {...handlers}
          onDragEnd={(e) => {
            // Only the Group itself moving = a whole-shape move; vertex-circle
            // drags set their own target and are handled below.
            if (e.target !== e.currentTarget) return;
            const dx = e.target.x();
            const dy = e.target.y();
            e.target.position({ x: 0, y: 0 });
            if (dx === 0 && dy === 0) return;
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'polygon'
                ? { ...s, points: s.points.map((v, i) => (i % 2 === 0 ? v + dx : v + dy)) }
                : s,
            );
          }}
        >
          <Line
            points={livePoints}
            closed
            fill={hitFill}
            stroke={outline}
            strokeWidth={sw}
            dash={dash}
          />
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

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-gray-900 overflow-hidden"
      style={{ cursor: showBrushCursor ? 'none' : (tool === 'magnetic' || tool === 'magic') ? 'crosshair' : undefined }}
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
        {/* Layer 0: image */}
        <Layer>
          {imageEl && meta && (
            <KonvaImage
              ref={imageRef}
              image={imageEl}
              width={meta.width}
              height={meta.height}
              listening={false}
            />
          )}
        </Layer>

        {/* Layer 1: committed shapes — cached + opacity applied once at the
            layer so overlapping same-class shapes render a uniform class color. */}
        <Layer ref={shapesLayerRef} listening={false} opacity={fillOpacity}>
          {displayShapes
            .filter((s) => renderClasses.find((c) => c.classId === s.classId)?.isVisible !== false)
            .map(renderShape)}
        </Layer>

        {/* Interactive layer: hit targets for select/move/edit (select tool only). */}
        {showInteractive && (
          <Layer>
            {storeShapes
              .filter((s) => classes.find((c) => c.classId === s.classId)?.isVisible !== false)
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
          {(tool === 'magic' || tool === 'pan') &&
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
          {(tool === 'magic' || tool === 'pan') &&
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
        </Layer>

        {/* Layer 3: in-progress brush stroke (imperatively updated, no React re-renders per move) */}
        <Layer ref={draftStrokeLayerRef} listening={false} opacity={fillOpacity}>
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
      </Stage>

      {/* Selection toolbar (select tool) */}
      {showInteractive && selectedShapeIds.length > 0 && (
        <div className="absolute top-2 left-2 flex items-center gap-2 bg-slate-800/90 text-slate-100 text-xs px-3 py-1.5 rounded-md shadow-lg">
          <span className="text-slate-300">
            {selectedShapeIds.length > 1
              ? `${selectedShapeIds.length} selected`
              : selectedShape?.kind === 'polygon'
                ? 'Drag a vertex to edit, or drag the shape to move'
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
          <button
            type="button"
            onClick={handleDeleteSelected}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-red-600 hover:bg-red-500 text-white font-medium"
            title="Delete selected (Del)"
          >
            <Trash size={13} />
            Delete{selectedShapeIds.length > 1 ? ` (${selectedShapeIds.length})` : ''}
          </button>
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

      {/* Zoom overlay */}
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-0.5 rounded pointer-events-none">
        {Math.round(transform.scaleX * 100)}%
      </div>
    </div>
  );
}
