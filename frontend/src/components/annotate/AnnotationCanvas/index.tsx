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
import { useRef, useState, useEffect, useCallback } from 'react';
import {
  Stage, Layer, Image as KonvaImage, Line, Rect, Ellipse, Group, Circle,
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

function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return Math.hypot(px - cx, py - cy);
}

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
  const { byImage, addShape, appendBrushStroke, appendEraseStroke, updateShape, removeShape } = useAnnotationStore();
  const { tool, brushSize, fillOpacity, selectedShapeId, setSelectedShapeId } = useToolStore();
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

  const resetMagnetic = useCallback(() => {
    magneticPrevRef.current = null;
    magneticSeedRef.current = null;
    setMagneticCommitted([]);
    setMagneticPreview([]);
  }, []);

  // Abandon any in-progress magnetic trace when the tool or slice changes.
  useEffect(() => {
    resetMagnetic();
  }, [tool, currentSlice, sourceKey, resetMagnetic]);

  // Escape cancels the entire in-progress shape (polygon vertices, magnetic
  // trace, or rect/ellipse drag) without committing anything.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      setDraftPoly([]);
      setDragStart(null);
      setDragCurrent(null);
      resetMagnetic();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [resetMagnetic]);

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

  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPreviewing) return;

    // Select tool: clicking empty canvas (the stage itself) clears the selection.
    // Clicks on a shape are handled by that shape's own onClick (which selects it).
    if (tool === 'select') {
      if (e.target === e.target.getStage()) setSelectedShapeId(null);
      return;
    }

    if (!sourceKey || !meta || activeClassId === null) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    if (tool === 'polygon') {
      setDraftPoly((prev) => [...prev, pos.x, pos.y]);
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

  const handleStageMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPreviewing) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    // Update brush cursor position imperatively — no React state, no re-render
    if (showBrushCursor && brushCursorRef.current) {
      brushCursorRef.current.position({ x: pos.x, y: pos.y });
      brushCursorLayerRef.current?.batchDraw();
    }

    if (!sourceKey || !meta) return;

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

  const handleStageMouseUp = () => {
    if (isPreviewing) return;
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

  const renderShape = (shape: Shape) => {
    const color =
      shape.id === activeBrushShapeId && activeClassId !== null
        ? colorForClass(activeClassId)
        : colorForClass(shape.classId);
    const isSelected = shape.id === selectedShapeId;
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

  const selectShape = (e: Konva.KonvaEventObject<MouseEvent>, id: string) => {
    e.cancelBubble = true;
    setSelectedShapeId(id);
  };

  /** Render a shape as an interactive, draggable hit target with a selection outline. */
  const renderInteractive = (shape: Shape) => {
    if (!sourceKey) return null;
    const selected = shape.id === selectedShapeId;
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
          draggable={selected}
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
          {selected &&
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
          x={shape.x} y={shape.y} width={shape.w} height={shape.h}
          fill={hitFill} stroke={outline} strokeWidth={sw} dash={dash}
          draggable={selected}
          {...handlers}
          onDragEnd={(e) => {
            const nx = e.target.x();
            const ny = e.target.y();
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'rectangle' ? { ...s, x: nx, y: ny } : s,
            );
          }}
        />
      );
    }

    if (shape.kind === 'ellipse') {
      return (
        <Ellipse
          key={shape.id}
          x={shape.cx} y={shape.cy} radiusX={shape.rx} radiusY={shape.ry}
          fill={hitFill} stroke={outline} strokeWidth={sw} dash={dash}
          draggable={selected}
          {...handlers}
          onDragEnd={(e) => {
            const nx = e.target.x();
            const ny = e.target.y();
            updateShape(sourceKey, currentSlice, shape.id, (s) =>
              s.kind === 'ellipse' ? { ...s, cx: nx, cy: ny } : s,
            );
          }}
        />
      );
    }

    if (shape.kind === 'brush') {
      return (
        <Group
          key={shape.id}
          draggable={selected}
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
  const selectedShape = sourceKey
    ? storeShapes.find((s) => s.id === selectedShapeId) ?? null
    : null;

  const handleDeleteSelected = () => {
    if (!sourceKey || !selectedShapeId) return;
    removeShape(sourceKey, currentSlice, selectedShapeId);
    setSelectedShapeId(null);
    setEditPoly(null);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-gray-900 overflow-hidden"
      style={{ cursor: showBrushCursor ? 'none' : tool === 'magnetic' ? 'crosshair' : undefined }}
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

          {/* Magnetic lasso: committed path (solid) + live edge-traced preview (dashed) */}
          {tool === 'magnetic' && magneticCommitted.length >= 2 && (
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
          {tool === 'magnetic' && magneticSeedRef.current && (
            <Circle
              x={magneticSeedRef.current.x}
              y={magneticSeedRef.current.y}
              radius={4 / transform.scaleX}
              fill={activeColor}
              listening={false}
            />
          )}
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
      {showInteractive && selectedShape && (
        <div className="absolute top-2 left-2 flex items-center gap-2 bg-slate-800/90 text-slate-100 text-xs px-3 py-1.5 rounded-md shadow-lg">
          <span className="text-slate-300">
            {selectedShape.kind === 'polygon'
              ? 'Drag a vertex to edit, or drag the shape to move'
              : 'Drag to move'}
          </span>
          <button
            type="button"
            onClick={handleDeleteSelected}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-red-600 hover:bg-red-500 text-white font-medium"
            title="Delete selected (Del)"
          >
            <Trash size={13} />
            Delete
          </button>
        </div>
      )}

      {showInteractive && !selectedShape && (
        <div className="absolute top-2 left-2 bg-slate-800/80 text-slate-300 text-xs px-3 py-1.5 rounded-md pointer-events-none">
          Click an annotation to select it
        </div>
      )}

      {tool === 'magnetic' && !isPreviewing && (
        <div className="absolute top-2 left-2 bg-slate-800/85 text-slate-200 text-xs px-3 py-1.5 rounded-md pointer-events-none">
          Click along an edge to trace it · double-click to finish
        </div>
      )}

      {/* Zoom overlay */}
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-0.5 rounded pointer-events-none">
        {Math.round(transform.scaleX * 100)}%
      </div>
    </div>
  );
}
