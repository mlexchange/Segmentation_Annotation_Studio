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
import type Konva from 'konva';
import { v4 as uuidv4 } from 'uuid';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore, type Shape, type BrushStroke } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { toImage, normalizeRect, normalizeEllipse } from '@/lib/geometry';
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
  } | null>(null);

  const { kind, source, serverUri, meta, currentSlice, renderOpts } = useDatasetStore();
  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;
  const { byImage, addShape, appendBrushStroke } = useAnnotationStore();
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
    appendBrushStroke(sourceKey, currentSlice, shapeId, { points: finalPoints, radius, mode });
  };

  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPreviewing) return;
    if (!sourceKey || !meta || activeClassId === null) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    if (tool === 'polygon') {
      setDraftPoly((prev) => [...prev, pos.x, pos.y]);
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
      const targetId =
        activeBrushShapeId ??
        shapes.filter((s) => s.kind === 'brush' && s.classId === activeClassId).slice(-1)[0]?.id ??
        null;
      if (targetId) {
        draftStrokeRef.current = { shapeId: targetId, mode: 'erase', points: [pos.x, pos.y], radius: brushSize };
        // Show a white dash to indicate erasing (destination-out not feasible in uncached layer)
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

  const renderShape = (shape: Shape) => {
    const color =
      shape.id === activeBrushShapeId && activeClassId !== null
        ? colorForClass(activeClassId)
        : colorForClass(shape.classId);
    const isSelected = shape.id === selectedShapeId;
    const strokeW = (isSelected ? 2 : 1) / transform.scaleX;

    if (shape.kind === 'polygon') {
      return (
        <Line
          key={shape.id}
          points={shape.points}
          closed
          fill={color}
          stroke={color}
          strokeWidth={strokeW}
          perfectDrawEnabled={false}
          onClick={() => setSelectedShapeId(shape.id)}
        />
      );
    }
    if (shape.kind === 'rectangle') {
      return (
        <Rect
          key={shape.id}
          x={shape.x} y={shape.y} width={shape.w} height={shape.h}
          fill={color}
          stroke={color} strokeWidth={strokeW}
          perfectDrawEnabled={false}
          onClick={() => setSelectedShapeId(shape.id)}
        />
      );
    }
    if (shape.kind === 'ellipse') {
      return (
        <Ellipse
          key={shape.id}
          x={shape.cx} y={shape.cy} radiusX={shape.rx} radiusY={shape.ry}
          fill={color}
          stroke={color} strokeWidth={strokeW}
          perfectDrawEnabled={false}
          onClick={() => setSelectedShapeId(shape.id)}
        />
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

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-gray-900 overflow-hidden"
      style={{ cursor: showBrushCursor ? 'none' : undefined }}
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

      {/* Zoom overlay */}
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-0.5 rounded pointer-events-none">
        {Math.round(transform.scaleX * 100)}%
      </div>
    </div>
  );
}
