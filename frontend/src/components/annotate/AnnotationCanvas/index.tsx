/**
 * AnnotationCanvas — react-konva Stage with image + shape layers.
 *
 * Layer 0: image (Konva filters: Brighten/Contrast)
 * Layer 1: committed shapes (listening=false, per-brush-instance Groups)
 * Layer 2: draft polygon + drag preview
 * Layer 3: brush/eraser size cursor preview
 */
import { useRef, useState, useCallback, useEffect } from 'react';
import {
  Stage, Layer, Image as KonvaImage, Line, Rect, Ellipse, Group, Circle,
} from 'react-konva';
import type Konva from 'konva';
import { v4 as uuidv4 } from 'uuid';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore, type Shape, type BrushStroke } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { toImage, normalizeRect, normalizeEllipse } from '@/lib/geometry';
import { useImageSlice } from '@/hooks/useImageSlice';
import { buildSourceKey } from '@/lib/sourceKey';

interface AnnotationCanvasProps {
  brightness: number;
  contrast: number;
  activeClassId: number | null;
  activeBrushShapeId: string | null;
  onNewBrushInstance: (id: string) => void;
}

export default function AnnotationCanvas({
  brightness,
  contrast,
  activeClassId,
  activeBrushShapeId,
  onNewBrushInstance,
}: AnnotationCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<Konva.Image>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const shapesLayerRef = useRef<Konva.Layer>(null);

  const { kind, source, serverUri, meta, currentSlice, renderOpts } = useDatasetStore();
  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;
  const { byImage, addShape, removeShape, appendBrushStroke, setShapes } = useAnnotationStore();
  const { tool, brushSize, fillOpacity, selectedShapeId, setSelectedShapeId } = useToolStore();
  const { classes } = useClassStore();

  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [transform, setTransform] = useState({ scaleX: 1, scaleY: 1, x: 0, y: 0 });

  // Draft polygon vertices (click-vertex mode)
  const [draftPoly, setDraftPoly] = useState<number[]>([]);
  // Draft rect/ellipse start
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);
  // True while a brush/eraser stroke is actively being drawn — suppresses the
  // (expensive) layer re-cache so live painting stays responsive.
  const [isDrawing, setIsDrawing] = useState(false);
  // Image-space pointer position for brush/eraser size preview (null when off-canvas).
  const [brushCursorPos, setBrushCursorPos] = useState<{ x: number; y: number } | null>(null);

  const { data: sliceUrl } = useImageSlice(source, kind, currentSlice, renderOpts, serverUri);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);

  // Load image element when URL changes
  useEffect(() => {
    if (!sliceUrl) return;
    const img = new window.Image();
    img.onload = () => setImageEl(img);
    img.src = sliceUrl;
  }, [sliceUrl]);

  // Apply brightness/contrast filters on image node
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

  // Fit container
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

  // Fit image to stage when image or stage size changes
  useEffect(() => {
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

  const shapes = sourceKey ? (byImage[sourceKey]?.[String(currentSlice)] ?? []) : [];

  // Cache the shapes layer so its children are flattened into a single bitmap
  // BEFORE the layer opacity is applied. This makes overlapping brush strokes
  // (and overlapping shapes of the same class) composite to one uniform color
  // instead of stacking alpha and looking darker where they overlap.
  useEffect(() => {
    const layer = shapesLayerRef.current;
    if (!layer) return;
    if (isDrawing) return; // don't re-cache mid-stroke (keeps painting smooth)
    layer.clearCache();
    if (shapes.length > 0) {
      // Cache at a pixel ratio matching the current zoom so the flattened
      // bitmap stays crisp when the Stage scales it. Clamped to avoid huge
      // canvases at extreme zoom.
      const pr = Math.min(Math.max(transform.scaleX, 1), 4);
      layer.cache({ pixelRatio: pr });
    }
    layer.batchDraw();
  }, [shapes, classes, fillOpacity, meta, transform.scaleX, isDrawing]);

  const colorForClass = (classId: number) =>
    classes.find((c) => c.classId === classId)?.color ?? '#ff0000';

  const activeColor =
    activeClassId !== null ? colorForClass(activeClassId) : '#4090ff';

  const showBrushCursor = (tool === 'brush' || tool === 'eraser') && !!meta;

  const getPointerImagePos = () => {
    const stage = stageRef.current;
    if (!stage) return null;
    const pos = stage.getPointerPosition();
    if (!pos) return null;
    return toImage(pos, transform);
  };

  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
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
      const stroke: BrushStroke = { points: [pos.x, pos.y], radius: brushSize, mode: 'paint' };
      const sliceShapes = byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const existingBrush = activeBrushShapeId
        ? sliceShapes.find((s) => s.id === activeBrushShapeId && s.kind === 'brush')
        : null;
      const canAppend = existingBrush && existingBrush.classId === activeClassId;
      if (canAppend && activeBrushShapeId) {
        appendBrushStroke(sourceKey, currentSlice, activeBrushShapeId, stroke);
      } else {
        const id = uuidv4();
        addShape(sourceKey, currentSlice, { id, classId: activeClassId, kind: 'brush', strokes: [stroke] });
        onNewBrushInstance(id);
      }
    } else if (tool === 'eraser') {
      setIsDrawing(true);
      const targetId = activeBrushShapeId ?? shapes.filter((s) => s.kind === 'brush' && s.classId === activeClassId).slice(-1)[0]?.id ?? null;
      if (targetId) {
        const eraseStroke: BrushStroke = { points: [pos.x, pos.y], radius: brushSize, mode: 'erase' };
        appendBrushStroke(sourceKey, currentSlice, targetId, eraseStroke);
      }
    }
  };

  const handleStageMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const pos = getPointerImagePos();
    if (showBrushCursor && pos) {
      setBrushCursorPos(pos);
    }
    if (!sourceKey || !meta || !pos) return;

    if ((tool === 'rectangle' || tool === 'ellipse') && dragStart && e.evt.buttons === 1) {
      setDragCurrent(pos);
    } else if (tool === 'brush' && e.evt.buttons === 1 && activeBrushShapeId) {
      const sliceShapes = byImage[sourceKey]?.[String(currentSlice)] ?? [];
      const brushShape = sliceShapes.find((s) => s.id === activeBrushShapeId && s.kind === 'brush');
      if (brushShape && brushShape.kind === 'brush') {
        const lastStroke = brushShape.strokes[brushShape.strokes.length - 1];
        const updatedStroke: BrushStroke = {
          ...lastStroke,
          points: [...lastStroke.points, pos.x, pos.y],
        };
        const newStrokes = [...brushShape.strokes.slice(0, -1), updatedStroke];
        setShapes(
          sourceKey,
          currentSlice,
          sliceShapes.map((s) => (s.id === activeBrushShapeId ? { ...brushShape, strokes: newStrokes } : s))
        );
      }
    } else if (tool === 'eraser' && e.evt.buttons === 1) {
      const targetId = activeBrushShapeId ?? shapes.filter((s) => s.kind === 'brush' && s.classId === activeClassId).slice(-1)[0]?.id ?? null;
      if (targetId) {
        const sliceShapes = byImage[sourceKey]?.[String(currentSlice)] ?? [];
        const brushShape = sliceShapes.find((s) => s.id === targetId && s.kind === 'brush');
        if (brushShape && brushShape.kind === 'brush') {
          const lastStroke = brushShape.strokes[brushShape.strokes.length - 1];
          if (lastStroke.mode === 'erase') {
            const updatedStroke: BrushStroke = { ...lastStroke, points: [...lastStroke.points, pos.x, pos.y] };
            const newStrokes = [...brushShape.strokes.slice(0, -1), updatedStroke];
            setShapes(sourceKey, currentSlice, sliceShapes.map((s) => (s.id === targetId ? { ...brushShape, strokes: newStrokes } : s)));
          }
        }
      }
    }
  };

  const handleStageMouseUp = () => {
    if (isDrawing) setIsDrawing(false);
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
    if (tool === 'polygon' && draftPoly.length >= 6 && sourceKey && activeClassId !== null) {
      const id = uuidv4();
      addShape(sourceKey, currentSlice, { id, classId: activeClassId, kind: 'polygon', points: draftPoly });
      setDraftPoly([]);
    }
  };

  const handleStageMouseLeave = () => {
    setBrushCursorPos(null);
    if (isDrawing) setIsDrawing(false);
    setDragStart(null);
    setDragCurrent(null);
  };

  // Hide brush cursor preview when switching away from brush/eraser.
  useEffect(() => {
    if (!showBrushCursor) setBrushCursorPos(null);
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

    // Fill is rendered at full color; the surrounding Layer applies opacity once
    // so overlapping shapes/strokes of the same class don't compound into a
    // darker/mismatched color.
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
      style={{ cursor: showBrushCursor && brushCursorPos ? 'none' : undefined }}
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
            layer so overlapping same-class shapes render a uniform class color
            (no darker overlap). */}
        <Layer ref={shapesLayerRef} listening={false} opacity={fillOpacity}>
          {shapes
            .filter((s) => classes.find((c) => c.classId === s.classId)?.isVisible !== false)
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

        {/* Layer 3: brush/eraser size preview */}
        <Layer listening={false}>
          {showBrushCursor && brushCursorPos && (
            <Circle
              x={brushCursorPos.x}
              y={brushCursorPos.y}
              radius={brushSize}
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
          )}
        </Layer>
      </Stage>

      {/* Zoom overlay */}
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-0.5 rounded pointer-events-none">
        {Math.round(transform.scaleX * 100)}%
      </div>
    </div>
  );
}
