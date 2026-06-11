/**
 * AnnotationCanvas — react-konva Stage with image + shape layers.
 *
 * Layer 0: image (Konva filters: Brighten/Contrast)
 * Layer 1: committed shapes (listening=false, per-brush-instance Groups)
 * Layer 2: draft shape
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

  const { kind, source, serverUri, meta, currentSlice, renderOpts } = useDatasetStore();
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

  const shapes = source ? (byImage[source]?.[String(currentSlice)] ?? []) : [];

  const colorForClass = (classId: number) =>
    classes.find((c) => c.classId === classId)?.color ?? '#ff0000';

  const getPointerImagePos = () => {
    const stage = stageRef.current;
    if (!stage) return null;
    const pos = stage.getPointerPosition();
    if (!pos) return null;
    return toImage(pos, transform);
  };

  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!source || !meta || activeClassId === null) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    if (tool === 'polygon') {
      setDraftPoly((prev) => [...prev, pos.x, pos.y]);
    } else if (tool === 'rectangle' || tool === 'ellipse') {
      setDragStart(pos);
      setDragCurrent(pos);
    } else if (tool === 'brush') {
      // Start a new brush stroke on the active brush instance (or create new)
      const stroke: BrushStroke = { points: [pos.x, pos.y], radius: brushSize, mode: 'paint' };
      if (activeBrushShapeId) {
        appendBrushStroke(source, currentSlice, activeBrushShapeId, stroke);
      } else {
        const id = uuidv4();
        addShape(source, currentSlice, { id, classId: activeClassId, kind: 'brush', strokes: [stroke] });
        onNewBrushInstance(id);
      }
    } else if (tool === 'eraser') {
      // Find target brush instance
      const targetId = activeBrushShapeId ?? shapes.filter((s) => s.kind === 'brush' && s.classId === activeClassId).slice(-1)[0]?.id ?? null;
      if (targetId) {
        const eraseStroke: BrushStroke = { points: [pos.x, pos.y], radius: brushSize, mode: 'erase' };
        appendBrushStroke(source, currentSlice, targetId, eraseStroke);
      }
    }
  };

  const handleStageMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!source || !meta) return;
    const pos = getPointerImagePos();
    if (!pos) return;

    if ((tool === 'rectangle' || tool === 'ellipse') && dragStart && e.evt.buttons === 1) {
      setDragCurrent(pos);
    } else if (tool === 'brush' && e.evt.buttons === 1 && activeBrushShapeId) {
      // Append point to current stroke
      const sliceShapes = byImage[source]?.[String(currentSlice)] ?? [];
      const brushShape = sliceShapes.find((s) => s.id === activeBrushShapeId && s.kind === 'brush');
      if (brushShape && brushShape.kind === 'brush') {
        const lastStroke = brushShape.strokes[brushShape.strokes.length - 1];
        const updatedStroke: BrushStroke = {
          ...lastStroke,
          points: [...lastStroke.points, pos.x, pos.y],
        };
        const newStrokes = [...brushShape.strokes.slice(0, -1), updatedStroke];
        setShapes(
          source,
          currentSlice,
          sliceShapes.map((s) => (s.id === activeBrushShapeId ? { ...brushShape, strokes: newStrokes } : s))
        );
      }
    } else if (tool === 'eraser' && e.evt.buttons === 1) {
      const targetId = activeBrushShapeId ?? shapes.filter((s) => s.kind === 'brush' && s.classId === activeClassId).slice(-1)[0]?.id ?? null;
      if (targetId) {
        const sliceShapes = byImage[source]?.[String(currentSlice)] ?? [];
        const brushShape = sliceShapes.find((s) => s.id === targetId && s.kind === 'brush');
        if (brushShape && brushShape.kind === 'brush') {
          const lastStroke = brushShape.strokes[brushShape.strokes.length - 1];
          if (lastStroke.mode === 'erase') {
            const updatedStroke: BrushStroke = { ...lastStroke, points: [...lastStroke.points, pos.x, pos.y] };
            const newStrokes = [...brushShape.strokes.slice(0, -1), updatedStroke];
            setShapes(source, currentSlice, sliceShapes.map((s) => (s.id === targetId ? { ...brushShape, strokes: newStrokes } : s)));
          }
        }
      }
    }
  };

  const handleStageMouseUp = () => {
    if (!source || !meta || activeClassId === null) return;

    if ((tool === 'rectangle' || tool === 'ellipse') && dragStart && dragCurrent) {
      const dx = dragCurrent.x - dragStart.x;
      const dy = dragCurrent.y - dragStart.y;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        const id = uuidv4();
        if (tool === 'rectangle') {
          const { x, y, w, h } = normalizeRect(dragStart.x, dragStart.y, dx, dy);
          addShape(source, currentSlice, { id, classId: activeClassId, kind: 'rectangle', x, y, w, h });
        } else {
          const { cx, cy, rx, ry } = normalizeEllipse(
            (dragStart.x + dragCurrent.x) / 2,
            (dragStart.y + dragCurrent.y) / 2,
            Math.abs(dx) / 2,
            Math.abs(dy) / 2
          );
          addShape(source, currentSlice, { id, classId: activeClassId, kind: 'ellipse', cx, cy, rx, ry });
        }
      }
      setDragStart(null);
      setDragCurrent(null);
    }
  };

  const handleStageDblClick = () => {
    // Close polygon on double-click
    if (tool === 'polygon' && draftPoly.length >= 6 && source && activeClassId !== null) {
      const id = uuidv4();
      addShape(source, currentSlice, { id, classId: activeClassId, kind: 'polygon', points: draftPoly });
      setDraftPoly([]);
    }
  };

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
          fill="rgba(100,160,255,0.3)" stroke="#4090ff" strokeWidth={1 / transform.scaleX}
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
          fill="rgba(100,160,255,0.3)" stroke="#4090ff" strokeWidth={1 / transform.scaleX}
          listening={false} perfectDrawEnabled={false}
        />
      );
    }
    return null;
  };

  const renderShape = (shape: Shape) => {
    const color = colorForClass(shape.classId);
    const isSelected = shape.id === selectedShapeId;
    const strokeW = (isSelected ? 2 : 1) / transform.scaleX;

    if (shape.kind === 'polygon') {
      return (
        <Line
          key={shape.id}
          points={shape.points}
          closed
          fill={color + Math.round(fillOpacity * 255).toString(16).padStart(2, '0')}
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
          fill={color + Math.round(fillOpacity * 255).toString(16).padStart(2, '0')}
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
          fill={color + Math.round(fillOpacity * 255).toString(16).padStart(2, '0')}
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
                opacity={fillOpacity}
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
    <div ref={containerRef} className="relative w-full h-full bg-gray-900 overflow-hidden">
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        draggable={tool === 'pan'}
        onMouseDown={handleStageMouseDown}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
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

        {/* Layer 1: committed shapes */}
        <Layer listening={false}>
          {shapes.map(renderShape)}
        </Layer>

        {/* Layer 2: draft polygon + drag preview */}
        <Layer>
          {draftPoly.length >= 2 && (
            <>
              <Line
                points={draftPoly}
                stroke="#4090ff"
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
                    fill="#4090ff"
                    listening={false}
                  />
                ) : null
              )}
            </>
          )}
          {renderDraftShape()}
        </Layer>
      </Stage>

      {/* Zoom overlay */}
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-0.5 rounded pointer-events-none">
        {Math.round(transform.scaleX * 100)}%
      </div>
    </div>
  );
}
