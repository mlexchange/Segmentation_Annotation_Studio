/**
 * ShapesLayer — the cached, non-interactive layer of committed annotations.
 *
 * Extracted from AnnotationCanvas and wrapped in `React.memo` for one reason: the
 * canvas takes brightness / contrast / levels / gamma / blur as props, so every
 * tick of a display slider re-rendered it and reconciled every Konva node on this
 * layer — despite shapes depending on none of those values. On a slice with many
 * annotations that dominated slider latency. Keeping this layer's props limited to
 * what it actually draws means those ticks now skip the entire shape tree.
 *
 * If you add a prop here, make sure it genuinely affects the drawn shapes;
 * anything that changes per-frame (a pan offset, a pointer position) would
 * reintroduce exactly the problem this exists to solve.
 */
import { memo } from 'react';
import { Layer, Line, Rect, Ellipse, Group, Shape as KonvaShape } from 'react-konva';
import type Konva from 'konva';
import type { Shape, EraseStroke } from '@/stores/annotationStore';
import type { AnnotationClass } from '@/stores/classStore';

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

/** Erase carve-outs rendered destination-out over the shape they belong to. */
function renderErased(erased?: EraseStroke[]) {
  return (erased ?? []).map((st, i) => (
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
}

export interface ShapesLayerProps {
  /** Konva layer ref — the parent owns caching/recaching of this layer. */
  layerRef: React.Ref<Konva.Layer>;
  shapes: Shape[];
  /** Class lookup for color + visibility (O(1) per shape). */
  classMap: Map<number, AnnotationClass>;
  fillOpacity: number;
  /** Current zoom, used only to keep stroke width constant on screen. */
  scaleX: number;
  selectedShapeIds: string[];
  /** The in-progress brush instance, recolored to the active class. */
  activeBrushShapeId: string | null;
  activeClassId: number | null;
  /** Image frame, for clipping and for giving hole-shapes a real self-rect. */
  imageWidth: number;
  imageHeight: number;
}

function ShapesLayerImpl({
  layerRef,
  shapes,
  classMap,
  fillOpacity,
  scaleX,
  selectedShapeIds,
  activeBrushShapeId,
  activeClassId,
  imageWidth,
  imageHeight,
}: ShapesLayerProps) {
  const colorForClass = (classId: number) => classMap.get(classId)?.color ?? '#ff0000';

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
      width={imageWidth}
      height={imageHeight}
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

  /** Render a committed shape (any kind), with the active brush instance recolored
   *  to the active class and erase strokes carved out. */
  const renderShape = (shape: Shape) => {
    const color =
      shape.id === activeBrushShapeId && activeClassId !== null
        ? colorForClass(activeClassId)
        : colorForClass(shape.classId);
    const isSelected = selectedShapeIds.includes(shape.id);
    const strokeW = (isSelected ? 2 : 1) / scaleX;

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
          {shape.strokes.map((stroke, i) => (
            <Line
              key={i}
              points={stroke.points}
              stroke={stroke.mode === 'erase' ? 'black' : color}
              strokeWidth={stroke.radius * 2}
              lineCap="round"
              lineJoin="round"
              {...(stroke.mode === 'erase'
                ? { globalCompositeOperation: 'destination-out' as const }
                : {})}
              perfectDrawEnabled={false}
              listening={false}
            />
          ))}
        </Group>
      );
    }
    return null;
  };

  return (
    <Layer
      ref={layerRef}
      listening={false}
      opacity={fillOpacity}
      clipX={0}
      clipY={0}
      clipWidth={imageWidth}
      clipHeight={imageHeight}
    >
      {shapes
        .filter((s) => classMap.get(s.classId)?.isVisible !== false)
        .map(renderShape)}
    </Layer>
  );
}

export default memo(ShapesLayerImpl);
