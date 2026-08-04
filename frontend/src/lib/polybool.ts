/**
 * polybool — convert annotation shapes to/from `polygon-clipping` geometry so we
 * can UNION overlapping regions with true polygon boolean ops instead of a
 * rasterize→re-vectorize round-trip. The boolean union preserves the input
 * polygons' existing vertices everywhere except along the seam where they merge,
 * so merging additional strokes/shapes no longer makes existing nodes drift or
 * erode (the mask round-trip regenerated every vertex from the pixel staircase).
 *
 * Brush shapes (thick polylines) have no meaningful polygon vertices, so they are
 * rasterized and vectorized once to obtain an outline — only their own (new) nodes
 * are approximate; the shapes they merge with keep their exact geometry.
 */
import polygonClipping, { type MultiPolygon, type Ring } from 'polygon-clipping';
import type { PolygonShape, Shape } from '@/stores/annotationStore';
import { fullResGridFor, rasterizeShapes } from '@/lib/rasterize';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import { v4 as uuidv4 } from 'uuid';

const ELLIPSE_SEGMENTS = 72;

/** flat [x,y,…] ring → a closed polygon-clipping Ring ([[x,y],…] with first==last). */
function flatToRing(flat: number[]): Ring {
  const ring: Ring = [];
  for (let i = 0; i + 1 < flat.length; i += 2) ring.push([flat[i], flat[i + 1]]);
  if (ring.length && (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1])) {
    ring.push([ring[0][0], ring[0][1]]);
  }
  return ring;
}

/** Signed-area magnitude (shoelace) of a flat ring — used to drop sliver artifacts. */
function ringArea(flat: number[]): number {
  let a = 0;
  const n = flat.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    a += flat[j * 2] * flat[i * 2 + 1] - flat[i * 2] * flat[j * 2 + 1];
  }
  return Math.abs(a) / 2;
}

/**
 * Convert a shape to `polygon-clipping` geometry (a MultiPolygon). Brush shapes are
 * rasterized + vectorized at full resolution; vector shapes map directly.
 */
export function shapeToMultiPolygon(shape: Shape, width: number, height: number): MultiPolygon {
  if (shape.kind === 'polygon') {
    return [[flatToRing(shape.points), ...(shape.holes ?? []).map(flatToRing)]];
  }
  if (shape.kind === 'rectangle') {
    const { x, y, w, h } = shape;
    return [[[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]]];
  }
  if (shape.kind === 'ellipse') {
    const ring: Ring = [];
    for (let i = 0; i <= ELLIPSE_SEGMENTS; i++) {
      const t = (i / ELLIPSE_SEGMENTS) * Math.PI * 2;
      ring.push([shape.cx + Math.cos(t) * shape.rx, shape.cy + Math.sin(t) * shape.ry]);
    }
    return [[ring]];
  }
  // Brush: rasterize this shape alone, then vectorize to an outline (with holes).
  const { gw, gh, scale } = fullResGridFor(width, height);
  const mask = rasterizeShapes([shape], gw, gh, scale);
  return maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
    .filter((p) => p.points.length >= 6)
    .map((p) => [flatToRing(p.points), ...p.holes.map(flatToRing)]);
}

/**
 * Convert a `polygon-clipping` MultiPolygon back to annotation polygon shapes. Each
 * output polygon becomes one PolygonShape (exterior ring + holes); slivers below
 * `minArea` px² and degenerate rings are dropped.
 */
export function multiPolygonToShapes(mp: MultiPolygon, classId: number, minArea = 1): PolygonShape[] {
  const shapes: PolygonShape[] = [];
  for (const poly of mp) {
    if (!poly.length) continue;
    const [outer, ...holeRings] = poly;
    const points = ringToFlat(outer);
    if (points.length < 6 || ringArea(points) < minArea) continue;
    const holes = holeRings
      .map(ringToFlat)
      .filter((h) => h.length >= 6 && ringArea(h) >= minArea);
    shapes.push({
      id: uuidv4(),
      classId,
      kind: 'polygon',
      points,
      ...(holes.length ? { holes } : {}),
    });
  }
  return shapes;
}

/** polygon-clipping Ring → flat [x,y,…], dropping the closing duplicate vertex. */
function ringToFlat(ring: Ring): number[] {
  const flat: number[] = [];
  const n = ring.length >= 2 && ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1]
    ? ring.length - 1
    : ring.length;
  for (let i = 0; i < n; i++) flat.push(ring[i][0], ring[i][1]);
  return flat;
}

/**
 * Build the eraser stroke's swept disk as polygon geometry (a round-capped capsule
 * chain), so it can be boolean-subtracted from shapes. Rasterized + vectorized once
 * per stroke; its own outline is a staircase, but that only affects the cut edge.
 */
export function eraseStampToMultiPolygon(points: number[], radius: number, width: number, height: number): MultiPolygon {
  const { gw, gh, scale } = fullResGridFor(width, height);
  const stamp: Shape = { id: 'stamp', classId: -1, kind: 'brush', strokes: [{ points, radius, mode: 'paint' }] };
  const mask = rasterizeShapes([stamp], gw, gh, scale);
  return maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 1, scale })
    .filter((p) => p.points.length >= 6)
    .map((p) => [flatToRing(p.points), ...p.holes.map(flatToRing)]);
}

/** Boolean-union a set of shapes into one MultiPolygon (empty if none / on failure). */
export function unionShapesToMultiPolygon(shapes: Shape[], width: number, height: number): MultiPolygon {
  const geoms = shapes.map((s) => shapeToMultiPolygon(s, width, height)).filter((g) => g.length > 0);
  if (geoms.length === 0) return [];
  try {
    return geoms.length === 1 ? geoms[0] : polygonClipping.union(geoms[0], ...geoms.slice(1));
  } catch {
    return [];
  }
}

/**
 * Boolean-subtract `stampMP` from a shape, preserving the shape's untouched
 * vertices (only the cut edge gets new points). Returns the remaining polygon
 * shape(s) — possibly `[]` when fully erased — or `null` if the op can't run
 * (empty geometry / failure), so callers fall back to the mask approach.
 */
export function subtractFromShape(shape: Shape, stampMP: MultiPolygon, width: number, height: number): PolygonShape[] | null {
  try {
    const shapeMP = shapeToMultiPolygon(shape, width, height);
    if (shapeMP.length === 0 || stampMP.length === 0) return null;
    return multiPolygonToShapes(polygonClipping.difference(shapeMP, stampMP), shape.classId);
  } catch {
    return null;
  }
}

/**
 * Boolean-union a set of shapes into merged polygon shape(s), preserving each
 * input's existing vertices except along merge seams. Returns null if the union
 * fails or is empty, so callers can fall back to the mask approach.
 */
export function unionShapesToPolygons(shapes: Shape[], classId: number, width: number, height: number): PolygonShape[] | null {
  try {
    const geoms = shapes
      .map((s) => shapeToMultiPolygon(s, width, height))
      .filter((g) => g.length > 0);
    if (geoms.length === 0) return null;
    const merged = geoms.length === 1 ? geoms[0] : polygonClipping.union(geoms[0], ...geoms.slice(1));
    const out = multiPolygonToShapes(merged, classId);
    return out.length ? out : null;
  } catch {
    return null;
  }
}
