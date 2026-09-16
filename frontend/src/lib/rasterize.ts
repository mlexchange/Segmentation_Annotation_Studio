/**
 * Rasterize annotation shapes to a binary mask (pure JS — no canvas), the
 * inverse of `maskToPolygons`. Used by threshold selection, mask cleanup
 * (morphology), and cross-slice interpolation, which all need to move between
 * the vector shape model and a pixel mask.
 *
 * Shapes are stored in IMAGE pixel coords; the mask is computed on a grid of
 * `gw × gh` where image coord `p` maps to grid `p / scale` (matching the
 * downsampling convention of `buildField` / `maskToPolygons`).
 */
import type { Shape } from '@/stores/annotationStore';

export interface MaskGrid {
  gw: number;
  gh: number;
  scale: number;
}

/** Grid dimensions for an image, downsampled so the long side is ≤ maxDim. */
export function gridFor(width: number, height: number, maxDim = 1600): MaskGrid {
  const scale = Math.max(1, Math.ceil(Math.max(width, height) / maxDim));
  return { gw: Math.max(1, Math.floor(width / scale)), gh: Math.max(1, Math.floor(height / scale)), scale };
}

/**
 * Grid for mask round-trips that must PRESERVE geometry (clip / merge / erase):
 * full native resolution (scale 1) for reasonably sized images, so re-vectorizing
 * an unchanged region is ~idempotent and existing nodes don't erode or shift a
 * little each time a stroke is added. Only very large images (>4096 px) downsample.
 *
 * `upscale` (1, 2, 4) raises the working resolution so sub-pixel geometry — e.g. a
 * Threshold Brush region traced at 2× — survives a clip/merge round-trip instead of
 * being re-snapped to the native pixel grid. `scale` becomes fractional (1/upscale);
 * `rasterizeShapes` divides by it and `maskToPolygons*` multiplies by it, so nothing
 * downstream needs to change.
 */
export function fullResGridFor(width: number, height: number, upscale = 1): MaskGrid {
  const u = Math.max(1, upscale);
  const emax = Math.max(width, height);
  const base = gridFor(width, height, emax <= 4096 ? emax : 1600);
  if (u === 1) return base;
  const scale = base.scale / u;
  return {
    gw: Math.max(1, Math.floor(width / scale)),
    gh: Math.max(1, Math.floor(height / scale)),
    scale,
  };
}

/**
 * Rasterize *shapes* into a `gw × gh` binary mask (Uint8Array of 0/1). Paint
 * strokes and shape bodies set 1; brush erase strokes and vector `erased`
 * carve-outs set 0, applied per-shape so a later shape can repaint.
 *
 * Pass `out` to render into an existing buffer instead of allocating one. Callers
 * that rasterize many shapes in a loop (overlap tests) reuse a single scratch
 * array this way — at full resolution each allocation is multiple megabytes, and
 * the garbage adds up fast. `out` must be `gw*gh` long and is NOT cleared: clear
 * it yourself when reusing, or leave it to accumulate a union deliberately.
 */
export function rasterizeShapes(shapes: Shape[], gw: number, gh: number, scale = 1, out?: Uint8Array): Uint8Array {
  const mask = out ?? new Uint8Array(gw * gh);
  const s = scale || 1;
  for (const shape of shapes) {
    if (shape.kind === 'polygon') {
      fillPolygon(mask, gw, gh, shape.points, s, 1);
      // Carve inner rings (holes) back out — e.g. from "invert shape".
      for (const hole of shape.holes ?? []) fillPolygon(mask, gw, gh, hole, s, 0);
    } else if (shape.kind === 'rectangle') {
      fillRect(mask, gw, gh, shape.x / s, shape.y / s, shape.w / s, shape.h / s, 1);
    } else if (shape.kind === 'ellipse') {
      fillEllipse(mask, gw, gh, shape.cx / s, shape.cy / s, shape.rx / s, shape.ry / s, 1);
    } else if (shape.kind === 'brush') {
      for (const st of shape.strokes) {
        if (st.mode === 'paint') stampStroke(mask, gw, gh, st.points, st.radius / s, s, 1);
      }
      for (const st of shape.strokes) {
        if (st.mode === 'erase') stampStroke(mask, gw, gh, st.points, st.radius / s, s, 0);
      }
    }
    for (const er of shape.erased ?? []) {
      stampStroke(mask, gw, gh, er.points, er.radius / s, s, 0);
    }
  }
  return mask;
}

/**
 * Union of independently-rasterized shapes. Unlike `rasterizeShapes([...])`
 * (which fills all shapes onto one mask in order, so a later shape's holes/erase
 * carve-outs can cut through an earlier shape's fill), each shape is rasterized
 * on its own and OR-ed in — so carve-outs stay scoped to their own shape. Use
 * this whenever combining DIFFERENT shapes (e.g. a class's region mask).
 */
export function rasterizeUnion(shapes: Shape[], gw: number, gh: number, scale = 1): Uint8Array {
  const mask = new Uint8Array(gw * gh);
  for (const shape of shapes) {
    const m = rasterizeShapes([shape], gw, gh, scale);
    for (let i = 0; i < m.length; i++) if (m[i]) mask[i] = 1;
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Primitives (grid coords unless noted; `scale` converts image → grid)
// ---------------------------------------------------------------------------

/** Even-odd scanline fill of a flat [x,y,…] image-coord polygon. */
function fillPolygon(mask: Uint8Array, gw: number, gh: number, imgPts: number[], scale: number, v: number): void {
  const n = imgPts.length / 2;
  if (n < 3) return;
  const xs = new Array<number>(n);
  const ys = new Array<number>(n);
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    xs[i] = imgPts[i * 2] / scale;
    ys[i] = imgPts[i * 2 + 1] / scale;
    if (ys[i] < minY) minY = ys[i];
    if (ys[i] > maxY) maxY = ys[i];
  }
  const y0 = Math.max(0, Math.ceil(minY));
  const y1 = Math.min(gh - 1, Math.floor(maxY));
  const xints: number[] = [];
  for (let y = y0; y <= y1; y++) {
    xints.length = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const yi = ys[i], yj = ys[j];
      if ((yi > y) !== (yj > y)) {
        xints.push(xs[i] + ((y - yi) / (yj - yi)) * (xs[j] - xs[i]));
      }
    }
    xints.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xints.length; k += 2) {
      const xa = Math.max(0, Math.ceil(xints[k]));
      const xb = Math.min(gw - 1, Math.floor(xints[k + 1]));
      for (let x = xa; x <= xb; x++) mask[y * gw + x] = v;
    }
  }
}

function fillRect(mask: Uint8Array, gw: number, gh: number, x: number, y: number, w: number, h: number, v: number): void {
  const x0 = Math.max(0, Math.floor(x));
  const y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(gw - 1, Math.ceil(x + w));
  const y1 = Math.min(gh - 1, Math.ceil(y + h));
  for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) mask[yy * gw + xx] = v;
}

function fillEllipse(mask: Uint8Array, gw: number, gh: number, cx: number, cy: number, rx: number, ry: number, v: number): void {
  const rxi = rx || 1, ryi = ry || 1;
  const x0 = Math.max(0, Math.floor(cx - rxi));
  const y0 = Math.max(0, Math.floor(cy - ryi));
  const x1 = Math.min(gw - 1, Math.ceil(cx + rxi));
  const y1 = Math.min(gh - 1, Math.ceil(cy + ryi));
  for (let yy = y0; yy <= y1; yy++) {
    const ny = (yy - cy) / ryi;
    for (let xx = x0; xx <= x1; xx++) {
      const nx = (xx - cx) / rxi;
      if (nx * nx + ny * ny <= 1) mask[yy * gw + xx] = v;
    }
  }
}

/** Stamp a round-capped thick polyline (image-coord points) with value *v*.
 *  `r` is the radius in GRID cells (i.e. image radius / scale).
 *
 *  `gate`, when given, restricts the stamp to cells where `gate[i]` is non-zero —
 *  this is what makes the Threshold Brush paint only inside its intensity band. */
export function stampStroke(mask: Uint8Array, gw: number, gh: number, imgPts: number[], r: number, scale: number, v: number, gate?: Uint8Array): void {
  const rad = Math.max(0.5, r);
  const r2 = rad * rad;
  const pts: number[] = [];
  for (let i = 0; i < imgPts.length; i++) pts.push(imgPts[i] / scale);
  const stampSeg = (ax: number, ay: number, bx: number, by: number) => {
    const minX = Math.max(0, Math.floor(Math.min(ax, bx) - rad));
    const maxX = Math.min(gw - 1, Math.ceil(Math.max(ax, bx) + rad));
    const minY = Math.max(0, Math.floor(Math.min(ay, by) - rad));
    const maxY = Math.min(gh - 1, Math.ceil(Math.max(ay, by) + rad));
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        let t = len2 ? ((x - ax) * dx + (y - ay) * dy) / len2 : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const cxp = ax + t * dx, cyp = ay + t * dy;
        const ddx = x - cxp, ddy = y - cyp;
        if (ddx * ddx + ddy * ddy > r2) continue;
        const i = y * gw + x;
        if (gate && !gate[i]) continue;
        mask[i] = v;
      }
    }
  };
  if (pts.length === 2) { stampSeg(pts[0], pts[1], pts[0], pts[1]); return; }
  for (let i = 0; i + 3 < pts.length; i += 2) stampSeg(pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
}
