/**
 * Client-side "magic wand" selection — runs entirely in the browser on the
 * rendered slice, so it's instant (no backend round-trip) and tolerance changes
 * update live.
 *
 * Pipeline: rendered image → downsampled grayscale field → flood-fill
 * (contiguous) or intensity-band (global) mask → connected components →
 * Moore-neighbor boundary trace → Douglas–Peucker simplify → polygons in
 * full-resolution IMAGE pixel coordinates.
 */
import { fillHoles } from '@/lib/morphology';

export interface GrayField {
  gw: number;
  gh: number;
  /** Image pixels per grid cell (downsample factor). */
  scale: number;
  gray: Float32Array;
  /** Sobel gradient magnitude normalised to ~[0,1] (edge barrier for flood). */
  grad?: Float32Array;
}

const MAX_POLYGONS = 300;

/** Sobel gradient magnitude, normalised to ~[0,1] so the edge threshold is
 * dataset-independent.
 *
 * Normalisation matters a lot for uniform (fully white/black) regions: a naive
 * sampled percentile lands near 0 there (edges are sparse), which would blow up
 * faint interior rendering noise into "walls" and trap the flood at the seed.
 * We instead use a full-array histogram percentile with a floor relative to the
 * strongest edge, so interior noise stays ~0 and real boundaries stay ~1. */
export function gradientField(gray: Float32Array, gw: number, gh: number): Float32Array {
  const grad = new Float32Array(gw * gh);
  let maxRaw = 0;
  for (let y = 1; y < gh - 1; y++) {
    for (let x = 1; x < gw - 1; x++) {
      const i = y * gw + x;
      const gx =
        -gray[i - gw - 1] - 2 * gray[i - 1] - gray[i + gw - 1] +
        gray[i - gw + 1] + 2 * gray[i + 1] + gray[i + gw + 1];
      const gy =
        -gray[i - gw - 1] - 2 * gray[i - gw] - gray[i - gw + 1] +
        gray[i + gw - 1] + 2 * gray[i + gw] + gray[i + gw + 1];
      const m = Math.hypot(gx, gy);
      grad[i] = m;
      if (m > maxRaw) maxRaw = m;
    }
  }
  if (maxRaw < 1e-6) return grad; // perfectly uniform → no edges, no walls

  // 99th-percentile via a 256-bin histogram over ALL pixels (robust + O(n)).
  const n = grad.length;
  const bins = new Int32Array(256);
  const toBin = 255 / maxRaw;
  for (let i = 0; i < n; i++) bins[(grad[i] * toBin) | 0]++;
  const target = 0.99 * n;
  let cum = 0;
  let p99 = maxRaw;
  for (let b = 0; b < 256; b++) {
    cum += bins[b];
    if (cum >= target) { p99 = (b / 255) * maxRaw; break; }
  }
  // Floor at 20% of the max edge so a uniform region (p99≈0) doesn't over-wall.
  const denom = Math.max(p99, 0.2 * maxRaw);
  const inv = 1 / denom;
  for (let i = 0; i < n; i++) grad[i] = Math.min(1, grad[i] * inv);
  return grad;
}

/** Build a grayscale + gradient field from an image (long side ≤ maxDim).
 *
 * 1600 keeps a 2560px slice at half-resolution (scale 2) — plenty of detail for
 * the edge-aware flood while keeping the per-click work ~4x cheaper than full res. */
export function buildField(
  image: HTMLImageElement,
  imgW: number,
  imgH: number,
  maxDim = 1600,
): GrayField | null {
  const scale = Math.max(1, Math.ceil(Math.max(imgW, imgH) / maxDim));
  const gw = Math.max(1, Math.floor(imgW / scale));
  const gh = Math.max(1, Math.floor(imgH / scale));
  const canvas = document.createElement('canvas');
  canvas.width = gw;
  canvas.height = gh;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(image, 0, 0, gw, gh);
  let data: Uint8ClampedArray;
  try {
    data = ctx.getImageData(0, 0, gw, gh).data;
  } catch {
    return null; // tainted canvas
  }
  const gray = new Float32Array(gw * gh);
  for (let i = 0; i < gw * gh; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  return { gw, gh, scale, gray, grad: gradientField(gray, gw, gh) };
}

/** Robust intensity spread (2nd–98th percentile) for tolerance scaling. */
function spread(gray: Float32Array): number {
  const n = gray.length;
  const step = Math.max(1, Math.floor(n / 10000));
  const s: number[] = [];
  for (let i = 0; i < n; i += step) s.push(gray[i]);
  s.sort((a, b) => a - b);
  const p2 = s[Math.floor(s.length * 0.02)] ?? s[0] ?? 0;
  const p98 = s[Math.floor(s.length * 0.98)] ?? s[s.length - 1] ?? 1;
  return Math.max(1, p98 - p2);
}

/** Cheap separable box blur (approx Gaussian) applied in-place-ish; returns new array. */
function boxBlur(gray: Float32Array, gw: number, gh: number, radius: number): Float32Array {
  if (radius < 1) return gray;
  const r = Math.round(radius);
  const tmp = new Float32Array(gray.length);
  const out = new Float32Array(gray.length);
  // horizontal
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let sum = 0, cnt = 0;
      for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < gw) { sum += gray[y * gw + xx]; cnt++; }
      }
      tmp[y * gw + x] = sum / cnt;
    }
  }
  // vertical
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      let sum = 0, cnt = 0;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < gh) { sum += tmp[yy * gw + x]; cnt++; }
      }
      out[y * gw + x] = sum / cnt;
    }
  }
  return out;
}

/** Perpendicular-distance polyline simplification (Douglas–Peucker). */
function simplify(pts: Array<[number, number]>, tol: number): Array<[number, number]> {
  if (pts.length < 3 || tol <= 0) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;
  const stack: Array<[number, number]> = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    let maxD = -1;
    let idx = -1;
    const [ax, ay] = pts[a];
    const [bx, by] = pts[b];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = pts[i];
      const d = Math.abs((px - ax) * dy - (py - ay) * dx) / len;
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > tol && idx > 0) {
      keep[idx] = 1;
      stack.push([a, idx], [idx, b]);
    }
  }
  return pts.filter((_, i) => keep[i]);
}

/** Chaikin corner-cutting: rounds a closed polygon (more iterations = smoother). */
function chaikin(pts: Array<[number, number]>, iterations: number): Array<[number, number]> {
  let p = pts;
  for (let it = 0; it < iterations; it++) {
    if (p.length < 3) break;
    const out: Array<[number, number]> = [];
    const n = p.length;
    for (let i = 0; i < n; i++) {
      const [x0, y0] = p[i];
      const [x1, y1] = p[(i + 1) % n];
      out.push([0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1]);
      out.push([0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1]);
    }
    p = out;
  }
  return p;
}

const NBR: Array<[number, number]> = [
  [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1],
]; // clockwise starting from West

/** Moore-neighbor boundary trace of the component with `labels[idx] === lbl`. */
function traceContour(
  labels: Int32Array, gw: number, gh: number, lbl: number, startIdx: number,
): Array<[number, number]> {
  const inside = (x: number, y: number) =>
    x >= 0 && x < gw && y >= 0 && y < gh && labels[y * gw + x] === lbl;

  const sx = startIdx % gw;
  const sy = (startIdx / gw) | 0;
  const contour: Array<[number, number]> = [[sx, sy]];
  let px = sx, py = sy;
  let bx = sx - 1, by = sy; // backtrack (came from the west background cell)
  const maxSteps = gw * gh * 8;

  for (let step = 0; step < maxSteps; step++) {
    let di = NBR.findIndex((d) => px + d[0] === bx && py + d[1] === by);
    if (di < 0) di = 0;
    let found = false;
    for (let k = 1; k <= 8; k++) {
      const idx = (di + k) % 8;
      const nx = px + NBR[idx][0];
      const ny = py + NBR[idx][1];
      if (inside(nx, ny)) {
        // new backtrack = the (background) cell examined just before this one
        bx = px + NBR[(idx + 7) % 8][0];
        by = py + NBR[(idx + 7) % 8][1];
        px = nx; py = ny;
        found = true;
        break;
      }
    }
    if (!found) break; // isolated pixel
    if (px === sx && py === sy) break; // returned to start
    contour.push([px, py]);
  }
  return contour;
}

interface SelectOpts {
  toleranceFrac: number;
  mode: 'contiguous' | 'global';
  /** 0–~5 edge smoothing: denoises the field, then rounds the contour. */
  smooth?: number;
  /** 0–1 edge barrier (contiguous only): higher = flood stops at weaker edges. */
  edgeStop?: number;
  minRegion?: number;  // min component size in grid pixels
}

interface MaskPolyOpts {
  /** Min component size, in mask pixels, to keep. */
  minRegion?: number;
  /** 0–~5: Douglas–Peucker tolerance + Chaikin rounding strength on the outline. */
  smooth?: number;
  /** Mask pixels per output image pixel (1 if the mask is already image-resolution). */
  scale?: number;
}

/**
 * Convert a binary mask into simplified, smoothed polygon contours in IMAGE
 * coordinates. Shared by the classic wand (`magicSelect`) and the SAM engine,
 * which both produce a `Uint8Array` mask and want editable polygon shapes.
 *
 * Labels 4-connected components, Moore-neighbor traces each, simplifies the
 * staircase with Douglas–Peucker, then rounds corners with Chaikin. Components
 * smaller than `minRegion` are dropped; output is capped at `MAX_POLYGONS`
 * (largest first).
 */
export function maskToPolygons(
  mask: Uint8Array,
  gw: number,
  gh: number,
  { minRegion = 12, smooth = 0, scale = 1 }: MaskPolyOpts = {},
): number[][] {
  const chaikinIters = Math.min(Math.round(smooth), 4);
  const dpTol = 1 + smooth * 0.6;

  // Label 4-connected components; remember each one's first (topmost-left) pixel.
  const labels = new Int32Array(gw * gh).fill(0);
  const starts: number[] = [];
  const sizes: number[] = [];
  let next = 1;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p] || labels[p]) continue;
    const lbl = next++;
    starts[lbl] = p;
    let size = 0;
    const stack = [p];
    labels[p] = lbl;
    while (stack.length) {
      const idx = stack.pop()!;
      size++;
      const x = idx % gw, y = (idx / gw) | 0;
      const push = (j: number) => { if (mask[j] && !labels[j]) { labels[j] = lbl; stack.push(j); } };
      if (x > 0) push(idx - 1);
      if (x < gw - 1) push(idx + 1);
      if (y > 0) push(idx - gw);
      if (y < gh - 1) push(idx + gw);
    }
    sizes[lbl] = size;
  }

  const polys: Array<{ area: number; flat: number[] }> = [];
  for (let lbl = 1; lbl < next; lbl++) {
    if (sizes[lbl] < minRegion) continue;
    let contour = traceContour(labels, gw, gh, lbl, starts[lbl]);
    contour = simplify(contour, dpTol);     // drop staircase collinear points
    contour = chaikin(contour, chaikinIters); // round the remaining corners
    if (contour.length < 3) continue;
    const flat: number[] = [];
    for (const [x, y] of contour) {
      flat.push(Math.round((x + 0.5) * scale * 100) / 100, Math.round((y + 0.5) * scale * 100) / 100);
    }
    polys.push({ area: sizes[lbl], flat });
  }

  polys.sort((a, b) => b.area - a.area);
  return polys.slice(0, MAX_POLYGONS).map((p) => p.flat);
}

/** Even-odd point-in-polygon test on a flat [x,y,…] ring. */
function pointInPolygonFlat(px: number, py: number, pts: number[]): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 2; i < pts.length; j = i, i += 2) {
    const xi = pts[i], yi = pts[i + 1];
    const xj = pts[j], yj = pts[j + 1];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * Like `maskToPolygons`, but also returns each region's enclosed voids as inner
 * rings ("holes"). Enclosed background (not connected to the image border) is
 * detected via `fillHoles`, vectorized, and each void assigned to the outer
 * polygon that contains it. Outer + hole rings are in IMAGE coordinates.
 */
export function maskToPolygonsWithHoles(
  mask: Uint8Array,
  gw: number,
  gh: number,
  opts: MaskPolyOpts = {},
): Array<{ points: number[]; holes: number[][] }> {
  const outers = maskToPolygons(mask, gw, gh, opts);
  const result = outers.map((points) => ({ points, holes: [] as number[][] }));
  if (result.length === 0) return result;

  // Enclosed background = pixels fillHoles would fill (interior voids only).
  const filled = fillHoles(mask, gw, gh);
  const holeMask = new Uint8Array(mask.length);
  let anyHole = false;
  for (let i = 0; i < mask.length; i++) {
    if (filled[i] && !mask[i]) { holeMask[i] = 1; anyHole = true; }
  }
  if (!anyHole) return result;

  const holeRings = maskToPolygons(holeMask, gw, gh, opts);
  for (const ring of holeRings) {
    // Single region (the common clip case): the hole is by construction inside
    // the only outer, so assign directly — a centroid test can misfire on a
    // concave outer (e.g. an irregular SAM/magic outline) and wrongly drop it.
    let owner = result.length === 1 ? result[0] : undefined;
    if (!owner) {
      let cx = 0, cy = 0;
      const n = ring.length / 2;
      for (let i = 0; i < ring.length; i += 2) { cx += ring[i]; cy += ring[i + 1]; }
      cx /= n; cy /= n;
      owner = result.find((r) => pointInPolygonFlat(cx, cy, r.points))
        // Fallback for concave outers: the outer whose bbox contains the ring.
        ?? result.find((r) => bboxContainsRing(r.points, ring));
    }
    if (owner) owner.holes.push(ring);
  }
  return result;
}

/** True if `outer`'s bounding box fully contains `ring`'s bounding box. */
function bboxContainsRing(outer: number[], ring: number[]): boolean {
  let ox0 = Infinity, oy0 = Infinity, ox1 = -Infinity, oy1 = -Infinity;
  for (let i = 0; i < outer.length; i += 2) {
    if (outer[i] < ox0) ox0 = outer[i];
    if (outer[i] > ox1) ox1 = outer[i];
    if (outer[i + 1] < oy0) oy0 = outer[i + 1];
    if (outer[i + 1] > oy1) oy1 = outer[i + 1];
  }
  let rx0 = Infinity, ry0 = Infinity, rx1 = -Infinity, ry1 = -Infinity;
  for (let i = 0; i < ring.length; i += 2) {
    if (ring[i] < rx0) rx0 = ring[i];
    if (ring[i] > rx1) rx1 = ring[i];
    if (ring[i + 1] < ry0) ry0 = ring[i + 1];
    if (ring[i + 1] > ry1) ry1 = ring[i + 1];
  }
  return rx0 >= ox0 && ry0 >= oy0 && rx1 <= ox1 && ry1 <= oy1;
}

/**
 * Select similar pixels from `seed` and return polygons in IMAGE coordinates.
 */
export function magicSelect(
  field: GrayField,
  seedXimg: number,
  seedYimg: number,
  { toleranceFrac, mode, smooth = 0, edgeStop = 0, minRegion = 12 }: SelectOpts,
): number[][] {
  const { gw, gh, scale, grad } = field;
  // Smoothing drives a pre-blur (denoise so the boundary is less ragged) here;
  // the Douglas–Peucker + Chaikin contour smoothing happens in maskToPolygons.
  const blurR = Math.min(Math.round(smooth), 3);
  const gray = blurR > 0 ? boxBlur(field.gray, gw, gh, blurR) : field.gray;
  const sx = Math.max(0, Math.min(gw - 1, Math.floor(seedXimg / scale)));
  const sy = Math.max(0, Math.min(gh - 1, Math.floor(seedYimg / scale)));
  const seedVal = gray[sy * gw + sx];
  const tolAbs = Math.max(0, toleranceFrac) * spread(field.gray);

  // Edge barrier: pixels whose normalised gradient exceeds this are walls the
  // flood won't cross — keeps a void's selection bounded by its rim instead of
  // leaking across a soft/ringy edge. Disabled when edgeStop is 0 or no grad.
  const wallLimit = edgeStop > 0 ? 1 - edgeStop : Infinity;
  const isWall = (i: number) => grad !== undefined && grad[i] >= wallLimit;

  const mask = new Uint8Array(gw * gh);
  if (mode === 'global') {
    for (let i = 0; i < mask.length; i++) {
      if (Math.abs(gray[i] - seedVal) <= tolAbs) mask[i] = 1;
    }
  } else {
    // Flood fill from the seed (4-connected): stay within tol of the seed value
    // and don't expand into edge (wall) pixels. The seed itself is always kept.
    const seedIdx = sy * gw + sx;
    const stack = [seedIdx];
    while (stack.length) {
      const idx = stack.pop()!;
      if (mask[idx]) continue;
      if (Math.abs(gray[idx] - seedVal) > tolAbs) continue;
      if (idx !== seedIdx && isWall(idx)) continue;
      mask[idx] = 1;
      const x = idx % gw, y = (idx / gw) | 0;
      if (x > 0) stack.push(idx - 1);
      if (x < gw - 1) stack.push(idx + 1);
      if (y > 0) stack.push(idx - gw);
      if (y < gh - 1) stack.push(idx + gw);
    }
  }

  return maskToPolygons(mask, gw, gh, { minRegion, smooth, scale });
}
