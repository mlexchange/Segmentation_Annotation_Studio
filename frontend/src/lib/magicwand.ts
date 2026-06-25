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

export interface GrayField {
  gw: number;
  gh: number;
  /** Image pixels per grid cell (downsample factor). */
  scale: number;
  gray: Float32Array;
}

const MAX_POLYGONS = 300;

/** Build a downsampled grayscale field from an image (long side ≤ maxDim). */
export function buildField(
  image: HTMLImageElement,
  imgW: number,
  imgH: number,
  maxDim = 1400,
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
  return { gw, gh, scale, gray };
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
  blur?: number;       // approx Gaussian radius (0 = off)
  minRegion?: number;  // min component size in grid pixels
  simplifyTol?: number;
}

/**
 * Select similar pixels from `seed` and return polygons in IMAGE coordinates.
 */
export function magicSelect(
  field: GrayField,
  seedXimg: number,
  seedYimg: number,
  { toleranceFrac, mode, blur = 0, minRegion = 12, simplifyTol = 1.2 }: SelectOpts,
): number[][] {
  const { gw, gh, scale } = field;
  const gray = blur > 0 ? boxBlur(field.gray, gw, gh, blur) : field.gray;
  const sx = Math.max(0, Math.min(gw - 1, Math.floor(seedXimg / scale)));
  const sy = Math.max(0, Math.min(gh - 1, Math.floor(seedYimg / scale)));
  const seedVal = gray[sy * gw + sx];
  const tolAbs = Math.max(0, toleranceFrac) * spread(field.gray);

  const mask = new Uint8Array(gw * gh);
  if (mode === 'global') {
    for (let i = 0; i < mask.length; i++) {
      if (Math.abs(gray[i] - seedVal) <= tolAbs) mask[i] = 1;
    }
  } else {
    // Flood fill from the seed (4-connected), staying within tol of the seed value.
    const stack = [sy * gw + sx];
    while (stack.length) {
      const idx = stack.pop()!;
      if (mask[idx]) continue;
      if (Math.abs(gray[idx] - seedVal) > tolAbs) continue;
      mask[idx] = 1;
      const x = idx % gw, y = (idx / gw) | 0;
      if (x > 0) stack.push(idx - 1);
      if (x < gw - 1) stack.push(idx + 1);
      if (y > 0) stack.push(idx - gw);
      if (y < gh - 1) stack.push(idx + gw);
    }
  }

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
    contour = simplify(contour, simplifyTol);
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
