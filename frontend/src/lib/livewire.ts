/**
 * Live-wire / "magnetic lasso" edge tracing.
 *
 * Builds a downsampled edge-cost map from the rendered slice (low cost on
 * strong image gradients), runs Dijkstra from a seed point, then backtracks
 * the least-cost path to the cursor — so the traced boundary hugs feature
 * edges. Coordinates are in/out in full-resolution IMAGE pixels.
 */

export interface CostMap {
  gw: number;
  gh: number;
  /** Image pixels per grid cell (downsample factor). */
  scale: number;
  /** Per-node traversal cost in [~0, 1]; ~0 on strong edges. */
  cost: Float32Array;
}

/** Build an edge-cost map from an image, downsampled so the long side <= maxDim. */
export function buildCostMap(
  image: HTMLImageElement,
  imgW: number,
  imgH: number,
  maxDim = 512,
): CostMap | null {
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
    return null; // tainted canvas (shouldn't happen for same-origin slices)
  }

  const gray = new Float32Array(gw * gh);
  for (let i = 0; i < gw * gh; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }

  // Sobel gradient magnitude.
  const grad = new Float32Array(gw * gh);
  let maxG = 0;
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
      if (m > maxG) maxG = m;
    }
  }

  const cost = new Float32Array(gw * gh);
  const inv = maxG > 0 ? 1 / maxG : 0;
  for (let i = 0; i < gw * gh; i++) {
    cost[i] = 1 - grad[i] * inv; // strong edge -> ~0 cost
  }
  return { gw, gh, scale, cost };
}

export function imageToGrid(cm: CostMap, x: number, y: number): number {
  const gx = Math.max(0, Math.min(cm.gw - 1, Math.floor(x / cm.scale)));
  const gy = Math.max(0, Math.min(cm.gh - 1, Math.floor(y / cm.scale)));
  return gy * cm.gw + gx;
}

/** Minimal binary min-heap keyed by float priority. */
class MinHeap {
  private ids: number[] = [];
  private keys: number[] = [];
  get size() { return this.ids.length; }
  push(id: number, key: number) {
    this.ids.push(id);
    this.keys.push(key);
    let i = this.ids.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.keys[p] <= this.keys[i]) break;
      this.swap(i, p);
      i = p;
    }
  }
  pop(): number {
    const n = this.ids.length;
    const top = this.ids[0];
    const lastId = this.ids.pop()!;
    const lastKey = this.keys.pop()!;
    if (n > 1) {
      this.ids[0] = lastId;
      this.keys[0] = lastKey;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = 2 * i + 2;
        let s = i;
        if (l < this.ids.length && this.keys[l] < this.keys[s]) s = l;
        if (r < this.ids.length && this.keys[r] < this.keys[s]) s = r;
        if (s === i) break;
        this.swap(i, s);
        i = s;
      }
    }
    return top;
  }
  private swap(a: number, b: number) {
    [this.ids[a], this.ids[b]] = [this.ids[b], this.ids[a]];
    [this.keys[a], this.keys[b]] = [this.keys[b], this.keys[a]];
  }
}

/** Dijkstra over the cost map from *seedIdx*; returns a back-pointer array. */
export function dijkstra(cm: CostMap, seedIdx: number): Int32Array {
  const { gw, gh, cost } = cm;
  const n = gw * gh;
  const dist = new Float32Array(n).fill(Infinity);
  const prev = new Int32Array(n).fill(-1);
  const done = new Uint8Array(n);
  dist[seedIdx] = 0;
  const heap = new MinHeap();
  heap.push(seedIdx, 0);

  while (heap.size > 0) {
    const u = heap.pop();
    if (done[u]) continue;
    done[u] = 1;
    const ux = u % gw;
    const uy = (u / gw) | 0;
    const du = dist[u];
    for (let dy = -1; dy <= 1; dy++) {
      const ny = uy + dy;
      if (ny < 0 || ny >= gh) continue;
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = ux + dx;
        if (nx < 0 || nx >= gw) continue;
        const v = ny * gw + nx;
        if (done[v]) continue;
        const diag = dx !== 0 && dy !== 0 ? Math.SQRT2 : 1;
        const w = (cost[v] + 0.02) * diag; // +base keeps weak edges traversable
        const nd = du + w;
        if (nd < dist[v]) {
          dist[v] = nd;
          prev[v] = u;
          heap.push(v, nd);
        }
      }
    }
  }
  return prev;
}

/**
 * Backtrack the least-cost path from *targetIdx* to the seed used to build
 * *prev*; returns flat [x0,y0,x1,y1,...] in image pixels (seed-first order).
 */
export function tracePath(cm: CostMap, prev: Int32Array, targetIdx: number): number[] {
  const grid: number[] = [];
  let cur = targetIdx;
  let guard = 0;
  const maxSteps = cm.gw * cm.gh;
  while (cur !== -1 && guard++ < maxSteps) {
    grid.push(cur);
    cur = prev[cur];
  }
  grid.reverse(); // seed -> target
  const half = cm.scale / 2;
  const out: number[] = [];
  for (const idx of grid) {
    const gx = idx % cm.gw;
    const gy = (idx / cm.gw) | 0;
    out.push(gx * cm.scale + half, gy * cm.scale + half);
  }
  return out;
}

/** Reduce a dense path to fewer vertices (keep every Nth point + endpoints). */
export function simplifyPath(points: number[], stride = 4): number[] {
  if (points.length <= 4) return points;
  const out: number[] = [];
  const nPts = points.length / 2;
  for (let i = 0; i < nPts; i++) {
    if (i % stride === 0 || i === nPts - 1) {
      out.push(points[i * 2], points[i * 2 + 1]);
    }
  }
  return out;
}
