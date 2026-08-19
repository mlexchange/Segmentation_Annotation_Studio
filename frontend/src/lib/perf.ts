/**
 * perf — opt-in timing instrumentation for the annotation workspace.
 *
 * Enabled by `?perf=1` in the URL (sticky for the session) or by setting
 * `localStorage.perf = '1'`; disabled by `?perf=0`. When off, `time()` is a
 * function call plus a boolean test and `mark()` returns immediately, so
 * instrumented paths can stay in the hot code without measurable cost.
 *
 * Exists because the interesting slices are the user's, not any we can synthesize:
 * the only way to confirm an optimization on a 4000² volume with a thousand shapes
 * is to measure it there. Rolling p50/p95 rather than a single number, because the
 * costs that hurt are the occasional long frames, not the average.
 */

const SAMPLE_LIMIT = 120;

export type PerfLabel =
  | 'commit'          // full stroke/shape commit (clip + merge + store write)
  | 'clip'            // clipShapesToOthers
  | 'merge'           // mergeNewWithSameClass
  | 'layer-cache'     // Konva shapes-layer cache() rebuild
  | 'threshold-field' // full-resolution threshold field build
  | 'overlay-field'   // downsampled overlay field build
  | 'overlay-paint'   // threshold overlay repaint
  | 'magic-field'     // magic-wand gray field build
  | 'cost-map';       // livewire cost map build

interface Series {
  samples: number[];
  count: number;
  /** Ring-buffer write position. */
  pos: number;
}

const series = new Map<PerfLabel, Series>();
let enabled = false;
/** Bumped on every record so subscribers can re-read cheaply. */
let version = 0;
const listeners = new Set<() => void>();

/** Read the enable flag from the URL / localStorage. Safe to call repeatedly. */
export function initPerf(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const param = new URLSearchParams(window.location.search).get('perf');
    if (param === '1') window.localStorage.setItem('perf', '1');
    if (param === '0') window.localStorage.removeItem('perf');
    enabled = window.localStorage.getItem('perf') === '1';
  } catch {
    enabled = false; // private mode / storage denied
  }
  return enabled;
}

export function perfEnabled(): boolean {
  return enabled;
}

// Notifications are throttled: samples can arrive many times per frame during a
// stroke, and re-rendering the HUD on each one would perturb the very timings it
// reports. Recording stays exact; only the display lags by up to NOTIFY_MS.
const NOTIFY_MS = 250;
let notifyTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleNotify(): void {
  if (notifyTimer != null || listeners.size === 0) return;
  notifyTimer = setTimeout(() => {
    notifyTimer = null;
    for (const fn of listeners) fn();
  }, NOTIFY_MS);
}

/** Record one timing sample (ms). No-op when disabled. */
export function mark(label: PerfLabel, ms: number): void {
  if (!enabled) return;
  let s = series.get(label);
  if (!s) { s = { samples: new Array(SAMPLE_LIMIT).fill(0), count: 0, pos: 0 }; series.set(label, s); }
  s.samples[s.pos] = ms;
  s.pos = (s.pos + 1) % SAMPLE_LIMIT;
  if (s.count < SAMPLE_LIMIT) s.count++;
  version++;
  scheduleNotify();
}

/** Time a synchronous function, recording under `label`. Returns its result. */
export function time<T>(label: PerfLabel, fn: () => T): T {
  if (!enabled) return fn();
  const t0 = performance.now();
  try {
    return fn();
  } finally {
    mark(label, performance.now() - t0);
  }
}

export interface PerfStat {
  label: PerfLabel;
  count: number;
  p50: number;
  p95: number;
  last: number;
}

/** Snapshot of every recorded series, sorted slowest-p95 first. */
export function snapshot(): PerfStat[] {
  const out: PerfStat[] = [];
  for (const [label, s] of series) {
    if (s.count === 0) continue;
    const vals = s.samples.slice(0, s.count).sort((a, b) => a - b);
    const at = (q: number) => vals[Math.min(vals.length - 1, Math.floor(q * vals.length))];
    const lastIdx = (s.pos - 1 + SAMPLE_LIMIT) % SAMPLE_LIMIT;
    out.push({ label, count: s.count, p50: at(0.5), p95: at(0.95), last: s.samples[lastIdx] });
  }
  return out.sort((a, b) => b.p95 - a.p95);
}

/** Subscribe to new samples (for useSyncExternalStore). */
export function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function getVersion(): number {
  return version;
}

export function resetPerf(): void {
  series.clear();
  version++;
  for (const fn of listeners) fn();
}
