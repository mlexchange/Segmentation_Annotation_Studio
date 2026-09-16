/**
 * Clipping must never silently degrade to "not clipped".
 *
 * The boolean clip path can fail on awkward geometry — and complex Threshold
 * Brush regions are exactly that. The old code returned `[]` from the union on
 * failure, which callers read as "nothing to clip against", so the new annotation
 * was committed overlapping its neighbour with no error anywhere. On a slice with
 * threshold-painted classes that made clip-to-other-classes look simply broken for
 * every subsequent annotation.
 *
 * These tests force the failure and assert the result is still clipped.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { clipShapesToOthers } from './clipToClasses';
import { unionShapesChecked } from './polybool';
import { fullResGridFor, rasterizeShapes } from './rasterize';
import { maskToPolygonsWithHoles } from './magicwand';
import type { Shape } from '@/stores/annotationStore';

const W = 256, H = 256;

afterEach(() => { vi.restoreAllMocks(); vi.resetModules(); });

/** Pixels where `a` and `b` both cover, ignoring shared-boundary slack. */
function realOverlap(a: Shape[], b: Shape[]): number {
  const { gw, gh, scale } = fullResGridFor(W, H);
  const ma = rasterizeShapes(a, gw, gh, scale);
  const mb = rasterizeShapes(b, gw, gh, scale);
  // Erode the intersection by requiring a 4-neighbourhood hit, so a shared edge
  // (which legitimately rasterizes into both) doesn't count as real overlap.
  let n = 0;
  for (let y = 1; y < gh - 1; y++) {
    for (let x = 1; x < gw - 1; x++) {
      const i = y * gw + x;
      if (!(ma[i] && mb[i])) continue;
      if (ma[i - 1] && mb[i - 1] && ma[i + 1] && mb[i + 1] &&
          ma[i - gw] && mb[i - gw] && ma[i + gw] && mb[i + gw]) n++;
    }
  }
  return n;
}

const otherClass: Shape = { id: 'o', classId: 2, kind: 'rectangle', x: 60, y: 60, w: 120, h: 120 };
const incoming: Shape = { id: 'n', classId: 1, kind: 'rectangle', x: 100, y: 100, w: 120, h: 120 };

describe('clip never silently degrades to unclipped', () => {
  it('clips normally when the boolean path works', () => {
    const res = clipShapesToOthers([incoming], [otherClass], W, H);
    expect(res.length).toBeGreaterThan(0);
    expect(realOverlap(res, [otherClass])).toBe(0);
  });

  it('still clips when polygon-clipping throws on every boolean op', async () => {
    // Simulate the failure mode: the library rejects this geometry outright.
    vi.doMock('polygon-clipping', () => ({
      default: {
        union: () => { throw new Error('boom'); },
        difference: () => { throw new Error('boom'); },
      },
    }));
    vi.resetModules();
    const { clipShapesToOthers: clipFresh } = await import('./clipToClasses');

    const res = clipFresh([incoming], [otherClass], W, H);
    expect(res.length).toBeGreaterThan(0);          // not dropped
    expect(realOverlap(res, [otherClass])).toBe(0); // and genuinely clipped
  });

  it('still clips when only the union fails', async () => {
    vi.doMock('polygon-clipping', () => ({
      default: {
        union: () => { throw new Error('boom'); },
        difference: (a: unknown) => a, // difference "works" but is a no-op
      },
    }));
    vi.resetModules();
    const { clipShapesToOthers: clipFresh } = await import('./clipToClasses');

    // Two other-class shapes so a union is actually required.
    const others: Shape[] = [
      otherClass,
      { id: 'o2', classId: 2, kind: 'rectangle', x: 150, y: 60, w: 60, h: 120 },
    ];
    const res = clipFresh([incoming], others, W, H);
    expect(realOverlap(res, others)).toBe(0);
  });
});

describe('unionShapesChecked reports failure instead of hiding it', () => {
  it('is ok for ordinary shapes', () => {
    const { mp, ok } = unionShapesChecked([otherClass], W, H);
    expect(ok).toBe(true);
    expect(mp.length).toBeGreaterThan(0);
  });

  it('reports ok:false when a union throws, keeping what it can', async () => {
    vi.doMock('polygon-clipping', () => ({
      default: { union: () => { throw new Error('boom'); }, difference: (a: unknown) => a },
    }));
    vi.resetModules();
    const { unionShapesChecked: checked } = await import('./polybool');

    const { mp, ok } = checked(
      [otherClass, { id: 'o2', classId: 2, kind: 'rectangle', x: 10, y: 10, w: 20, h: 20 }],
      W, H,
    );
    expect(ok).toBe(false);         // the caller can now react
    expect(mp.length).toBeGreaterThan(0); // and we kept the first shape's geometry
  });

  it('is ok:true (empty) when there is genuinely nothing to union', () => {
    const { mp, ok } = unionShapesChecked([], W, H);
    expect(ok).toBe(true);
    expect(mp).toEqual([]);
  });
});

describe('clipping against threshold-brush geometry', () => {
  /** Many small speckled regions with holes, as a threshold stroke produces. */
  function thresholdShapes(classId: number): Shape[] {
    const { gw, gh, scale } = fullResGridFor(W, H);
    const mask = new Uint8Array(gw * gh);
    let seed = 99;
    const rnd = () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let y = 40; y < 180; y++) for (let x = 40; x < 200; x++) if (rnd() > 0.45) mask[y * gw + x] = 1;
    return maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
      .filter((p) => p.points.length >= 6)
      .map((p, i) => ({
        id: `t${i}`, classId, kind: 'polygon' as const,
        points: p.points, ...(p.holes.length ? { holes: p.holes } : {}),
      }));
  }

  it('clips a new annotation against hundreds of speckled threshold regions', () => {
    const others = thresholdShapes(2);
    expect(others.length).toBeGreaterThan(50); // genuinely the hard case
    const res = clipShapesToOthers([incoming], others, W, H);
    expect(realOverlap(res, others)).toBe(0);
  });
});

describe('batched multi-shape clip matches per-shape clip', () => {
  /** The per-shape path, forced by clipping each shape in its own call. */
  function perShape(news: Shape[], slice: Shape[]): Shape[] {
    return news.flatMap((n) => clipShapesToOthers([n], slice, W, H));
  }

  const slice: Shape[] = [
    { id: 'o1', classId: 2, kind: 'rectangle', x: 40, y: 40, w: 90, h: 90 },
    { id: 'o2', classId: 2, kind: 'ellipse', cx: 180, cy: 170, rx: 45, ry: 30 },
    { id: 's1', classId: 1, kind: 'rectangle', x: 200, y: 30, w: 30, h: 30 },
  ];
  const news: Shape[] = [
    { id: 'n1', classId: 1, kind: 'rectangle', x: 80, y: 80, w: 80, h: 80 },
    { id: 'n2', classId: 1, kind: 'rectangle', x: 150, y: 140, w: 70, h: 70 },
    { id: 'n3', classId: 1, kind: 'rectangle', x: 30, y: 190, w: 50, h: 40 },
  ];

  it('covers the same region', () => {
    const batched = clipShapesToOthers(news, slice, W, H);
    const single = perShape(news, slice);
    const { gw, gh, scale } = fullResGridFor(W, H);
    const a = rasterizeShapes(batched, gw, gh, scale);
    const b = rasterizeShapes(single, gw, gh, scale);
    let diff = 0, set = 0;
    for (let i = 0; i < a.length; i++) { if (a[i] || b[i]) set++; if (a[i] !== b[i]) diff++; }
    expect(diff / Math.max(1, set)).toBeLessThan(0.01);
  });

  it('still excludes the other classes', () => {
    const batched = clipShapesToOthers(news, slice, W, H);
    expect(realOverlap(batched, slice.filter((s) => s.classId === 2))).toBe(0);
  });

  it('clips a 100+ region threshold commit correctly', () => {
    const { gw, gh, scale } = fullResGridFor(W, H);
    const mask = new Uint8Array(gw * gh);
    let seed = 3;
    const rnd = () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    for (let y = 30; y < 220; y++) for (let x = 30; x < 220; x++) if (rnd() > 0.35) mask[y * gw + x] = 1;
    const regions: Shape[] = maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
      .filter((p) => p.points.length >= 6)
      .map((p, i) => ({ id: `r${i}`, classId: 1, kind: 'polygon' as const,
        points: p.points, ...(p.holes.length ? { holes: p.holes } : {}) }));
    expect(regions.length).toBeGreaterThan(40); // genuinely a multi-region commit

    const blocker: Shape = { id: 'b', classId: 2, kind: 'rectangle', x: 90, y: 90, w: 80, h: 80 };
    const res = clipShapesToOthers(regions, [blocker], W, H);
    expect(res.length).toBeGreaterThan(0);
    expect(realOverlap(res, [blocker])).toBe(0);
  });
});
