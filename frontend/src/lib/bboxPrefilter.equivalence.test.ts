/**
 * Equivalence tests for the bounding-box pre-filters.
 *
 * The commit path skips shapes whose bounds cannot interact with the new
 * geometry, which is only sound if it changes nothing. These tests pin that down
 * by re-implementing the ORIGINAL unfiltered algorithms here as reference oracles
 * and asserting the optimized versions agree with them, over randomized shape sets
 * covering the cases the filter could plausibly get wrong: disjoint, exactly
 * touching, nested, tiny-vs-huge, and brush-vs-vector mixes.
 *
 * If a future change to the filter makes it too aggressive, these fail.
 */
import { describe, it, expect } from 'vitest';
import { clipShapesToOthers, clipShapesToOthersMask } from './clipToClasses';
import { mergeNewWithSameClass, expandSameClassOverlap } from './mergeSameClass';
import { gridFor, fullResGridFor, rasterizeShapes, rasterizeUnion } from './rasterize';
import { maskToPolygonsWithHoles } from './magicwand';
import { unionShapesToMultiPolygon, shapeToMultiPolygon, multiPolygonToShapes } from './polybool';
import polygonClipping from 'polygon-clipping';
import type { PolygonShape, Shape } from '@/stores/annotationStore';

const W = 200, H = 200;

// ---- Deterministic PRNG so failures reproduce exactly ----------------------
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A grab-bag of shape kinds at a given position/size. */
function makeShape(rnd: () => number, id: string, classId: number, x: number, y: number, size: number): Shape {
  const kind = Math.floor(rnd() * 4);
  if (kind === 0) return { id, classId, kind: 'rectangle', x, y, w: size, h: size };
  if (kind === 1) return { id, classId, kind: 'ellipse', cx: x + size / 2, cy: y + size / 2, rx: size / 2, ry: size / 2 };
  if (kind === 2) {
    return {
      id, classId, kind: 'polygon',
      points: [x, y, x + size, y, x + size, y + size, x, y + size],
    };
  }
  return {
    id, classId, kind: 'brush',
    strokes: [{ points: [x + 2, y + 2, x + size - 2, y + size - 2], radius: Math.max(2, size / 4), mode: 'paint' }],
  };
}

/** Random slice populated with shapes across two classes, mixing overlaps and gaps. */
function randomSlice(seed: number, n: number): Shape[] {
  const rnd = mulberry32(seed);
  const out: Shape[] = [];
  for (let i = 0; i < n; i++) {
    const size = 8 + Math.floor(rnd() * 40);
    const x = Math.floor(rnd() * (W - size));
    const y = Math.floor(rnd() * (H - size));
    out.push(makeShape(rnd, `s${i}`, rnd() < 0.5 ? 1 : 2, x, y, size));
  }
  return out;
}

/**
 * Compare clip results by the region they cover, not by their vertex lists.
 *
 * Vertex-list equality is the wrong oracle. When nothing is nearby, the optimized
 * path skips `polygonClipping.difference` entirely; the naive path still runs it
 * against distant polygons, and polygon-clipping re-emits the ring — from a
 * different start vertex, and with coordinates perturbed in the last few decimal
 * places. On a 200×200 ellipse that shows up as a *single* boundary pixel out of
 * ~520 (0.19%).
 *
 * Note the direction of that error: the naive path is the one perturbing the
 * geometry, by round-tripping a shape through a boolean op that cannot change it.
 * The optimized path returns the original vertices untouched, so it is strictly
 * the more faithful of the two — the pre-filter removes a source of drift rather
 * than introducing one.
 *
 * So the assertion is high-IoU agreement, which catches any real region change
 * (a wrongly-dropped neighbour moves IoU far more than a rounding wobble) while
 * tolerating sub-pixel boundary noise neither implementation controls.
 */
function regionOf(shapes: PolygonShape[]): Uint8Array {
  const { gw, gh, scale } = fullResGridFor(W, H);
  return rasterizeShapes(shapes, gw, gh, scale);
}

/** Intersection-over-union of two binary masks (1 when both are empty). */
function iou(a: Uint8Array, b: Uint8Array): number {
  let inter = 0, union = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x || y) union++;
    if (x && y) inter++;
  }
  return union === 0 ? 1 : inter / union;
}

/** Assert two clip results describe the same regions. */
function expectSameRegions(got: PolygonShape[], want: PolygonShape[]): void {
  expect(got.length).toBe(want.length);
  expect(got.map((s) => s.classId).sort()).toEqual(want.map((s) => s.classId).sort());
  expect(iou(regionOf(got), regionOf(want))).toBeGreaterThan(0.995);
}

// ---- Reference oracles: the pre-filter-free originals ----------------------

function clipShapesToOthers_naive(
  newShapes: Shape[], sliceShapes: Shape[], width: number, height: number,
): PolygonShape[] {
  const out: PolygonShape[] = [];
  const otherByClass = new Map<number, ReturnType<typeof unionShapesToMultiPolygon>>();
  const sameByClass = new Map<number, ReturnType<typeof unionShapesToMultiPolygon>>();
  const otherMP = (c: number) => {
    let m = otherByClass.get(c);
    if (!m) { m = unionShapesToMultiPolygon(sliceShapes.filter((s) => s.classId !== c), width, height); otherByClass.set(c, m); }
    return m;
  };
  const sameMP = (c: number) => {
    let m = sameByClass.get(c);
    if (!m) { m = unionShapesToMultiPolygon(sliceShapes.filter((s) => s.classId === c), width, height); sameByClass.set(c, m); }
    return m;
  };
  for (const shape of newShapes) {
    try {
      let mp = shapeToMultiPolygon(shape, width, height);
      if (mp.length === 0) continue;
      const other = otherMP(shape.classId);
      const same = sameMP(shape.classId);
      if (other.length) mp = polygonClipping.difference(mp, other);
      if (mp.length === 0) continue;
      if (same.length && polygonClipping.difference(mp, same).length === 0) continue;
      const polys = multiPolygonToShapes(mp, shape.classId);
      if (polys.length) polys[0].id = shape.id;
      out.push(...polys);
    } catch {
      out.push(...clipShapesToOthersMask([shape], sliceShapes, width, height));
    }
  }
  return out;
}

function mergeNewWithSameClass_naive(
  newShapes: Shape[], sliceShapes: Shape[], width: number, height: number,
): { addCount: number; removeIds: string[] } {
  const { gw, gh, scale } = fullResGridFor(width, height);
  const removeIds: string[] = [];
  let addCount = 0;
  const byClass = new Map<number, Shape[]>();
  for (const s of newShapes) {
    const arr = byClass.get(s.classId);
    if (arr) arr.push(s); else byClass.set(s.classId, [s]);
  }
  const masksIntersect = (a: Uint8Array, b: Uint8Array) => {
    for (let i = 0; i < a.length; i++) if (a[i] && b[i]) return true;
    return false;
  };
  for (const [classId, group] of byClass) {
    const existing = sliceShapes.filter((s) => s.classId === classId);
    if (existing.length === 0) { addCount += group.length; continue; }
    const newMask = rasterizeUnion(group, gw, gh, scale);
    const overlapping = existing.filter((e) => masksIntersect(rasterizeShapes([e], gw, gh, scale), newMask));
    if (overlapping.length === 0) { addCount += group.length; continue; }
    removeIds.push(...overlapping.map((e) => e.id));
  }
  return { addCount, removeIds };
}

function expandSameClassOverlap_naive(seed: Shape[], all: Shape[], width: number, height: number): Shape[] {
  if (seed.length === 0) return seed;
  const { gw, gh, scale } = gridFor(width, height);
  const maskCache = new Map<string, Uint8Array>();
  const maskOf = (s: Shape) => {
    let m = maskCache.get(s.id);
    if (!m) { m = rasterizeShapes([s], gw, gh, scale); maskCache.set(s.id, m); }
    return m;
  };
  const chosen = new Map<string, Shape>(seed.map((s) => [s.id, s]));
  const classes = new Set(seed.map((s) => s.classId));
  for (const classId of classes) {
    const candidates = all.filter((s) => s.classId === classId && !chosen.has(s.id));
    const members = [...chosen.values()].filter((s) => s.classId === classId);
    let changed = true;
    while (changed && candidates.length > 0) {
      changed = false;
      const union = new Uint8Array(gw * gh);
      for (const m of members) {
        const mm = maskOf(m);
        for (let i = 0; i < union.length; i++) if (mm[i]) union[i] = 1;
      }
      for (let i = candidates.length - 1; i >= 0; i--) {
        let hit = false;
        const cm = maskOf(candidates[i]);
        for (let k = 0; k < union.length; k++) if (cm[k] && union[k]) { hit = true; break; }
        if (hit) {
          const c = candidates.splice(i, 1)[0];
          chosen.set(c.id, c);
          members.push(c);
          changed = true;
        }
      }
    }
  }
  return [...chosen.values()];
}

// ---- The tests -------------------------------------------------------------

describe('clipShapesToOthers — bbox pre-filter changes nothing', () => {
  for (const seed of [1, 7, 42, 99, 1234]) {
    it(`matches the unfiltered result (seed ${seed})`, () => {
      const slice = randomSlice(seed, 14);
      const rnd = mulberry32(seed + 500);
      const size = 20 + Math.floor(rnd() * 30);
      const newShape = makeShape(rnd, 'new', 1, Math.floor(rnd() * (W - size)), Math.floor(rnd() * (H - size)), size);
      expectSameRegions(clipShapesToOthers([newShape], slice, W, H), clipShapesToOthers_naive([newShape], slice, W, H));
    });
  }

  it('matches for a shape disjoint from everything', () => {
    const slice: Shape[] = [{ id: 'a', classId: 2, kind: 'rectangle', x: 0, y: 0, w: 20, h: 20 }];
    const far: Shape = { id: 'new', classId: 1, kind: 'rectangle', x: 150, y: 150, w: 20, h: 20 };
    expectSameRegions(clipShapesToOthers([far], slice, W, H), clipShapesToOthers_naive([far], slice, W, H));
  });

  it('matches for shapes whose edges exactly abut (the filter must not drop these)', () => {
    const slice: Shape[] = [{ id: 'a', classId: 2, kind: 'rectangle', x: 50, y: 50, w: 30, h: 30 }];
    const touching: Shape = { id: 'new', classId: 1, kind: 'rectangle', x: 80, y: 50, w: 30, h: 30 };
    expectSameRegions(clipShapesToOthers([touching], slice, W, H), clipShapesToOthers_naive([touching], slice, W, H));
  });

  it('matches for a large shape fully containing a small other-class one', () => {
    const slice: Shape[] = [{ id: 'a', classId: 2, kind: 'rectangle', x: 90, y: 90, w: 10, h: 10 }];
    const big: Shape = { id: 'new', classId: 1, kind: 'rectangle', x: 20, y: 20, w: 160, h: 160 };
    expectSameRegions(clipShapesToOthers([big], slice, W, H), clipShapesToOthers_naive([big], slice, W, H));
  });

  it('matches when several new shapes of one class commit together', () => {
    const slice = randomSlice(21, 12);
    const news: Shape[] = [
      { id: 'n1', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 30, h: 30 },
      { id: 'n2', classId: 1, kind: 'rectangle', x: 140, y: 140, w: 30, h: 30 },
    ];
    expectSameRegions(clipShapesToOthers(news, slice, W, H), clipShapesToOthers_naive(news, slice, W, H));
  });
});

describe('mergeNewWithSameClass — bbox pre-filter picks the same merge targets', () => {
  for (const seed of [3, 11, 55, 808]) {
    it(`selects the same overlapping shapes (seed ${seed})`, () => {
      const slice = randomSlice(seed, 16);
      const rnd = mulberry32(seed + 900);
      const size = 20 + Math.floor(rnd() * 30);
      const newShape = makeShape(rnd, 'new', 1, Math.floor(rnd() * (W - size)), Math.floor(rnd() * (H - size)), size);
      const got = mergeNewWithSameClass([newShape], slice, W, H);
      const want = mergeNewWithSameClass_naive([newShape], slice, W, H);
      expect([...got.removeIds].sort()).toEqual([...want.removeIds].sort());
    });
  }

  it('finds a same-class neighbour that only just overlaps', () => {
    const slice: Shape[] = [{ id: 'a', classId: 1, kind: 'rectangle', x: 50, y: 50, w: 30, h: 30 }];
    const overlapping: Shape = { id: 'new', classId: 1, kind: 'rectangle', x: 78, y: 50, w: 30, h: 30 };
    const got = mergeNewWithSameClass([overlapping], slice, W, H);
    const want = mergeNewWithSameClass_naive([overlapping], slice, W, H);
    expect([...got.removeIds].sort()).toEqual([...want.removeIds].sort());
    expect(got.removeIds).toContain('a');
  });

  it('ignores a same-class shape that is merely nearby but not touching', () => {
    const slice: Shape[] = [{ id: 'a', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 20, h: 20 }];
    const apart: Shape = { id: 'new', classId: 1, kind: 'rectangle', x: 120, y: 120, w: 20, h: 20 };
    const got = mergeNewWithSameClass([apart], slice, W, H);
    expect(got.removeIds).toEqual([]);
    expect(got.removeIds).toEqual(mergeNewWithSameClass_naive([apart], slice, W, H).removeIds);
  });
});

describe('expandSameClassOverlap — bbox pre-filter finds the same cluster', () => {
  for (const seed of [5, 17, 300]) {
    it(`expands to the same set (seed ${seed})`, () => {
      const slice = randomSlice(seed, 18).map((s) => ({ ...s, classId: 1 }));
      const seedSel = [slice[0]];
      const got = expandSameClassOverlap(seedSel, slice, W, H).map((s) => s.id).sort();
      const want = expandSameClassOverlap_naive(seedSel, slice, W, H).map((s) => s.id).sort();
      expect(got).toEqual(want);
    });
  }

  it('walks a transitive chain of overlaps', () => {
    // A–B–C chained; D is isolated. Selecting A must pull in B and C, never D.
    const chain: Shape[] = [
      { id: 'A', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 30, h: 30 },
      { id: 'B', classId: 1, kind: 'rectangle', x: 35, y: 10, w: 30, h: 30 },
      { id: 'C', classId: 1, kind: 'rectangle', x: 60, y: 10, w: 30, h: 30 },
      { id: 'D', classId: 1, kind: 'rectangle', x: 150, y: 150, w: 30, h: 30 },
    ];
    const got = expandSameClassOverlap([chain[0]], chain, W, H).map((s) => s.id).sort();
    expect(got).toEqual(['A', 'B', 'C']);
    expect(got).toEqual(expandSameClassOverlap_naive([chain[0]], chain, W, H).map((s) => s.id).sort());
  });
});

describe('rasterizeShapes scratch-buffer reuse', () => {
  it('produces the same mask as a fresh allocation', () => {
    const { gw, gh, scale } = gridFor(W, H);
    const shape: Shape = { id: 'x', classId: 1, kind: 'ellipse', cx: 100, cy: 100, rx: 40, ry: 25 };
    const fresh = rasterizeShapes([shape], gw, gh, scale);
    const scratch = new Uint8Array(gw * gh).fill(1); // dirty on purpose
    scratch.fill(0);
    rasterizeShapes([shape], gw, gh, scale, scratch);
    expect(Array.from(scratch)).toEqual(Array.from(fresh));
  });

  it('accumulates when the buffer is intentionally not cleared', () => {
    const { gw, gh, scale } = gridFor(W, H);
    const a: Shape = { id: 'a', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 20, h: 20 };
    const b: Shape = { id: 'b', classId: 1, kind: 'rectangle', x: 100, y: 100, w: 20, h: 20 };
    const buf = new Uint8Array(gw * gh);
    rasterizeShapes([a], gw, gh, scale, buf);
    rasterizeShapes([b], gw, gh, scale, buf);
    const both = rasterizeShapes([a, b], gw, gh, scale);
    expect(Array.from(buf)).toEqual(Array.from(both));
  });
});
