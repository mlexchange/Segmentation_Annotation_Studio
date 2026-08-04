import { describe, it, expect } from 'vitest';
import { buildLabelVolume } from './labelVolume';
import type { VolumeDims } from './volumeDims';
import type { Shape } from '@/stores/annotationStore';

describe('buildLabelVolume', () => {
  it('returns an all-zero volume of the right length when slicesForSource is undefined', () => {
    const dims: VolumeDims = { nz: 2, ny: 4, nx: 5, sxy: 1, sz: 1 };
    const vol = buildLabelVolume(undefined, [1, 2], dims);
    expect(vol.length).toBe(2 * 4 * 5);
    expect(vol.every((v) => v === 0)).toBe(true);
  });

  it('returns an all-zero volume when slicesForSource is an empty object', () => {
    const dims: VolumeDims = { nz: 1, ny: 3, nx: 3, sxy: 1, sz: 1 };
    const vol = buildLabelVolume({}, [1], dims);
    expect(vol.length).toBe(9);
    expect(vol.every((v) => v === 0)).toBe(true);
  });

  it('rasterizes a single polygon to its classOrder index + 1', () => {
    const dims: VolumeDims = { nz: 1, ny: 8, nx: 8, sxy: 1, sz: 1 };
    const shape: Shape = { id: 'a', classId: 1, kind: 'polygon', points: [1, 1, 5, 1, 5, 5, 1, 5] };
    const vol = buildLabelVolume({ '0': [shape] }, [1, 2], dims);
    // Interior of the square is class 1 -> classOrder index 0 -> value 1.
    expect(vol[3 * 8 + 3]).toBe(1);
    // Outside the square is background.
    expect(vol[7 * 8 + 7]).toBe(0);
  });

  it('gives two different non-overlapping classes their own class-index values', () => {
    const dims: VolumeDims = { nz: 1, ny: 8, nx: 8, sxy: 1, sz: 1 };
    const classShape: Shape = { id: 'c1', classId: 1, kind: 'polygon', points: [0, 0, 3, 0, 3, 3, 0, 3] };
    const otherClassShape: Shape = { id: 'c2', classId: 2, kind: 'polygon', points: [5, 5, 8, 5, 8, 8, 5, 8] };
    const vol = buildLabelVolume({ '0': [classShape, otherClassShape] }, [1, 2], dims);
    expect(vol[1 * 8 + 1]).toBe(1); // inside classShape (classId 1 -> index 0 -> value 1)
    expect(vol[6 * 8 + 6]).toBe(2); // inside otherClassShape (classId 2 -> index 1 -> value 2)
  });

  it('lets the later class in classOrder win in an overlap region on the same slice', () => {
    const dims: VolumeDims = { nz: 1, ny: 8, nx: 8, sxy: 1, sz: 1 };
    // classOrder = [10, 20]: classId 20 is later, so it should win the overlap.
    const first: Shape = { id: 'f', classId: 10, kind: 'polygon', points: [0, 0, 5, 0, 5, 5, 0, 5] };
    const second: Shape = { id: 's', classId: 20, kind: 'polygon', points: [3, 3, 7, 3, 7, 7, 3, 7] };
    const vol = buildLabelVolume({ '0': [first, second] }, [10, 20], dims);
    expect(vol[1 * 8 + 1]).toBe(1); // classId 10-only region
    expect(vol[6 * 8 + 6]).toBe(2); // classId 20-only region
    expect(vol[4 * 8 + 4]).toBe(2); // overlap region: later class (20) wins
  });

  it('carves an erase stroke out of the interior, leaving the rest of the shape filled', () => {
    const dims: VolumeDims = { nz: 1, ny: 8, nx: 8, sxy: 1, sz: 1 };
    const shape: Shape = {
      id: 'd',
      classId: 5,
      kind: 'polygon',
      points: [0, 0, 8, 0, 8, 8, 0, 8],
      erased: [{ points: [4, 4, 4, 4], radius: 1 }],
    };
    const vol = buildLabelVolume({ '0': [shape] }, [5], dims);
    expect(vol[4 * 8 + 4]).toBe(0); // erased center
    expect(vol[0 * 8 + 0]).toBe(1); // corner still filled (classOrder index 0 -> value 1)
  });

  it('lets an adjacent same-class shape fill show through a hole (rasterizeUnion, not rasterizeShapes)', () => {
    const dims: VolumeDims = { nz: 1, ny: 8, nx: 8, sxy: 1, sz: 1 };
    // Outer frame with a hole carved out of its own fill.
    const outer: Shape = {
      id: 'outer',
      classId: 7,
      kind: 'polygon',
      points: [0, 0, 8, 0, 8, 8, 0, 8],
      holes: [[2, 2, 6, 2, 6, 6, 2, 6]],
    };
    // A second, same-class shape that exactly covers the hole region. Because
    // labelVolume.ts must call rasterizeUnion (which rasterizes each shape
    // independently and ORs the results) rather than rasterizeShapes (which
    // fills shapes onto one mask in sequence, letting a later shape's hole
    // carve through an earlier shape's fill), this adjacent fill must show
    // through even though it exactly overlaps `outer`'s hole.
    const patch: Shape = { id: 'patch', classId: 7, kind: 'rectangle', x: 2, y: 2, w: 4, h: 4 };
    const vol = buildLabelVolume({ '0': [outer, patch] }, [7], dims);
    expect(vol[4 * 8 + 4]).toBe(1); // hole region, but filled by the adjacent patch
    expect(vol[1 * 8 + 1]).toBe(1); // outer frame, outside the hole
  });

  it('maps multiple source slices to the same z (nearest-sample) with later-source-slice-wins', () => {
    // sz=2, nz=3: round(0/2)=0, round(3/2)=2, round(4/2)=2 — slices "3" and "4"
    // land on the same z=2. classOrder is [100, 200] so classId 100 has a
    // LOWER classOrder index than 200; if classId 100 still wins at the
    // shared z, that proves it's ascending-source-slice-processing-order
    // (not classOrder position) that decides the winner across slices.
    const dims: VolumeDims = { nz: 3, ny: 4, nx: 4, sxy: 1, sz: 2 };
    const classOrder = [100, 200];
    const atOrigin: Shape = { id: 's0', classId: 100, kind: 'rectangle', x: 0, y: 0, w: 0, h: 0 };
    const slice3: Shape = { id: 's3', classId: 200, kind: 'rectangle', x: 2, y: 2, w: 0, h: 0 };
    const slice4: Shape = { id: 's4', classId: 100, kind: 'rectangle', x: 2, y: 2, w: 0, h: 0 };
    const vol = buildLabelVolume({ '0': [atOrigin], '3': [slice3], '4': [slice4] }, classOrder, dims);
    const planeSize = dims.ny * dims.nx;
    expect(vol[0 * planeSize + (0 * dims.nx + 0)]).toBe(1); // z=0, from source slice "0"
    // z=2 pixel (2,2): slice "3" wrote 2 first, slice "4" (numerically later)
    // overwrote it with 1 — later source slice wins, not higher classOrder index.
    expect(vol[2 * planeSize + (2 * dims.nx + 2)]).toBe(1);
    // z=2 pixel (0,0) was never painted — distinct from z=0's painted origin.
    expect(vol[2 * planeSize + (0 * dims.nx + 0)]).toBe(0);
  });

  it('silently skips shapes whose classId is absent from classOrder', () => {
    // A class can be deleted from classStore while its shapes still exist in
    // an old draft; such shapes have no position in classOrder to render at,
    // so they are skipped rather than throwing.
    const dims: VolumeDims = { nz: 1, ny: 4, nx: 4, sxy: 1, sz: 1 };
    const known: Shape = { id: 'k', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 0, h: 0 };
    const unknown: Shape = { id: 'u', classId: 99, kind: 'rectangle', x: 2, y: 2, w: 0, h: 0 };
    const vol = buildLabelVolume({ '0': [known, unknown] }, [1], dims);
    expect(vol[0 * 4 + 0]).toBe(1); // known class rendered
    expect(vol[2 * 4 + 2]).toBe(0); // unknown class silently skipped
  });

  it('clamps classOrder positions at/after 254 to voxel value 255', () => {
    const dims: VolumeDims = { nz: 1, ny: 4, nx: 4, sxy: 1, sz: 1 };
    const classOrder = Array.from({ length: 300 }, (_, i) => i); // classOrder[260] === 260
    const shape: Shape = { id: 'far', classId: 260, kind: 'rectangle', x: 0, y: 0, w: 3, h: 3 };
    const vol = buildLabelVolume({ '0': [shape] }, classOrder, dims);
    expect(vol.every((v) => v === 255)).toBe(true);
  });
});
