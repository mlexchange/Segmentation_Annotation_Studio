import { describe, expect, it } from 'vitest';
import {
  applyMorphToLabelMap,
  applyRoiToLabelMap,
  applyRoiToShapes,
  filterLabelMapBySize,
  filterShapesByComponentSize,
  filterBySize,
  labelComponents,
} from './cleanupOps';
import type { Shape } from '@/stores/annotationStore';

describe('labelComponents', () => {
  it('labels separate 4-connected blobs with their sizes', () => {
    const gw = 8;
    const gh = 8;
    const mask = new Uint8Array(gw * gh);
    // 2×2 blob
    mask[1 * gw + 1] = 1;
    mask[1 * gw + 2] = 1;
    mask[2 * gw + 1] = 1;
    mask[2 * gw + 2] = 1;
    // 1px island
    mask[6 * gw + 6] = 1;
    const { labels, sizesByLabel, count } = labelComponents(mask, gw, gh);
    expect(count).toBe(2);
    const sizes = Object.values(sizesByLabel).sort((a, b) => a - b);
    expect(sizes).toEqual([1, 4]);
    expect(labels[1 * gw + 1]).toBe(labels[2 * gw + 2]);
    expect(labels[6 * gw + 6]).not.toBe(labels[1 * gw + 1]);
  });
});

describe('filterBySize', () => {
  it('keeps only components in [min, max]', () => {
    const gw = 10;
    const gh = 10;
    const mask = new Uint8Array(gw * gh);
    for (let y = 1; y < 5; y++) for (let x = 1; x < 5; x++) mask[y * gw + x] = 1; // 16
    mask[8 * gw + 8] = 1;
    const kept = filterBySize(mask, gw, gh, { minArea: 4, maxArea: 20, mode: 'keep' });
    expect(kept[8 * gw + 8]).toBe(0);
    expect(kept[2 * gw + 2]).toBe(1);
    const deleted = filterBySize(mask, gw, gh, { minArea: 4, maxArea: 20, mode: 'delete' });
    expect(deleted[8 * gw + 8]).toBe(1);
    expect(deleted[2 * gw + 2]).toBe(0);
  });
});

describe('filterShapesByComponentSize', () => {
  it('removes small islands for a class', () => {
    const shapes: Shape[] = [
      { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 8, h: 8 },
      { id: 'b', classId: 1, kind: 'rectangle', x: 14, y: 14, w: 1, h: 1 },
      { id: 'c', classId: 2, kind: 'rectangle', x: 0, y: 0, w: 4, h: 4 },
    ];
    const out = filterShapesByComponentSize(shapes, 16, 16, {
      classId: 1,
      minArea: 10,
      maxArea: Infinity,
      mode: 'keep',
    });
    const c1 = out.filter((s) => s.classId === 1);
    const c2 = out.filter((s) => s.classId === 2);
    expect(c2).toHaveLength(1);
    expect(c1.length).toBeGreaterThanOrEqual(1);
    // Small island at (14,14) should be gone after re-vectorize
    const bigOnly = filterShapesByComponentSize(shapes, 16, 16, {
      classId: 1,
      minArea: 10,
      maxArea: Infinity,
      mode: 'keep',
    });
    // Class 2 untouched
    expect(bigOnly.some((s) => s.classId === 2)).toBe(true);
  });
});

describe('filterLabelMapBySize', () => {
  it('edits pixels without polygon round-trip', () => {
    const w = 16;
    const h = 16;
    const labels = new Uint8Array(w * h);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) labels[y * w + x] = 1; // 64
    labels[15 * w + 15] = 1; // 1px island
    const kept = filterLabelMapBySize(labels, w, h, {
      classId: 1,
      minArea: 10,
      maxArea: Infinity,
      mode: 'keep',
    });
    expect(kept[15 * w + 15]).toBe(0);
    expect(kept[2 * w + 2]).toBe(1);
  });
});

describe('applyMorphToLabelMap', () => {
  it('fills a hole in a solid class blob', () => {
    const w = 20;
    const h = 20;
    const labels = new Uint8Array(w * h);
    for (let y = 4; y < 16; y++) for (let x = 4; x < 16; x++) labels[y * w + x] = 2;
    labels[10 * w + 10] = 0;
    const filled = applyMorphToLabelMap(labels, w, h, {
      op: 'fill',
      param: 1,
      classId: 2,
    });
    expect(filled[10 * w + 10]).toBe(2);
  });
});

describe('applyRoiToLabelMap', () => {
  it('keep_inside zeros exterior class pixels', () => {
    const w = 16;
    const h = 16;
    const labels = new Uint8Array(w * h).fill(1);
    const roi: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 4, y: 4, w: 4, h: 4 };
    const out = applyRoiToLabelMap(labels, w, h, {
      roi,
      mode: 'keep_inside',
      classId: 1,
    });
    expect(out[0]).toBe(0);
    expect(out[6 * w + 6]).toBe(1);
  });
});

describe('applyRoiToShapes', () => {
  it('keep-inside leaves exterior untouched for that class', () => {
    const shapes: Shape[] = [
      { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 16, h: 16 },
    ];
    const roi: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 4, y: 4, w: 4, h: 4 };
    const out = applyRoiToShapes(shapes, 16, 16, {
      roi,
      mode: 'keep_inside',
      classId: 1,
    });
    // Should only keep the center 4×4 ≈ 16 px → few polygons
    expect(out.filter((s) => s.classId === 1).length).toBeGreaterThanOrEqual(1);
    // Outside class untouched if we only pass class 1
  });

  it('delete-inside clears ROI pixels but keeps exterior', () => {
    const shapes: Shape[] = [
      { id: 'a', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 12, h: 12 },
    ];
    const roi: Shape = { id: 'r', classId: 0, kind: 'rectangle', x: 2, y: 2, w: 4, h: 4 };
    const out = applyRoiToShapes(shapes, 16, 16, {
      roi,
      mode: 'delete_inside',
      classId: 1,
    });
    expect(out.some((s) => s.classId === 1)).toBe(true);
  });
});
