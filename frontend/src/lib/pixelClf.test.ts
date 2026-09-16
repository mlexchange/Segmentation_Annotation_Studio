import { describe, expect, it } from 'vitest';
import {
  applyProbaThresholdRgba,
  classProbaMaskSetName,
  labelMapToPolygonShapes,
  parseHexColor,
} from './pixelClf';

describe('parseHexColor', () => {
  it('parses 6-digit hex', () => {
    expect(parseHexColor('#ff0000')).toEqual([255, 0, 0]);
  });
});

describe('classProbaMaskSetName', () => {
  it('formats label and percent threshold', () => {
    expect(classProbaMaskSetName(2, 0.65, 'membrane')).toBe('membrane p≥65%');
  });

  it('falls back to class id when label missing', () => {
    expect(classProbaMaskSetName(3, 0.5, null)).toBe('class 3 p≥50%');
  });
});

describe('applyProbaThresholdRgba', () => {
  it('makes pixels below the cut transparent and colors the rest viridis', () => {
    const rgba = new Uint8ClampedArray([
      100, 100, 100, 255,
      255, 255, 255, 255,
    ]);
    applyProbaThresholdRgba(rgba, 0.5); // cut = 0.5
    expect(rgba[3]).toBe(0); // below: transparent
    expect(rgba[7]).toBe(255); // above: opaque
    // p=1 remaps to viridis end (yellowish)
    expect(rgba[4]).toBeGreaterThan(200);
    expect(rgba[5]).toBeGreaterThan(200);
  });
});

describe('labelMapToPolygonShapes', () => {
  it('emits a polygon for a solid class blob', () => {
    const w = 16;
    const h = 16;
    const labels = new Uint8Array(w * h);
    for (let y = 4; y < 12; y++) {
      for (let x = 4; x < 12; x++) labels[y * w + x] = 1;
    }
    const shapes = labelMapToPolygonShapes(labels, w, h, [1], { minRegion: 4, smooth: 0 });
    expect(shapes.length).toBeGreaterThanOrEqual(1);
    expect(shapes[0].classId).toBe(1);
    expect(shapes[0].kind).toBe('polygon');
    expect(shapes[0].points.length).toBeGreaterThanOrEqual(6);
  });

  it('skips pixels covered by preserveShapes', () => {
    const w = 16;
    const h = 16;
    const labels = new Uint8Array(w * h).fill(1);
    const preserve = [
      {
        id: 'a',
        classId: 1,
        kind: 'rectangle' as const,
        x: 0,
        y: 0,
        w: 16,
        h: 16,
      },
    ];
    const shapes = labelMapToPolygonShapes(labels, w, h, [1], {
      minRegion: 4,
      preserveShapes: preserve,
      smooth: 0,
    });
    expect(shapes).toHaveLength(0);
  });

  it('carves holes so nested class regions do not fill through each other', () => {
    const w = 32;
    const h = 32;
    const labels = new Uint8Array(w * h);
    // Outer class 1 border; inner 10×10 is class 2
    for (let y = 2; y < 30; y++) {
      for (let x = 2; x < 30; x++) labels[y * w + x] = 1;
    }
    for (let y = 11; y < 21; y++) {
      for (let x = 11; x < 21; x++) labels[y * w + x] = 2;
    }
    const shapes = labelMapToPolygonShapes(labels, w, h, [1, 2], {
      minRegion: 4,
      smooth: 0,
    });
    const c1 = shapes.filter((s) => s.classId === 1);
    const c2 = shapes.filter((s) => s.classId === 2);
    expect(c1.length).toBeGreaterThanOrEqual(1);
    expect(c2.length).toBeGreaterThanOrEqual(1);
    // Class 1 must have a hole for the class 2 island (pixel-accurate commit).
    expect(c1.some((s) => (s.holes?.length ?? 0) > 0)).toBe(true);
  });
});
