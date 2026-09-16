import { describe, it, expect } from 'vitest';
import { buildLiveMaskVolume } from './volumeMaskPreview';
import type { Shape } from '@/stores/annotationStore';

const rect = (classId: number, x: number, y: number, w: number, h: number): Shape => ({
  id: `${classId}-${x}-${y}`,
  kind: 'rectangle',
  classId,
  x, y, w, h,
});

describe('buildLiveMaskVolume', () => {
  it('returns null when there are no shapes anywhere', () => {
    expect(buildLiveMaskVolume({}, 64, 64, 4)).toBeNull();
    expect(buildLiveMaskVolume({ '0': [] }, 64, 64, 4)).toBeNull();
  });

  it('returns null when nSlices is not positive', () => {
    expect(buildLiveMaskVolume({ '0': [rect(1, 0, 0, 4, 4)] }, 64, 64, 0)).toBeNull();
  });

  it('produces dims in [width, height, depth] order matching the real dataset shape', () => {
    const volume = buildLiveMaskVolume({ '0': [rect(1, 0, 0, 8, 8)] }, 64, 32, 10, 1000);
    expect(volume).not.toBeNull();
    expect(volume!.dims).toEqual([64, 32, 10]);
    expect(volume!.data.length).toBe(64 * 32 * 10);
  });

  it('leaves unannotated slices as all-background (0)', () => {
    const volume = buildLiveMaskVolume({ '2': [rect(1, 0, 0, 8, 8)] }, 16, 16, 4, 1000);
    const [w, h] = volume!.dims;
    const sliceBytes = w * h;
    const slice0 = volume!.data.subarray(0, sliceBytes);
    expect(slice0.every((v) => v === 0)).toBe(true);
  });

  it('paints the annotated slice with the shape\'s class id', () => {
    const volume = buildLiveMaskVolume({ '1': [rect(3, 0, 0, 8, 8)] }, 16, 16, 4, 1000);
    const [w, h] = volume!.dims;
    const sliceBytes = w * h;
    const slice1 = volume!.data.subarray(sliceBytes, sliceBytes * 2);
    expect(slice1[0]).toBe(3);
    expect(Math.max(...slice1)).toBe(3);
  });

  it('resolves overlapping different-class shapes by ascending class id (higher wins)', () => {
    const shapes = [rect(1, 0, 0, 16, 16), rect(5, 0, 0, 16, 16)];
    const volume = buildLiveMaskVolume({ '0': shapes }, 16, 16, 1, 1000);
    expect(volume!.data[0]).toBe(5);
  });

  it('downsamples to at most maxDim on the longest in-plane edge', () => {
    const volume = buildLiveMaskVolume({ '0': [rect(1, 0, 0, 100, 100)] }, 2048, 1024, 2, 256);
    const [w, h] = volume!.dims;
    expect(Math.max(w, h)).toBeLessThanOrEqual(256);
  });
});
