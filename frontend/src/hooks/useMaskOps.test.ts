import { beforeEach, describe, expect, it } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useMaskOps } from './useMaskOps';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';

const SOURCE = 'local:x.tif';

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useDatasetStore.setState({
    meta: { nSlices: 5, height: 64, width: 64, dtype: 'uint8', isRgb: false, valueRange: [0, 255] },
    currentSlice: 0,
  } as any);
});

describe('useMaskOps.applyCleanup', () => {
  it('returns false with no sourceKey', () => {
    const { result } = renderHook(() => useMaskOps(null, 1));
    expect(result.current.applyCleanup('fill')).toBe(false);
  });

  it('returns false with no active class', () => {
    const { result } = renderHook(() => useMaskOps(SOURCE, null));
    expect(result.current.applyCleanup('fill')).toBe(false);
  });

  it('returns false with no dataset meta loaded', () => {
    useDatasetStore.setState({ meta: null } as any);
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    expect(result.current.applyCleanup('fill')).toBe(false);
  });

  it('returns false when the active class has no shapes on the current slice', () => {
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    expect(result.current.applyCleanup('fill')).toBe(false);
  });

  it('re-rasterizes the active class shapes through a fill-holes pass', () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 5, y: 5, w: 20, h: 20 },
    ]);
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    const ok = result.current.applyCleanup('fill');
    expect(ok).toBe(true);
    const shapes = useAnnotationStore.getState().byImage[SOURCE]['0'];
    expect(shapes.length).toBeGreaterThan(0);
    expect(shapes.every((s) => s.classId === 1)).toBe(true);
  });

  it('only touches shapes of the active class, leaving other classes untouched', () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 5, y: 5, w: 20, h: 20 },
    ]);
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 2, [
      { id: 's2', classId: 2, kind: 'rectangle', x: 40, y: 40, w: 10, h: 10 },
    ]);
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    result.current.applyCleanup('grow', 2);
    const slice = useAnnotationStore.getState().byImage[SOURCE]['0'];
    expect(slice.some((s) => s.classId === 2)).toBe(true);
  });

  it.each(['fill', 'islands', 'smooth', 'grow', 'shrink'] as const)('%s op runs without throwing', (op) => {
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 5, y: 5, w: 20, h: 20 },
    ]);
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    expect(() => result.current.applyCleanup(op)).not.toThrow();
  });
});

describe('useMaskOps.copyToNext', () => {
  it('returns false at the last slice', () => {
    useDatasetStore.setState({ currentSlice: 4 } as any);
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 4, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    expect(result.current.copyToNext()).toBe(false);
  });

  it('copies the active class shapes to the next slice', () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    const { result } = renderHook(() => useMaskOps(SOURCE, 1));
    expect(result.current.copyToNext()).toBe(true);
    expect(useAnnotationStore.getState().byImage[SOURCE]['1']).toHaveLength(1);
  });
});
