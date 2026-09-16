import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { useSave } from './useSave';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

const SOURCE = 'local:x.tif';

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useClassStore.setState({ classes: [] });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('useSave', () => {
  it('starts clean (not dirty)', async () => {
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    expect(result.current.isDirty).toBe(false);
  });

  it('becomes dirty after a store change', async () => {
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    act(() => {
      useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
        { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
      ]);
    });
    expect(result.current.isDirty).toBe(true);
  });

  it('resets dirty/versions/lastSavedAt when sourceKey changes', async () => {
    const { result, rerender } = renderHook(({ src }) => useSave(src), { initialProps: { src: SOURCE } });
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    act(() => {
      useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, []);
    });
    expect(result.current.isDirty).toBe(true);

    act(() => { rerender({ src: 'local:other.tif' }); });
    expect(result.current.isDirty).toBe(false);
  });

  it('buildSavePayload returns null with no sourceKey', () => {
    const { result } = renderHook(() => useSave(null));
    expect(result.current.buildSavePayload()).toBeNull();
    expect(result.current.saveSummary).toEqual({ shapeCount: 0, classCount: 0 });
  });

  it('saveSummary reflects real shape/class counts for the source', async () => {
    act(() => {
      useClassStore.setState({ classes: [{ classId: 1, label: 'Cell', color: '#f00', isVisible: true }] });
      useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
        { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
        { id: 's2', classId: 1, kind: 'rectangle', x: 5, y: 5, w: 2, h: 2 },
      ]);
    });
    const { result } = renderHook(() => useSave(SOURCE));
    expect(result.current.saveSummary).toEqual({ shapeCount: 2, classCount: 1 });
  });

  it('refreshVersions populates the versions list', async () => {
    (fetch as any).mockResolvedValue({ ok: true, json: async () => [{ version: 1, saved_at: 't', shape_count: 0, class_count: 0 }] });
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(result.current.versions).toHaveLength(1));
  });

  it('save() POSTs the payload, clears dirty, and sets lastSavedAt', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => [] }) // initial refreshVersions on mount
      .mockResolvedValueOnce({ ok: true, json: async () => ({ saved_at: '2024-01-01T00:00:00Z' }) }) // save
      .mockResolvedValueOnce({ ok: true, json: async () => [{ version: 1, saved_at: '2024-01-01T00:00:00Z', shape_count: 0, class_count: 0 }] }); // post-save refresh

    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    let ok = false;
    await act(async () => {
      ok = await result.current.save({ annotatedBy: 'Alice', notes: 'n' });
    });
    expect(ok).toBe(true);
    expect(result.current.lastSavedAt).toBe('2024-01-01T00:00:00Z');
    expect(result.current.isDirty).toBe(false);

    const [url, init] = (fetch as any).mock.calls[1];
    expect(url).toContain('/api/annotations/save?source_key=');
    const body = JSON.parse(init.body);
    expect(body.annotated_by).toBe('Alice');
    expect(body.notes).toBe('n');
  });

  it('save() returns false and does not throw on a failed request', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => [] })
      .mockResolvedValueOnce({ ok: false, status: 500 });
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    let ok = true;
    await act(async () => {
      ok = await result.current.save();
    });
    expect(ok).toBe(false);
  });

  it('fetchVersionPayload caches results by version number', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => [] })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ payload: { classes: [], slices: {}, split_by_slice: {}, negative_slices: [] } }) });
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    let payload1;
    await act(async () => { payload1 = await result.current.fetchVersionPayload(1); });
    expect(payload1).toEqual({ classes: [], slices: {}, split_by_slice: {}, negative_slices: [] });

    const callsBefore = (fetch as any).mock.calls.length;
    await act(async () => { await result.current.fetchVersionPayload(1); });
    expect((fetch as any).mock.calls.length).toBe(callsBefore); // cached, no new fetch
  });

  it('restoreVersion loads a version into the stores and marks dirty', async () => {
    (fetch as any)
      .mockResolvedValueOnce({ ok: true, json: async () => [] })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          payload: {
            classes: [{ classId: 1, label: 'Restored', color: '#000', isVisible: true }],
            slices: { '0': [{ id: 'r1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1 }] },
            split_by_slice: {},
            negative_slices: [],
          },
        }),
      });
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    await act(async () => { await result.current.restoreVersion(1); });

    expect(useClassStore.getState().classes[0].label).toBe('Restored');
    expect(useAnnotationStore.getState().byImage[SOURCE]?.['0']).toHaveLength(1);
    expect(result.current.isDirty).toBe(true);
  });

  it('markClean clears the dirty flag', async () => {
    const { result } = renderHook(() => useSave(SOURCE));
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    act(() => {
      useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, []);
    });
    act(() => result.current.markClean());
    expect(result.current.isDirty).toBe(false);
  });
});
