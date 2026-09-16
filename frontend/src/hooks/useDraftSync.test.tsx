import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { loadDraft, useDraftSync } from './useDraftSync';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useClassStore.setState({ classes: [] });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
  vi.stubGlobal('sendBeacon', undefined);
  Object.defineProperty(window.navigator, 'sendBeacon', { value: vi.fn(), configurable: true });
});

afterEach(() => {
  cleanup(); // unmount while fetch/sendBeacon are still stubbed, before unstubbing them
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('loadDraft', () => {
  it('returns null on 404', async () => {
    (fetch as any).mockResolvedValue({ status: 404, ok: false });
    expect(await loadDraft('local:x.tif')).toBeNull();
  });

  it('returns null on any error', async () => {
    (fetch as any).mockRejectedValue(new Error('down'));
    expect(await loadDraft('local:x.tif')).toBeNull();
  });

  it('returns the parsed draft on success', async () => {
    (fetch as any).mockResolvedValue({ status: 200, ok: true, json: async () => ({ payload: { classes: [] } }) });
    expect(await loadDraft('local:x.tif')).toEqual({ payload: { classes: [] } });
  });
});

describe('useDraftSync', () => {
  it('does nothing when sourceKey is null', () => {
    vi.useFakeTimers();
    renderHook(() => useDraftSync(null));
    vi.advanceTimersByTime(5000);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('autosaves 1.5s after a store change, with the right payload', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    useClassStore.setState({ classes: [{ classId: 1, label: 'Cell', color: '#f00', isVisible: true }] });
    renderHook(() => useDraftSync('local:x.tif'));

    act(() => {
      useAnnotationStore.getState().replaceClassShapesOnSlice('local:x.tif', 0, 1, [
        { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
      ]);
    });
    await vi.advanceTimersByTimeAsync(1500);

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = (fetch as any).mock.calls[0];
    expect(url).toContain('/api/annotations/draft?source_key=');
    expect(init.method).toBe('PUT');
    const body = JSON.parse(init.body);
    expect(body.classes).toEqual([{ classId: 1, label: 'Cell', color: '#f00', isVisible: true }]);
    expect(body.slices['0']).toHaveLength(1);
  });

  it('re-arms the debounce on each subsequent change (no save before the quiet period)', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    renderHook(() => useDraftSync('local:x.tif'));
    act(() => { useAnnotationStore.getState().replaceClassShapesOnSlice('local:x.tif', 0, 1, []); });
    await vi.advanceTimersByTimeAsync(1000);
    act(() => { useAnnotationStore.getState().replaceClassShapesOnSlice('local:x.tif', 1, 1, []); });
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetch).not.toHaveBeenCalled(); // only 1s since the last change, need 1.5s
    await vi.advanceTimersByTimeAsync(500);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('flushes immediately (bypassing the debounce) when the component unmounts', () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useDraftSync('local:x.tif'));
    unmount();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('sends a beacon on beforeunload with the latest payload', () => {
    renderHook(() => useDraftSync('local:x.tif'));
    window.dispatchEvent(new Event('beforeunload'));
    expect(navigator.sendBeacon).toHaveBeenCalledWith(
      expect.stringContaining('/api/annotations/draft?source_key='),
      expect.any(String),
    );
  });
});
