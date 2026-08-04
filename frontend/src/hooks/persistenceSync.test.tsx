import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { loadDraft, putDraft, useDraftSync } from './useDraftSync';
import { loadGuide, useGuideSync } from './useGuideSync';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { useReferenceGuideStore } from '@/stores/referenceGuideStore';

function response(status: number, body: unknown = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function putCalls() {
  return vi.mocked(fetch).mock.calls.filter(([, init]) => init?.method === 'PUT');
}

describe('draft persistence', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn());
    useAnnotationStore.getState().reset();
    useAnnotationStore.temporal.getState().clear();
    useClassStore.getState().setClasses([]);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('distinguishes a confirmed missing draft from a load error', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(response(404))
      .mockResolvedValueOnce(response(503));

    await expect(loadDraft('missing')).resolves.toEqual({ status: 'not-found' });
    await expect(loadDraft('failed')).resolves.toMatchObject({
      status: 'error',
      error: expect.any(Error),
    });
  });

  it('checks PUT status instead of reporting an HTTP failure as saved', async () => {
    vi.mocked(fetch).mockResolvedValue(response(500));
    await expect(putDraft('sample', { slices: {} })).rejects.toThrow(/500/);
  });

  it('never autosaves after a failed load, even if in-memory stores change', async () => {
    vi.mocked(fetch).mockResolvedValue(response(503));
    await loadDraft('load-error');
    vi.mocked(fetch).mockClear();

    const { unmount } = renderHook(() => useDraftSync('load-error'));
    act(() => {
      useAnnotationStore.getState().setSplitForSlice('load-error', 0, 'train');
      vi.advanceTimersByTime(2_000);
    });
    expect(putCalls()).toHaveLength(0);
    unmount();
    expect(putCalls()).toHaveLength(0);
  });

  it('does not save a confirmed empty draft until the first real edit', async () => {
    vi.mocked(fetch).mockResolvedValue(response(404));
    await loadDraft('new-sample');
    vi.mocked(fetch).mockClear();
    vi.mocked(fetch).mockResolvedValue(response(204));

    const { unmount } = renderHook(() => useDraftSync('new-sample'));
    act(() => vi.advanceTimersByTime(2_000));
    expect(putCalls()).toHaveLength(0);

    act(() => useAnnotationStore.getState().toggleNegativeSlice('new-sample', 3));
    act(() => vi.advanceTimersByTime(1_500));
    await act(async () => Promise.resolve());
    expect(putCalls()).toHaveLength(1);
    unmount();
  });

  it('flushes dirty data on unload with a compatible keepalive PUT', async () => {
    vi.mocked(fetch).mockResolvedValue(response(404));
    await loadDraft('unload-sample');
    vi.mocked(fetch).mockClear();
    vi.mocked(fetch).mockResolvedValue(response(204));

    const { unmount } = renderHook(() => useDraftSync('unload-sample'));
    act(() => useAnnotationStore.getState().toggleNegativeSlice('unload-sample', 1));
    act(() => window.dispatchEvent(new Event('beforeunload')));

    expect(putCalls()).toHaveLength(1);
    expect(putCalls()[0][1]).toEqual(expect.objectContaining({ method: 'PUT', keepalive: true }));
    unmount();
  });

  // Regression: TrainPage's "Import as annotations" writes shapes into the
  // store for the CURRENTLY OPEN sample, but only AnnotatePage used to mount
  // useDraftSync. Returning to Annotate afterward established a "clean"
  // baseline that already included the import (the fingerprint was captured
  // AFTER the import landed), so autosave never fired and the import was lost
  // on reload or clobbered by re-opening the sample from Browse. Fixed by
  // mounting useDraftSync in TrainPage too — these tests simulate exactly
  // that cross-page sequence without needing either page component.
  it('autosaves an import made from a second mount of the hook (e.g. Train after Annotate)', async () => {
    vi.mocked(fetch).mockResolvedValue(response(404));
    await loadDraft('cross-page-sample');
    vi.mocked(fetch).mockClear();
    vi.mocked(fetch).mockResolvedValue(response(204));

    // First mount (e.g. AnnotatePage): nothing happens here, just establishes
    // and then abandons a hook instance, like navigating away unedited.
    const first = renderHook(() => useDraftSync('cross-page-sample'));
    first.unmount();
    expect(putCalls()).toHaveLength(0);

    // Second mount of the SAME sourceKey (e.g. TrainPage, after "Import as
    // annotations"). The draft-load revision is module-level state and is
    // still ready from the loadDraft() call above, so this mount's baseline
    // is captured immediately and the import below is correctly seen as dirty.
    const second = renderHook(() => useDraftSync('cross-page-sample'));
    act(() => {
      useAnnotationStore.getState().addShapes('cross-page-sample', 0, [
        { id: 'pred_1', classId: 1, kind: 'polygon', points: [0, 0, 10, 0, 10, 10] },
      ]);
    });
    // Separate act() calls: the debounce-scheduling effect only runs once React
    // commits the store update above, so advancing fake timers in the SAME act()
    // callback (before that effect has flushed) advances a clock with no timer
    // pending yet — the effect then schedules a fresh 1500ms timer AFTER the
    // advance, which never fires within this test. Splitting them, matching
    // every other test in this file, gives the effect a commit point first.
    act(() => vi.advanceTimersByTime(1_500));
    await act(async () => Promise.resolve());

    expect(putCalls()).toHaveLength(1);
    const body = JSON.parse(putCalls()[0][1]!.body as string);
    expect(body.slices['0']).toEqual([expect.objectContaining({ id: 'pred_1', classId: 1 })]);
    second.unmount();
  });

  it('flushes an import on unmount even before the debounce fires (leaving the tab quickly)', async () => {
    vi.mocked(fetch).mockResolvedValue(response(404));
    await loadDraft('quick-nav-sample');
    vi.mocked(fetch).mockClear();
    vi.mocked(fetch).mockResolvedValue(response(204));

    const { unmount } = renderHook(() => useDraftSync('quick-nav-sample'));
    act(() => {
      useAnnotationStore.getState().addShapes('quick-nav-sample', 0, [
        { id: 'pred_1', classId: 1, kind: 'polygon', points: [0, 0, 10, 0, 10, 10] },
      ]);
    });
    // No advanceTimersByTime: unmount happens before the 1500ms debounce
    // would have fired on its own — the cleanup-flush effect must still save.
    unmount();

    expect(putCalls()).toHaveLength(1);
  });
});

describe('guide persistence', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn());
    useReferenceGuideStore.getState().clear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('distinguishes not-found from server and network errors', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(response(404))
      .mockResolvedValueOnce(response(500))
      .mockRejectedValueOnce(new Error('offline'));

    await expect(loadGuide('missing-guide')).resolves.toEqual({ status: 'not-found' });
    await expect(loadGuide('bad-guide')).resolves.toMatchObject({ status: 'error' });
    await expect(loadGuide('offline-guide')).resolves.toMatchObject({ status: 'error' });
  });

  it('does not replace and autosave a guide when loading it fails', async () => {
    vi.mocked(fetch).mockResolvedValue(response(503));
    const { unmount } = renderHook(() => useGuideSync('guide-error'));
    await act(async () => { await Promise.resolve(); });
    expect(useReferenceGuideStore.getState().loadStatus).toBe('error');
    act(() => vi.advanceTimersByTime(2_000));
    expect(putCalls()).toHaveLength(0);
    unmount();
  });

  it('treats a 404 as a clean empty baseline and saves only after an edit', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(response(404)).mockResolvedValue(response(204));
    const { unmount } = renderHook(() => useGuideSync('new-guide'));
    await act(async () => { await Promise.resolve(); });
    expect(useReferenceGuideStore.getState().loadStatus).toBe('ready');
    act(() => vi.advanceTimersByTime(2_000));
    expect(putCalls()).toHaveLength(0);

    act(() => useReferenceGuideStore.getState().setNotes('real edit'));
    act(() => vi.advanceTimersByTime(1_000));
    await act(async () => Promise.resolve());
    expect(putCalls()).toHaveLength(1);
    unmount();
  });
});
