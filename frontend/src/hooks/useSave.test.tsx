import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useSave } from './useSave';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

function response(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

describe('useSave dirty tracking', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response(200, [])));
    useAnnotationStore.getState().reset();
    useAnnotationStore.temporal.getState().clear();
    useClassStore.getState().setClasses([]);
  });

  it('marks the very first edit after mounting dirty', async () => {
    const { result } = renderHook(() => useSave('sample'));
    expect(result.current.isDirty).toBe(false);

    act(() => useAnnotationStore.getState().setSplitForSlice('sample', 0, 'train'));
    await waitFor(() => expect(result.current.isDirty).toBe(true));
  });

  it('becomes clean only for the exact snapshot that successfully saved', async () => {
    const saveResponse = deferred<Response>();
    vi.mocked(fetch).mockImplementation((input, init) => {
      const url = String(input);
      if (url.includes('/save?') && init?.method === 'POST') return saveResponse.promise;
      return Promise.resolve(response(200, []));
    });

    const { result } = renderHook(() => useSave('sample'));
    act(() => useAnnotationStore.getState().toggleNegativeSlice('sample', 1));
    await waitFor(() => expect(result.current.isDirty).toBe(true));

    let saving!: Promise<boolean>;
    act(() => { saving = result.current.save(); });
    act(() => useAnnotationStore.getState().toggleNegativeSlice('sample', 2));

    await act(async () => {
      saveResponse.resolve(response(200, { saved_at: '2026-08-03T12:00:00Z' }));
      await saving;
    });
    expect(result.current.isDirty).toBe(true);
  });

  it('markClean fingerprints the current state rather than suppressing the next edit', async () => {
    const { result } = renderHook(() => useSave('sample'));
    act(() => useAnnotationStore.getState().toggleNegativeSlice('sample', 1));
    await waitFor(() => expect(result.current.isDirty).toBe(true));
    act(() => result.current.markClean());
    expect(result.current.isDirty).toBe(false);

    act(() => useAnnotationStore.getState().toggleNegativeSlice('sample', 2));
    await waitFor(() => expect(result.current.isDirty).toBe(true));
  });
});
