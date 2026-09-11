/**
 * The slice cache hands out blob object URLs, which the browser keeps alive until
 * explicitly revoked — evicting the query is not enough. This is exactly the kind
 * of leak that never shows up in a feature test (everything still works, the tab
 * just grows), so it gets its own.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import { installImageSliceGc } from './useImageSlice';

let revoked: string[] = [];

beforeEach(() => {
  revoked = [];
  vi.stubGlobal('URL', {
    ...URL,
    createObjectURL: (b: Blob) => `blob:mock/${(b as unknown as { id?: string }).id ?? 'x'}`,
    revokeObjectURL: (u: string) => { revoked.push(u); },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Seed the cache with a slice query already holding a blob URL. */
function seedSlice(client: QueryClient, sliceIndex: number, url: string) {
  const key = ['imageSlice', 'src', 'tiled', sliceIndex, {}, null];
  client.setQueryData(key, url);
  return key;
}

describe('installImageSliceGc', () => {
  it('revokes a slice URL when its query is removed from the cache', () => {
    const client = new QueryClient();
    const stop = installImageSliceGc(client);
    const key = seedSlice(client, 0, 'blob:mock/slice0');

    client.removeQueries({ queryKey: key, exact: true });

    expect(revoked).toEqual(['blob:mock/slice0']);
    stop();
  });

  it('revokes the previous URL when a slice is refetched into the same key', () => {
    const client = new QueryClient();
    const stop = installImageSliceGc(client);
    const key = seedSlice(client, 3, 'blob:mock/old');

    client.setQueryData(key, 'blob:mock/new');

    expect(revoked).toEqual(['blob:mock/old']);
    stop();
  });

  it('revokes every visited slice as the cache is cleared', () => {
    const client = new QueryClient();
    const stop = installImageSliceGc(client);
    for (let i = 0; i < 5; i++) seedSlice(client, i, `blob:mock/s${i}`);

    client.clear();

    expect(revoked.sort()).toEqual(
      ['blob:mock/s0', 'blob:mock/s1', 'blob:mock/s2', 'blob:mock/s3', 'blob:mock/s4'],
    );
    stop();
  });

  it('ignores non-slice queries and non-blob data', () => {
    const client = new QueryClient();
    const stop = installImageSliceGc(client);
    client.setQueryData(['somethingElse', 1], 'blob:mock/not-a-slice');
    client.setQueryData(['imageSlice', 'src', 'tiled', 9, {}, null], { notAString: true });

    client.clear();

    expect(revoked).toEqual([]);
    stop();
  });

  it('stops revoking once unsubscribed', () => {
    const client = new QueryClient();
    const stop = installImageSliceGc(client);
    stop();
    seedSlice(client, 1, 'blob:mock/after-stop');
    client.clear();
    expect(revoked).toEqual([]);
  });
});
