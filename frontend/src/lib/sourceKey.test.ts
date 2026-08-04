import { describe, it, expect } from 'vitest';
import { buildSourceKey, isAnnotatedPath, parseSourceKey } from './sourceKey';

const URI = 'http://127.0.0.1:8010';
const key = (path: string) => buildSourceKey('tiled', path, URI);

describe('isAnnotatedPath', () => {
  it('matches the sample own key', () => {
    const keys = new Set([key('browse/ds')]);
    expect(isAnnotatedPath(keys, 'browse/ds', URI)).toBe(true);
  });

  it('matches a volume whose individual array was annotated', () => {
    // "Annotate first image" keys the array, not the volume.
    const keys = new Set([key('browse/ds/img_00003')]);
    expect(isAnnotatedPath(keys, 'browse/ds', URI)).toBe(true);
  });

  it('does not badge a slice because a sibling slice was annotated', () => {
    const keys = new Set([key('browse/ds')]);
    expect(isAnnotatedPath(keys, 'browse/ds/img_00003', URI)).toBe(false);
  });

  it('does not match a different dataset with a shared name prefix', () => {
    const keys = new Set([key('browse/ds_2/img_1')]);
    expect(isAnnotatedPath(keys, 'browse/ds', URI)).toBe(false);
  });

  it('does not match the same path on another server', () => {
    const keys = new Set([buildSourceKey('tiled', 'browse/ds', 'http://other:8010')]);
    expect(isAnnotatedPath(keys, 'browse/ds', URI)).toBe(false);
  });

  it('is false for an empty key set', () => {
    expect(isAnnotatedPath(new Set(), 'browse/ds', URI)).toBe(false);
  });
});

describe('source-key round trips', () => {
  it.each([
    ['http://example.test', 'browse/sample'],
    ['https://example.test:8443', 'folder/δ sample:01'],
    ['http://[2001:db8::1]:8010', 'unicode/雪'],
  ])('round-trips tiled server %s', (serverUri, path) => {
    expect(parseSourceKey(buildSourceKey('tiled', path, serverUri))).toEqual({
      kind: 'tiled',
      path,
      serverUri,
    });
  });

  it('round-trips local paths containing colons and Unicode', () => {
    const path = 'drive:name/測定/sample.tif';
    expect(parseSourceKey(buildSourceKey('local', path))).toEqual({
      kind: 'local',
      path,
      serverUri: null,
    });
  });

  it('parses the legacy empty-server tiled form', () => {
    expect(parseSourceKey('tiled::browse/sample')).toEqual({
      kind: 'tiled',
      path: 'browse/sample',
      serverUri: null,
    });
  });

  it('rejects malformed keys instead of silently treating them as local paths', () => {
    expect(() => parseSourceKey('mystery:sample')).toThrow(/source key/i);
  });
});
