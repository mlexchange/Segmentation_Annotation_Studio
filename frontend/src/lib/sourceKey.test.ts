import { describe, it, expect } from 'vitest';
import { buildSourceKey, isAnnotatedPath, parseSourceKey } from './sourceKey';

const URI = 'http://127.0.0.1:8010';
const key = (path: string) => buildSourceKey('tiled', path, URI);

describe('parseSourceKey', () => {
  it('round-trips a tiled key whose server URI has a port (colons in serverUri)', () => {
    const sk = buildSourceKey('tiled', 'browse/rec20260221_135217_petiole22', URI);
    expect(parseSourceKey(sk)).toEqual({ kind: 'tiled', source: 'browse/rec20260221_135217_petiole22', serverUri: URI });
  });

  it('round-trips a tiled key with a nested path', () => {
    const sk = buildSourceKey('tiled', 'browse/ds/img_00003', URI);
    expect(parseSourceKey(sk)).toEqual({ kind: 'tiled', source: 'browse/ds/img_00003', serverUri: URI });
  });

  it('round-trips a local key', () => {
    const sk = buildSourceKey('local', 'some/rel/path.tif');
    expect(parseSourceKey(sk)).toEqual({ kind: 'local', source: 'some/rel/path.tif', serverUri: null });
  });

  it('handles a tiled key with no server URI', () => {
    const sk = buildSourceKey('tiled', 'browse/ds', null);
    expect(parseSourceKey(sk)).toEqual({ kind: 'tiled', source: 'browse/ds', serverUri: null });
  });
});

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
