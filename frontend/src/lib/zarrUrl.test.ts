import { describe, it, expect } from 'vitest';
import { buildZarrUrl, zarrRootFor, describeUnavailable } from './zarrUrl';

describe('zarrRootFor', () => {
  it('appends the zarr v2 root to a bare origin', () => {
    expect(zarrRootFor('http://127.0.0.1:8010')).toBe('http://127.0.0.1:8010/zarr/v2');
  });

  it('tolerates a trailing slash', () => {
    expect(zarrRootFor('http://127.0.0.1:8010/')).toBe('http://127.0.0.1:8010/zarr/v2');
  });

  it('replaces a REST prefix rather than nesting under it', () => {
    // .../api/v1/zarr/v2/... would 404 in a way that reads as missing data.
    expect(zarrRootFor('https://tiled.example.org/api/v1')).toBe('https://tiled.example.org/zarr/v2');
  });

  it('does not hardcode a port', () => {
    // start_all.sh reassigns Tiled's port when 8010 is busy; the URI is the
    // source of truth, so whatever port it carries must survive.
    expect(zarrRootFor('http://127.0.0.1:8777')).toContain(':8777');
  });
});

describe('buildZarrUrl', () => {
  const server = 'http://127.0.0.1:8010';

  it('addresses a tiled node under the zarr root', () => {
    expect(buildZarrUrl('tiled', 'scans/petiole22', server)).toEqual({
      url: 'http://127.0.0.1:8010/zarr/v2/scans/petiole22',
      reason: null,
    });
  });

  it('preserves path separators while escaping each segment', () => {
    const { url } = buildZarrUrl('tiled', 'my scans/sample #1', server);
    expect(url).toBe('http://127.0.0.1:8010/zarr/v2/my%20scans/sample%20%231');
  });

  it('ignores empty path segments', () => {
    const { url } = buildZarrUrl('tiled', '/scans//petiole22/', server);
    expect(url).toBe('http://127.0.0.1:8010/zarr/v2/scans/petiole22');
  });

  it.each([
    ['no source open', null, null, server, 'no-source'],
    ['no path', 'tiled', null, server, 'no-source'],
    ['local file', 'local', 'foo.tif', null, 'local-source'],
    ['server unresolved', 'tiled', 'scans/x', null, 'no-server'],
  ])('reports %s rather than guessing a URL', (_label, kind, source, uri, reason) => {
    const result = buildZarrUrl(kind as string | null, source as string | null, uri as string | null);
    expect(result.url).toBeNull();
    expect(result.reason).toBe(reason);
  });

  it('never emits credentials in the URL', () => {
    // Anonymous read access is the contract; a key in the query string would
    // put a write-capable credential in browser history and referrers.
    const { url } = buildZarrUrl('tiled', 'scans/x', server);
    expect(url).not.toMatch(/api_key|token|Authorization/i);
  });
});

describe('describeUnavailable', () => {
  it('explains every reason', () => {
    for (const reason of ['no-source', 'local-source', 'no-server'] as const) {
      expect(describeUnavailable(reason).length).toBeGreaterThan(0);
    }
  });
});
