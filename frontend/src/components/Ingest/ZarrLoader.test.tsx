import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ZarrLoader from './ZarrLoader';

function jsonResponse(body: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => body, text: async () => JSON.stringify(body) } as Response;
}

const ZARR_INFO = {
  name: 'volume.zarr',
  path: '/data/volume.zarr',
  levels: [
    { path: '0', shape: [100, 512, 512], dtype: 'uint16', n_slices: 100, height: 512, width: 512, downsample: [1, 1, 1] },
    { path: '1', shape: [100, 256, 256], dtype: 'uint16', n_slices: 100, height: 256, width: 256, downsample: [1, 2, 2] },
  ],
  full_shape: [100, 512, 512],
  dtype: 'uint16',
  voxel_size: [1.5, 1.5, 1.5],
  voxel_unit: 'um',
};

function renderLoader(overrides: Partial<React.ComponentProps<typeof ZarrLoader>> = {}) {
  const props = {
    serverUri: 'http://tiled.example',
    onBrowse: vi.fn(),
    onAnnotate: vi.fn(),
    ...overrides,
  };
  const utils = render(<ZarrLoader {...props} />);
  return { ...utils, props };
}

async function typePathAndInspect(user: ReturnType<typeof userEvent.setup>, path = '/data/volume.zarr') {
  const input = screen.getByPlaceholderText('/absolute/path/to/volume.zarr');
  await user.type(input, path);
  await user.click(screen.getByRole('button', { name: /Inspect/ }));
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('ZarrLoader', () => {
  it('disables Inspect until a path is entered', () => {
    renderLoader();
    expect(screen.getByRole('button', { name: /Inspect/ })).toBeDisabled();
  });

  it('inspects a path, shows volume info and resolution levels, no conflict', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight')) return jsonResponse({ exists: false });
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    await typePathAndInspect(user);

    expect(await screen.findByText('volume.zarr')).toBeInTheDocument();
    expect(screen.getByText(/100 × 512 × 512 · uint16 · 1.5 um\/voxel/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Load volume/ })).not.toBeDisabled();
  });

  it('inspecting via Enter key also triggers the request', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight')) return jsonResponse({ exists: false });
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    const input = screen.getByPlaceholderText('/absolute/path/to/volume.zarr');
    await user.type(input, '/data/volume.zarr{Enter}');
    expect(await screen.findByText('volume.zarr')).toBeInTheDocument();
  });

  it('shows an error message when inspect fails with a JSON detail', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse({ detail: 'no such path' }, false, 404);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    await typePathAndInspect(user);
    expect(await screen.findByText('no such path')).toBeInTheDocument();
  });

  it('switching resolution level updates the coarse-level warning', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight')) return jsonResponse({ exists: false });
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    await typePathAndInspect(user);
    await screen.findByText('volume.zarr');

    // Full res (level 0) selected by default: no coarse warning.
    expect(screen.queryByText(/Annotations are stored in full-resolution/)).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /1\/2/ }));
    expect(await screen.findByText(/Annotations are stored in full-resolution/)).toBeInTheDocument();
  });

  it('shows a blocking conflict when the destination holds different uploaded data', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight'))
        return jsonResponse({
          exists: true,
          existing: { child_count: 4, external: false, sample_name: 'existing', n_images: 4 },
        });
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    await typePathAndInspect(user);

    expect(await screen.findByText(/and holds 4 uploaded images/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Load volume/ })).toBeDisabled();
    // Not an "external" zarr, so no "Replace it" shortcut is offered.
    expect(screen.queryByRole('button', { name: 'Replace it' })).not.toBeInTheDocument();
  });

  it('offers "Replace it" for a conflict with a previously loaded external zarr, and registers on click', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string, opts?: RequestInit) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight'))
        return jsonResponse({
          exists: true,
          existing: { child_count: 1, external: true, sample_name: 'volume', n_images: null },
        });
      if (url.includes('/api/zarr/register')) {
        const body = JSON.parse(opts!.body as string);
        expect(body.on_conflict).toBe('replace');
        return jsonResponse({ tiled_path: 'browse/volume' });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    await typePathAndInspect(user);

    expect(await screen.findByText(/previously loaded Zarr/)).toBeInTheDocument();
    // Load volume stays enabled for an external conflict.
    expect(screen.getByRole('button', { name: /Load volume/ })).not.toBeDisabled();

    await user.click(screen.getByRole('button', { name: 'Replace it' }));
    expect(await screen.findByText(/Loaded 100 slices/)).toBeInTheDocument();
  });

  it('registers a new volume (Load volume), shows success, and Annotate/Browse call back with the level path', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string, opts?: RequestInit) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight')) return jsonResponse({ exists: false });
      if (url.includes('/api/zarr/register')) {
        const body = JSON.parse(opts!.body as string);
        expect(body).toMatchObject({
          path: '/data/volume.zarr',
          container_path: 'browse',
          on_conflict: 'fail',
          server_uri: 'http://tiled.example',
        });
        return jsonResponse({ tiled_path: 'browse/volume' });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    const { props } = renderLoader();
    await typePathAndInspect(user);
    await screen.findByText('volume.zarr');

    await user.click(screen.getByRole('button', { name: /Load volume/ }));
    expect(await screen.findByText(/Loaded 100 slices — no data was copied\./)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Annotate' }));
    expect(props.onAnnotate).toHaveBeenCalledWith('browse/volume/0');

    await user.click(screen.getByRole('button', { name: 'Browse' }));
    expect(props.onBrowse).toHaveBeenCalledWith('browse', 1);
  });

  it('shows an error message when register fails', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(ZARR_INFO);
      if (url.includes('/api/zarr/preflight')) return jsonResponse({ exists: false });
      if (url.includes('/api/zarr/register')) return jsonResponse({ detail: 'disk full' }, false, 500);
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderLoader();
    await typePathAndInspect(user);
    await screen.findByText('volume.zarr');

    await user.click(screen.getByRole('button', { name: /Load volume/ }));
    expect(await screen.findByText('disk full')).toBeInTheDocument();
  });

  describe('server-side directory browser', () => {
    function mockBrowseFetch(entriesByRel: Record<string, { name: string; path: string; is_dir: boolean }[]>) {
      return vi.fn(async (url: string) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data/raw' });
        if (url.includes('/api/local/list')) {
          const rel = new URL(url, 'http://x').searchParams.get('rel') ?? '';
          return jsonResponse(entriesByRel[rel] ?? []);
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
    }

    it('opens the browser, fetches the root, and lists directories (files filtered out)', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(
        mockBrowseFetch({
          '': [
            { name: 'scratch', path: 'scratch', is_dir: true },
            { name: 'readme.txt', path: 'readme.txt', is_dir: false },
          ],
        }),
      );
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      expect(await screen.findByText('/data/raw')).toBeInTheDocument();
      expect(screen.getByText('scratch')).toBeInTheDocument();
      expect(screen.queryByText('readme.txt')).not.toBeInTheDocument();
    });

    it('descends into a plain subfolder, and selecting a .zarr entry fills the path and closes the browser', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(
        mockBrowseFetch({
          '': [{ name: 'scratch', path: 'scratch', is_dir: true }],
          scratch: [{ name: 'ant_m12.zarr', path: 'scratch/ant_m12.zarr', is_dir: true }],
        }),
      );
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      await screen.findByText('scratch');
      await user.click(screen.getByText('scratch'));

      expect(await screen.findByText('ant_m12.zarr')).toBeInTheDocument();
      await user.click(screen.getByText('ant_m12.zarr'));

      // Browser closes and the path field now has the full absolute path —
      // no more guessing a container-visible path blind.
      expect(screen.queryByText('ant_m12.zarr')).not.toBeInTheDocument();
      expect(screen.getByPlaceholderText('/absolute/path/to/volume.zarr')).toHaveValue(
        '/data/raw/scratch/ant_m12.zarr',
      );
    });

    it('"Use this folder" selects the currently-browsed directory even without a .zarr suffix', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(
        mockBrowseFetch({ '': [{ name: 'my_volume', path: 'my_volume', is_dir: true }] }),
      );
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      await user.click(await screen.findByText('my_volume'));
      await screen.findByText(/Use this folder/);
      await user.click(screen.getByText(/Use this folder/));

      expect(screen.getByPlaceholderText('/absolute/path/to/volume.zarr')).toHaveValue('/data/raw/my_volume');
    });

    it('shows an error if the root or a directory listing fails', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
        if (url.includes('/api/local/root')) return jsonResponse({ detail: 'no access' }, false, 500);
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      expect(await screen.findByText('no access')).toBeInTheDocument();
    });

    it('closes the browser when Browse… is clicked again', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(mockBrowseFetch({ '': [] }));
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      await screen.findByText('/data/raw');
      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      expect(screen.queryByText('/data/raw')).not.toBeInTheDocument();
    });

    it('an empty directory shows actionable LOCAL_SOURCE_DIR guidance, not a bare "no sub-folders" dead end', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(mockBrowseFetch({ '': [] }));
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      expect(await screen.findByText('No sub-folders here.')).toBeInTheDocument();
      expect(screen.getByText(/LOCAL_SOURCE_DIR=\/path\/to\/your\/data/)).toBeInTheDocument();
    });

    it('overriding the root (e.g. anywhere on disk when run via start_all.sh) browses from there instead', async () => {
      const listCalls: { root: string; rel: string }[] = [];
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) {
          const u = new URL(url, 'http://x');
          const root = u.searchParams.get('root') ?? '';
          const rel = u.searchParams.get('rel') ?? '';
          listCalls.push({ root, rel });
          if (root === '/Users/me/tomo' && rel === '') {
            return jsonResponse([{ name: 'ant_m12.zarr', path: 'ant_m12.zarr', is_dir: true }]);
          }
          return jsonResponse([]);
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      await screen.findByText('/data');

      const rootField = screen.getByPlaceholderText('Root to browse from');
      await user.clear(rootField);
      await user.type(rootField, '/Users/me/tomo');
      await user.click(screen.getByRole('button', { name: 'Go' }));

      expect(await screen.findByText('ant_m12.zarr')).toBeInTheDocument();
      expect(await screen.findByText('/Users/me/tomo')).toBeInTheDocument();
      expect(listCalls).toContainEqual({ root: '/Users/me/tomo', rel: '' });

      await user.click(screen.getByText('ant_m12.zarr'));
      expect(screen.getByPlaceholderText('/absolute/path/to/volume.zarr')).toHaveValue(
        '/Users/me/tomo/ant_m12.zarr',
      );
    });

    it('sends the granted root on every subsequent list call (breadcrumbs, descending)', async () => {
      const listCalls: { root: string; rel: string }[] = [];
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) {
          const u = new URL(url, 'http://x');
          const root = u.searchParams.get('root') ?? '';
          const rel = u.searchParams.get('rel') ?? '';
          listCalls.push({ root, rel });
          if (rel === '') return jsonResponse([{ name: 'sub', path: 'sub', is_dir: true }]);
          return jsonResponse([]);
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      await user.click(await screen.findByText('sub'));

      expect(listCalls).toContainEqual({ root: '/data', rel: 'sub' });
    });
  });

  it('a bare-array store (single level, empty level.path) builds the Tiled path without a trailing slash', async () => {
    const BARE_ARRAY_INFO = {
      name: 'plain.zarr',
      path: '/data/plain.zarr',
      levels: [
        { path: '', shape: [6, 10, 12], dtype: 'uint16', n_slices: 6, height: 10, width: 12, downsample: [1, 1, 1] },
      ],
      full_shape: [6, 10, 12],
      dtype: 'uint16',
      voxel_size: null,
      voxel_unit: null,
    };
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/zarr/inspect')) return jsonResponse(BARE_ARRAY_INFO);
      if (url.includes('/api/zarr/preflight')) return jsonResponse({ exists: false });
      if (url.includes('/api/zarr/register')) return jsonResponse({ tiled_path: 'browse/plain' });
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    const { props } = renderLoader();
    await typePathAndInspect(user, '/data/plain.zarr');
    await screen.findByText('plain.zarr');

    await user.click(screen.getByRole('button', { name: /Load volume/ }));
    await screen.findByText(/Loaded 6 slices — no data was copied\./);

    await user.click(screen.getByRole('button', { name: 'Annotate' }));
    expect(props.onAnnotate).toHaveBeenCalledWith('browse/plain');
  });

  describe('scan folder for Zarr volumes', () => {
    async function openBrowserAt(user: ReturnType<typeof userEvent.setup>, root = '/data') {
      await user.click(screen.getByRole('button', { name: /Browse…/ }));
      await screen.findByText(root);
    }

    it('scans the currently-browsed folder and reports registered/skipped/errors', async () => {
      const scanCalls: unknown[] = [];
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string, init?: RequestInit) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) return jsonResponse([]);
        if (url.includes('/api/scan-datasets')) {
          scanCalls.push(JSON.parse(String(init?.body)));
          return jsonResponse({
            scanned: 3,
            registered: [{ name: 'a.zarr', key: 'a', tiled_path: 'browse/a' }],
            skipped: ['b'],
            shadowed: [],
            errors: [{ name: 'c.zarr', error: 'boom' }],
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      const { props } = renderLoader({ serverUri: 'http://tiled.example' });

      await openBrowserAt(user);
      await user.click(screen.getByRole('button', { name: /Scan folder for datasets/ }));

      expect(await screen.findByText(/Scanned 3 — registered 1 new, skipped 1 already present, 1 failed\./))
        .toBeInTheDocument();
      expect(screen.getByText('a.zarr')).toBeInTheDocument();
      expect(screen.getByText(/c\.zarr/)).toBeInTheDocument();
      expect(screen.getByText(/boom/)).toBeInTheDocument();
      expect(scanCalls).toEqual([
        { scan_root: '/data', container_path: 'browse', server_uri: 'http://tiled.example' },
      ]);

      await user.click(screen.getByRole('button', { name: 'Go to Browse' }));
      expect(props.onBrowse).toHaveBeenCalledWith('browse', 1);
    });

    it('scans a descended-into subfolder using its full path, not just the root', async () => {
      const scanCalls: unknown[] = [];
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string, init?: RequestInit) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) {
          const rel = new URL(url, 'http://x').searchParams.get('rel') ?? '';
          if (rel === '') return jsonResponse([{ name: 'scratch', path: 'scratch', is_dir: true }]);
          return jsonResponse([]);
        }
        if (url.includes('/api/scan-datasets')) {
          scanCalls.push(JSON.parse(String(init?.body)));
          return jsonResponse({ scanned: 0, registered: [], skipped: [], shadowed: [], errors: [] });
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await openBrowserAt(user);
      await user.click(await screen.findByText('scratch'));
      await user.click(await screen.findByRole('button', { name: /Scan folder for datasets \(scratch\)/ }));

      await screen.findByText(/Scanned 0/);
      expect(scanCalls).toEqual([
        { scan_root: '/data/scratch', container_path: 'browse', server_uri: 'http://tiled.example' },
      ]);
    });

    it('shows an error if the scan request fails', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) return jsonResponse([]);
        if (url.includes('/api/scan-datasets')) return jsonResponse({ detail: 'no such directory' }, false, 404);
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await openBrowserAt(user);
      await user.click(screen.getByRole('button', { name: /Scan folder for datasets/ }));
      expect(await screen.findByText('no such directory')).toBeInTheDocument();
    });

    it('does not show "Go to Browse" when nothing new was registered', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) return jsonResponse([]);
        if (url.includes('/api/scan-datasets')) {
          return jsonResponse({ scanned: 1, registered: [], skipped: ['already-there'], shadowed: [], errors: [] });
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await openBrowserAt(user);
      await user.click(screen.getByRole('button', { name: /Scan folder for datasets/ }));
      await screen.findByText(/Scanned 1/);
      expect(screen.queryByRole('button', { name: 'Go to Browse' })).not.toBeInTheDocument();
    });

    it('shows a shadowed collision distinctly, and retrying it merges into the existing summary', async () => {
      const scanCalls: unknown[] = [];
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string, init?: RequestInit) => {
        if (url.includes('/api/local/root')) return jsonResponse({ root: '/data' });
        if (url.includes('/api/local/list')) return jsonResponse([]);
        if (url.includes('/api/scan-datasets')) {
          const body = JSON.parse(String(init?.body));
          scanCalls.push(body);
          if (body.renames) {
            return jsonResponse({
              scanned: 1,
              registered: [{ name: 'stack_a', key: 'stack_a_images', tiled_path: 'browse/stack_a_images' }],
              skipped: [],
              shadowed: [],
              errors: [],
            });
          }
          return jsonResponse({
            scanned: 1,
            registered: [],
            skipped: [],
            shadowed: [
              { name: 'stack_a', key: 'stack_a', existing_kind: 'zarr', suggested_key: 'stack_a_images' },
            ],
            errors: [],
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      });
      const user = userEvent.setup();
      renderLoader();

      await openBrowserAt(user);
      await user.click(screen.getByRole('button', { name: /Scan folder for datasets/ }));

      expect(await screen.findByText(/Scanned 1 — registered 0 new, skipped 0 already present, 1 shadowed\./))
        .toBeInTheDocument();
      expect(screen.getByText('stack_a')).toBeInTheDocument();
      expect(screen.getByText(/already registered as a/)).toBeInTheDocument();
      expect(screen.getByText('zarr')).toBeInTheDocument();
      const input = screen.getByDisplayValue('stack_a_images');

      await user.click(screen.getByRole('button', { name: 'Register as this' }));

      // The shadowed entry is gone once resolved, replaced by a registered
      // one (rendered by name, so still "stack_a" — merged in, not dangling).
      await waitFor(() => {
        expect(screen.queryByRole('button', { name: 'Register as this' })).not.toBeInTheDocument();
      });
      expect(screen.getByText(/registered 1 new/)).toBeInTheDocument();
      expect(scanCalls).toContainEqual(
        expect.objectContaining({ renames: { stack_a: 'stack_a_images' } }),
      );
      expect(input).toBeDefined();
    });
  });
});
