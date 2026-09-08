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
});
