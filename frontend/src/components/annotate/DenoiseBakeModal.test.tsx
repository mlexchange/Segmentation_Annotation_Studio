import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DenoiseBakeModal, { type DenoiseBakeModalProps } from './DenoiseBakeModal';

function renderModal(overrides: Partial<DenoiseBakeModalProps> = {}) {
  const props: DenoiseBakeModalProps = {
    open: true,
    onClose: vi.fn(),
    source: 'browse/foo',
    serverUri: 'http://tiled.example',
    denoise: { method: 'tv', strength: 0.42 },
    nSlices: 10,
    methodLabel: 'Total variation',
    ...overrides,
  };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');
  const utils = render(
    <QueryClientProvider client={queryClient}>
      <DenoiseBakeModal {...props} />
    </QueryClientProvider>
  );
  return { ...utils, props, invalidateSpy };
}

beforeEach(() => {
  global.fetch = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('DenoiseBakeModal', () => {
  it('renders nothing when open is false', () => {
    const { container } = renderModal({ open: false });
    expect(container).toBeEmptyDOMElement();
  });

  it('shows method label, strength, and slice count; prefills destination', () => {
    renderModal({ source: 'browse/foo', methodLabel: 'Total variation', denoise: { method: 'tv', strength: 0.42 }, nSlices: 10 });
    expect(screen.getByText(/Total variation, strength 0\.42/)).toBeInTheDocument();
    expect(screen.getByText(/all 10 slices/)).toBeInTheDocument();
    expect(screen.getByDisplayValue('browse/foo_denoised')).toBeInTheDocument();
  });

  it('uses singular "slice" when nSlices is 1', () => {
    renderModal({ nSlices: 1 });
    expect(screen.getByText(/all 1 slice and written/)).toBeInTheDocument();
  });

  it('calls onClose from the Cancel button and the X button', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    renderModal({ onClose });
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledTimes(1);
    await user.click(screen.getByLabelText('Close'));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('disables Save copy when destination is blank', async () => {
    const user = userEvent.setup();
    renderModal();
    const input = screen.getByDisplayValue('browse/foo_denoised');
    await user.clear(input);
    expect(screen.getByRole('button', { name: 'Save copy' })).toBeDisabled();
  });

  it('resets destination/description when reopened for a different source', () => {
    const { rerender, props } = renderModal({ source: 'browse/foo' });
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <DenoiseBakeModal {...props} source="browse/bar" />
      </QueryClientProvider>
    );
    expect(screen.getByDisplayValue('browse/bar_denoised')).toBeInTheDocument();
  });

  it('starts a bake job, polls status, shows progress, then completion and invalidates the browse query', async () => {
    let statusCall = 0;
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string, opts?: RequestInit) => {
      if (url.includes('/api/denoise/bake')) {
        expect(opts?.method).toBe('POST');
        const body = JSON.parse(opts!.body as string);
        expect(body).toEqual({
          source: 'browse/foo',
          server_uri: 'http://tiled.example',
          method: 'tv',
          strength: 0.42,
          target_path: 'browse/foo_denoised',
          description: '',
        });
        return { ok: true, json: async () => ({ job_id: 'job-1' }) };
      }
      if (url.includes('/api/export/status/job-1')) {
        statusCall++;
        if (statusCall === 1) {
          return {
            ok: true,
            json: async () => ({ state: 'running', phase: 'Denoising', done: 3, total: 10, error: null, result: null }),
          };
        }
        return {
          ok: true,
          json: async () => ({ state: 'done', phase: 'Done', done: 10, total: 10, error: null, result: { path: 'browse/foo_denoised', n_slices: 10 } }),
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    });

    const user = userEvent.setup();
    const { invalidateSpy } = renderModal();
    await user.click(screen.getByRole('button', { name: 'Save copy' }));

    await waitFor(() => {
      expect(screen.getByText(/Denoising — 3\/10/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('Saved 10 slices to browse/foo_denoised.')).toBeInTheDocument();
    }, { timeout: 3000 });

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['browse'] });
  }, 10000);

  it('shows a cancelled message when the result reports cancelled', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/denoise/bake')) {
        return { ok: true, json: async () => ({ job_id: 'job-2' }) };
      }
      if (url.includes('/api/export/status/job-2')) {
        return {
          ok: true,
          json: async () => ({ state: 'done', phase: 'Done', done: 4, total: 10, error: null, result: { cancelled: true } }),
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: 'Save copy' }));
    await waitFor(() => {
      expect(screen.getByText('Stopped — the partial copy was discarded.')).toBeInTheDocument();
    }, { timeout: 3000 });
  }, 10000);

  it('shows an error message when the bake request fails', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/denoise/bake')) {
        return { ok: false, status: 500, json: async () => ({ detail: 'boom' }) };
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: 'Save copy' }));
    await waitFor(() => {
      expect(screen.getByText('boom')).toBeInTheDocument();
    });
  });

  it('calls the cancel endpoint when Stop is clicked while running', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(async (url: string) => {
      if (url.includes('/api/denoise/bake')) {
        return { ok: true, json: async () => ({ job_id: 'job-3' }) };
      }
      if (url.includes('/api/export/status/job-3')) {
        // Never resolves to done so it stays in "running" while we click Stop.
        return {
          ok: true,
          json: async () => ({ state: 'running', phase: 'Denoising', done: 1, total: 10, error: null, result: null }),
        };
      }
      if (url.includes('/api/export/cancel/job-3')) {
        return { ok: true, json: async () => ({}) };
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: 'Save copy' }));
    await waitFor(() => screen.getByRole('button', { name: 'Stop' }));
    await user.click(screen.getByRole('button', { name: 'Stop' }));
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/export/cancel/job-3'),
        expect.objectContaining({ method: 'POST' })
      );
    });
  });
});
