import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DenoisePanel, { type DenoiseMethodInfo } from './index';
import type { DenoiseOpts } from '@/stores/datasetStore';

const METHODS: DenoiseMethodInfo[] = [
  { method: 'none', label: 'None', cost: 'cheap', description: '', available: true, z_radius: 0 },
  { method: 'gaussian', label: 'Gaussian', cost: 'cheap', description: 'Simple blur.', available: true, z_radius: 0 },
  { method: 'bilateral', label: 'Bilateral', cost: 'slow', description: 'Edge-preserving.', available: true, z_radius: 0 },
  { method: 'nlm', label: 'NLM', cost: 'moderate', description: 'Non-local means.', available: false, z_radius: 0 },
];

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function baseDenoise(overrides: Partial<DenoiseOpts> = {}): DenoiseOpts {
  return { method: 'none', strength: 0.5, ...overrides } as DenoiseOpts;
}

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (String(url).includes('/api/denoise/methods')) {
        return { ok: true, json: async () => ({ methods: METHODS }) } as Response;
      }
      if (String(url).includes('/api/denoise/auto')) {
        return { ok: true, json: async () => ({ strength: 0.42, noise_sigma: 0.01 }) } as Response;
      }
      return { ok: false, json: async () => ({}) } as Response;
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('DenoisePanel', () => {
  it('loads and lists the available methods from the server', async () => {
    const onChange = vi.fn();
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise()}
        onChange={onChange}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'None' })).toBeInTheDocument();
    });
    expect(screen.getByRole('option', { name: 'Gaussian' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /Bilateral \(slow\)/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /NLM — unavailable/ })).toBeDisabled();
  });

  it('does not show the strength slider or extra controls when method is none', async () => {
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'none' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByRole('option', { name: 'None' })).toBeInTheDocument());
    expect(screen.queryByText('Strength')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Auto/ })).not.toBeInTheDocument();
  });

  it('shows the strength slider, description, and Auto button when a method is active', async () => {
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByText('Simple blur.')).toBeInTheDocument());
    expect(screen.getByText('Strength')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Auto/ })).toBeInTheDocument();
  });

  it('calls onChange with the new method when the select changes', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'none' })}
        onChange={onChange}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByRole('option', { name: 'Gaussian' })).toBeInTheDocument());
    await user.selectOptions(screen.getByRole('combobox'), 'gaussian');
    expect(onChange).toHaveBeenCalledWith({ method: 'gaussian' });
  });

  it('shows the "Save denoised copy" button only when onBake is provided', async () => {
    const onBake = vi.fn();
    const user = userEvent.setup();
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
        onBake={onBake}
      />,
    );
    await waitFor(() => expect(screen.getByText('Save denoised copy…')).toBeInTheDocument());
    await user.click(screen.getByText('Save denoised copy…'));
    expect(onBake).toHaveBeenCalledOnce();
  });

  it('omits "Save denoised copy" when onBake is not provided', async () => {
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByText('Strength')).toBeInTheDocument());
    expect(screen.queryByText('Save denoised copy…')).not.toBeInTheDocument();
  });

  it('shows the busy indicator only while active and busy', async () => {
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
        busy
      />,
    );
    await waitFor(() => expect(screen.getByText('filtering…')).toBeInTheDocument());
  });

  it('applyAuto calls onChange with the measured strength', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={onChange}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={3}
      />,
    );
    await waitFor(() => expect(screen.getByRole('button', { name: /Auto/ })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Auto/ }));
    await waitFor(() => expect(onChange).toHaveBeenCalledWith({ strength: 0.42 }));
    const [url] = (fetch as any).mock.calls.find(([u]: [string]) => String(u).includes('/api/denoise/auto'));
    expect(url).toContain('slice_index=3');
    expect(url).toContain('method=gaussian');
  });

  it('shows a clean-slice warning when Auto measures zero strength', async () => {
    (fetch as any).mockImplementation(async (url: string) => {
      if (String(url).includes('/api/denoise/methods')) {
        return { ok: true, json: async () => ({ methods: METHODS }) };
      }
      if (String(url).includes('/api/denoise/auto')) {
        return { ok: true, json: async () => ({ strength: 0, noise_sigma: 0.002 }) };
      }
      return { ok: false, json: async () => ({}) };
    });
    const user = userEvent.setup();
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByRole('button', { name: /Auto/ })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Auto/ }));
    expect(await screen.findByText(/This slice looks clean/)).toBeInTheDocument();
  });

  it('shows an error message when the Auto request fails', async () => {
    (fetch as any).mockImplementation(async (url: string) => {
      if (String(url).includes('/api/denoise/methods')) {
        return { ok: true, json: async () => ({ methods: METHODS }) };
      }
      if (String(url).includes('/api/denoise/auto')) {
        return { ok: false, json: async () => ({}) };
      }
      return { ok: false, json: async () => ({}) };
    });
    const user = userEvent.setup();
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByRole('button', { name: /Auto/ })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Auto/ }));
    expect(await screen.findByText('Could not measure this slice')).toBeInTheDocument();
  });

  it('disables the Auto button when there is no source', async () => {
    renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source={null}
        kind={null}
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByRole('button', { name: /Auto/ })).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /Auto/ })).toBeDisabled();
  });

  it('clears a stale auto error when the method changes', async () => {
    (fetch as any).mockImplementation(async (url: string) => {
      if (String(url).includes('/api/denoise/methods')) {
        return { ok: true, json: async () => ({ methods: METHODS }) };
      }
      if (String(url).includes('/api/denoise/auto')) {
        return { ok: false, json: async () => ({}) };
      }
      return { ok: false, json: async () => ({}) };
    });
    const user = userEvent.setup();
    const { rerender } = renderWithClient(
      <DenoisePanel
        denoise={baseDenoise({ method: 'gaussian' })}
        onChange={vi.fn()}
        source="a.tif"
        kind="local"
        serverUri={null}
        sliceIndex={0}
      />,
    );
    await waitFor(() => expect(screen.getByRole('button', { name: /Auto/ })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Auto/ }));
    expect(await screen.findByText('Could not measure this slice')).toBeInTheDocument();

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    rerender(
      <QueryClientProvider client={client}>
        <DenoisePanel
          denoise={baseDenoise({ method: 'bilateral' })}
          onChange={vi.fn()}
          source="a.tif"
          kind="local"
          serverUri={null}
          sliceIndex={0}
        />
      </QueryClientProvider>,
    );
    expect(screen.queryByText('Could not measure this slice')).not.toBeInTheDocument();
  });
});
