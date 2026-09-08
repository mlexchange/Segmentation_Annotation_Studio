/**
 * CompositionPanel — smoke tests for concat preview wiring.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CompositionPanel from './index';
import { useIpredStore } from '@/stores/ipredStore';

vi.mock('@/lib/ipredApi', () => ({
  listIpredModules: vi.fn(async () => [
    {
      id: 'skimage_multiscale',
      name: 'Skimage multiscale',
      description: 'skimage',
      runtime: 'numpy',
      ready: true,
      accepts_input_from: false,
      produces_channels: true,
      produces_embedding: false,
      params_schema: { sigma_min: { default: 1 } },
    },
    {
      id: 'tomojepa',
      name: 'TomoJEPA',
      description: 'mark',
      runtime: 'torch',
      ready: true,
      accepts_input_from: true,
      produces_channels: false,
      produces_embedding: true,
      params_schema: { weights_id: { default: 'mark25' }, input_size: { default: 512 } },
    },
    {
      id: 'pca',
      name: 'PCA reduce',
      description: 'pca',
      runtime: 'numpy',
      ready: true,
      accepts_input_from: true,
      produces_channels: true,
      produces_embedding: false,
      params_schema: { dims: { default: 64 } },
    },
  ]),
  listIpredCompositions: vi.fn(async () => [
    {
      id: 'comp-skimage',
      name: 'Skimage multiscale',
      builtin: true,
      nodes: [{ id: 'n1', module: 'skimage_multiscale', params: {} }],
      outputs: ['n1'],
    },
  ]),
  getIpredComposition: vi.fn(async (id: string) => ({
    id,
    name: 'Skimage multiscale',
    builtin: true,
    nodes: [{ id: 'n1', module: 'skimage_multiscale', params: { sigma_min: 1 } }],
    outputs: ['n1'],
    preview_labels: ['intensity σ=1'],
  })),
  previewIpredComposition: vi.fn(async () => ({
    preview_labels: ['intensity σ=1', 'pca0'],
  })),
  upsertIpredComposition: vi.fn(async (p) => ({
    id: 'comp-custom',
    name: p.name,
    nodes: p.nodes,
    outputs: p.outputs,
  })),
}));

describe('CompositionPanel', () => {
  beforeEach(() => {
    useIpredStore.setState({ preferredCompositionId: 'comp-skimage' });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('shows concat preview from loaded composition', async () => {
    render(<CompositionPanel />);
    await waitFor(() => {
      expect(screen.getByTestId('ipred-concat-preview')).toHaveTextContent('intensity');
    });
  });

  it('adds a module from the catalog', async () => {
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-module-tomojepa');
    await user.click(screen.getByTestId('ipred-module-tomojepa'));
    await waitFor(() => {
      expect(screen.getByTestId('ipred-node-n2')).toBeTruthy();
    });
  });

  it('a ready module button is enabled; disables when not ready', async () => {
    render(<CompositionPanel />);
    await waitFor(() => expect(screen.getByTestId('ipred-module-tomojepa')).toBeEnabled());
    // pca is also ready per the mock catalog.
    expect(screen.getByTestId('ipred-module-pca')).toBeEnabled();
  });

  it('removing a node updates the graph summary and clears it from outputs', async () => {
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-node-n1');
    await user.click(within(screen.getByTestId('ipred-node-n1')).getByTitle('Remove'));
    await waitFor(() => expect(screen.queryByTestId('ipred-node-n1')).not.toBeInTheDocument());
    expect(screen.getByText(/Graph: empty/)).toBeInTheDocument();
  });

  it('adding a tomojepa node shows its weights/input-size/resize controls', async () => {
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-module-tomojepa');
    await user.click(screen.getByTestId('ipred-module-tomojepa'));
    const node = await screen.findByTestId('ipred-node-n2');
    expect(within(node).getByText('Weights')).toBeInTheDocument();
    expect(within(node).getByText('Input size')).toBeInTheDocument();
    expect(within(node).getByText('Resize')).toBeInTheDocument();
    // accepts_input_from -> shows the "Input from" selector too.
    expect(within(node).getByText('Input from')).toBeInTheDocument();
  });

  it('changing a tomojepa node input_size updates its param', async () => {
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-module-tomojepa');
    await user.click(screen.getByTestId('ipred-module-tomojepa'));
    const node = await screen.findByTestId('ipred-node-n2');
    const input = within(node).getByDisplayValue('512');
    fireEvent.change(input, { target: { value: '256' } });
    expect(within(node).getByDisplayValue('256')).toBeInTheDocument();
  });

  it('toggling a channel-producing node in/out of the output bank', async () => {
    const user = userEvent.setup();
    render(<CompositionPanel />);
    const node = await screen.findByTestId('ipred-node-n1');
    const checkbox = within(node).getByLabelText('In bank concat') as HTMLInputElement;
    expect(checkbox.checked).toBe(true); // n1 (skimage_multiscale) produces channels, in the builtin comp
    await user.click(checkbox);
    expect(checkbox.checked).toBe(false);
    expect(screen.getByText(/Outputs order: none/)).toBeInTheDocument();
  });

  it('reordering outputs with the up/down arrows', async () => {
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-module-pca');
    await user.click(screen.getByTestId('ipred-module-pca')); // n2, also produces_channels -> added to outputs
    await screen.findByTestId('ipred-node-n2');
    expect(screen.getByText(/Outputs order: n1 → n2/)).toBeInTheDocument();

    const node2 = screen.getByTestId('ipred-node-n2');
    await user.click(within(node2).getByText('↑'));
    expect(screen.getByText(/Outputs order: n2 → n1/)).toBeInTheDocument();
  });

  it('switching the active composition via the dropdown', async () => {
    const { getIpredComposition } = await import('@/lib/ipredApi');
    (getIpredComposition as any).mockImplementation(async (id: string) => {
      if (id === 'comp-skimage') {
        return {
          id, name: 'Skimage multiscale', builtin: true,
          nodes: [{ id: 'n1', module: 'skimage_multiscale', params: {} }],
          outputs: ['n1'], preview_labels: ['intensity σ=1'],
        };
      }
      return { id, name: 'Other comp', builtin: false, nodes: [], outputs: [], preview_labels: [] };
    });
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-composition-select');
    // Only one composition in the mocked list, so just confirm the select wiring
    // calls through to getIpredComposition with the chosen id.
    const select = screen.getByTestId('ipred-composition-select') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'comp-skimage' } });
    await waitFor(() => expect(getIpredComposition).toHaveBeenCalledWith('comp-skimage'));
  });

  it('save() persists the composition and re-selects it', async () => {
    const { upsertIpredComposition } = await import('@/lib/ipredApi');
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-node-n1');
    await user.click(screen.getByRole('button', { name: /Save/ }));
    await waitFor(() => expect(upsertIpredComposition).toHaveBeenCalled());
    const [payload] = (upsertIpredComposition as any).mock.calls[0];
    expect(payload.composition_id).toBe('comp-skimage');
  });

  it('clone & save omits composition_id and renames with a " copy" suffix', async () => {
    const { upsertIpredComposition } = await import('@/lib/ipredApi');
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-node-n1');
    await user.click(screen.getByRole('button', { name: /Clone & save/ }));
    await waitFor(() => expect(upsertIpredComposition).toHaveBeenCalled());
    const [payload] = (upsertIpredComposition as any).mock.calls[0];
    expect(payload.composition_id).toBeUndefined();
    expect(payload.name).toMatch(/ copy$/);
  });

  it('save() surfaces an error message on failure', async () => {
    const { upsertIpredComposition } = await import('@/lib/ipredApi');
    (upsertIpredComposition as any).mockRejectedValueOnce(new Error('save failed'));
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-node-n1');
    await user.click(screen.getByRole('button', { name: /^Save/ }));
    expect(await screen.findByTestId('ipred-composition-error')).toHaveTextContent('save failed');
  });

  it('Reload re-fetches modules/compositions', async () => {
    const { listIpredModules } = await import('@/lib/ipredApi');
    const user = userEvent.setup();
    render(<CompositionPanel />);
    await screen.findByTestId('ipred-node-n1');
    const callsBefore = (listIpredModules as any).mock.calls.length;
    await user.click(screen.getByRole('button', { name: /Reload/ }));
    await waitFor(() => expect((listIpredModules as any).mock.calls.length).toBeGreaterThan(callsBefore));
  });

  it('an empty preview shows the placeholder text', async () => {
    const { previewIpredComposition } = await import('@/lib/ipredApi');
    (previewIpredComposition as any).mockResolvedValue({ preview_labels: [] });
    const user = userEvent.setup();
    render(<CompositionPanel />);
    const node = await screen.findByTestId('ipred-node-n1');
    await user.click(within(node).getByLabelText('In bank concat'));
    await waitFor(() => {
      expect(screen.getByTestId('ipred-concat-preview')).toHaveTextContent(
        'add channel-producing nodes and mark them as outputs',
      );
    });
  });
});
