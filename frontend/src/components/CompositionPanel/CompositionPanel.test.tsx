/**
 * CompositionPanel — smoke tests for concat preview wiring.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CompositionPanel from './index';
import { useConnectionStore } from '@/stores/connectionStore';

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
    useConnectionStore.setState({
      preferredCompositionId: 'comp-skimage',
      preferredFeatureSetupId: 'comp-skimage',
    });
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
});
