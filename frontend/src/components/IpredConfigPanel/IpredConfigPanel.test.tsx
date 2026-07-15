/**
 * IpredConfigPanel — health + trainer.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import IpredConfigPanel from './index';
import { useConnectionStore } from '@/stores/connectionStore';

vi.mock('@/lib/ipredApi', () => ({
  ipredHealth: vi.fn(async () => ({ status: 'ok', service: 'ipred' })),
  listIpredTrainers: vi.fn(async () => ['catboost']),
}));

describe('IpredConfigPanel', () => {
  beforeEach(() => {
    useConnectionStore.setState({
      preferredCompositionId: 'comp-skimage-slimsam',
      preferredFeatureSetupId: 'comp-skimage-slimsam',
      preferredTrainerId: 'catboost',
      preferredTrainerConfig: { iterations: 200, depth: 6, learning_rate: 0.1 },
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('shows reachable health when ipred responds ok', async () => {
    render(<IpredConfigPanel />);
    await waitFor(() => {
      expect(screen.getByTestId('ipred-health-status')).toHaveTextContent('ipred reachable');
    });
  });

  it('shows active composition id', async () => {
    render(<IpredConfigPanel />);
    await waitFor(() => {
      expect(screen.getByText(/comp-skimage-slimsam/)).toBeTruthy();
    });
  });

  it('shows down health when health check fails', async () => {
    const { ipredHealth } = await import('@/lib/ipredApi');
    vi.mocked(ipredHealth).mockRejectedValueOnce(new Error('offline'));
    render(<IpredConfigPanel />);
    await waitFor(() => {
      expect(screen.getByTestId('ipred-health-status')).toHaveTextContent('unreachable');
    });
  });
});
