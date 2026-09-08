import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LayersPanel from './index';
import { useClassStore } from '@/stores/classStore';
import { useLayerVisibilityStore } from '@/stores/layerVisibilityStore';

const initialLayerState = useLayerVisibilityStore.getState();

beforeEach(() => {
  useLayerVisibilityStore.setState(initialLayerState, true);
  useClassStore.setState({ classes: [] });
});

afterEach(() => {
  cleanup();
});

describe('LayersPanel', () => {
  it('disables groups whose data has not loaded', () => {
    render(<LayersPanel />);
    expect(screen.getByRole('button', { name: /Features/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Probability/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Predictions/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Suggest/ })).toBeDisabled();
  });

  it('enables a group once its data is present', () => {
    render(<LayersPanel hasFeatures />);
    expect(screen.getByRole('button', { name: /Features/ })).toBeEnabled();
  });

  it('toggling a group flips its visibility icon', async () => {
    const user = userEvent.setup();
    render(<LayersPanel />);
    const imageToggle = screen.getByRole('button', { name: /Image/ });
    expect(useLayerVisibilityStore.getState().groups.image).toBe(true);
    await user.click(imageToggle);
    expect(useLayerVisibilityStore.getState().groups.image).toBe(false);
  });

  it('annotations group disabled with no classes, shown once classes exist', () => {
    render(<LayersPanel />);
    expect(screen.getByRole('button', { name: /Annotations/ })).toBeDisabled();
    expect(screen.queryByText('Human')).not.toBeInTheDocument();

    cleanup();
    useClassStore.setState({ classes: [{ classId: 1, label: 'Cell', color: '#f00', isVisible: true }] });
    render(<LayersPanel />);
    expect(screen.getByRole('button', { name: /Annotations/ })).toBeEnabled();
    expect(screen.getByText('Human')).toBeInTheDocument();
    expect(screen.getByText('Predicted')).toBeInTheDocument();
  });

  it('probability opacity slider only shows once proba is loaded and visible', () => {
    render(<LayersPanel hasProba />);
    expect(screen.getByText('Probability opacity')).toBeInTheDocument();
  });

  it('prediction controls list per-class visibility toggles', () => {
    useClassStore.setState({
      classes: [
        { classId: 1, label: 'Cell', color: '#f00', isVisible: true },
        { classId: 2, label: 'Wall', color: '#0f0', isVisible: true },
      ],
    });
    render(<LayersPanel hasPredictions predictionClassIds={[1, 2]} />);
    expect(screen.getByText('Cell')).toBeInTheDocument();
    expect(screen.getByText('Wall')).toBeInTheDocument();
  });

  it('clicking a prediction class toggle flips its visibility', async () => {
    useClassStore.setState({ classes: [{ classId: 1, label: 'Cell', color: '#f00', isVisible: true }] });
    const user = userEvent.setup();
    render(<LayersPanel hasPredictions predictionClassIds={[1]} />);
    await user.click(screen.getByText('Cell'));
    expect(useLayerVisibilityStore.getState().predictionClassVisible[1]).toBe(false);
  });
});
