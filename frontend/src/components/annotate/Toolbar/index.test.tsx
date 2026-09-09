import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Toolbar from './index';
import { useToolStore } from '@/stores/toolStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import * as editHistory from '@/hooks/editHistory';

// Avoid spinning up the real SAM worker (unavailable in jsdom) — the magic-tool
// panel only needs a stable, controllable status/support surface.
vi.mock('@/hooks/useSam', () => ({
  useSam: vi.fn(() => ({
    status: 'idle',
    error: null,
    backend: null,
    webgpu: false,
    supported: true,
    ensureEncoded: vi.fn(),
    segment: vi.fn(),
  })),
}));

vi.mock('@/hooks/editHistory', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/editHistory')>();
  return {
    ...actual,
    undo: vi.fn(),
    redo: vi.fn(),
  };
});

const initialToolState = useToolStore.getState();

beforeEach(() => {
  useToolStore.setState(initialToolState, true);
  useAnnotationStore.getState().reset();
  useAnnotationStore.temporal.getState().clear();
  vi.mocked(editHistory.undo).mockClear();
  vi.mocked(editHistory.redo).mockClear();
});

afterEach(() => {
  cleanup();
});

describe('Toolbar', () => {
  it('renders every tool as a radio button, defaulting to Pan active', () => {
    render(<Toolbar />);
    const group = screen.getByRole('radiogroup', { name: 'Drawing tools' });
    const radios = within(group).getAllByRole('radio');
    expect(radios).toHaveLength(11);
    expect(screen.getByRole('radio', { name: /Pan/ })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('radio', { name: /Brush/ })).toHaveAttribute('aria-checked', 'false');
  });

  it('clicking a tool button selects it in the store and updates aria-checked', async () => {
    const user = userEvent.setup();
    render(<Toolbar />);
    await user.click(screen.getByRole('radio', { name: /Brush \(b\)/ }));
    expect(useToolStore.getState().tool).toBe('brush');
    expect(screen.getByRole('radio', { name: /Brush/ })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('radio', { name: /Pan/ })).toHaveAttribute('aria-checked', 'false');
  });

  it('reflects the active tool from the store even when set externally', () => {
    useToolStore.setState({ tool: 'polygon' });
    render(<Toolbar />);
    expect(screen.getByRole('radio', { name: /Polygon/ })).toHaveAttribute('aria-checked', 'true');
  });

  it('disables every tool button and shows the no-class hint when disabled', () => {
    render(<Toolbar disabled />);
    expect(screen.getByText('Add a class above to start annotating.')).toBeInTheDocument();
    const group = screen.getByRole('radiogroup', { name: 'Drawing tools' });
    within(group).getAllByRole('radio').forEach((btn) => {
      expect(btn).toBeDisabled();
      // Disabled tools are never shown as active, even the current tool.
      expect(btn).toHaveAttribute('aria-checked', 'false');
    });
  });

  it('does not show the no-class hint when not disabled', () => {
    render(<Toolbar />);
    expect(screen.queryByText('Add a class above to start annotating.')).not.toBeInTheDocument();
  });

  describe('undo/redo', () => {
    it('disables undo and redo when the history is empty', () => {
      render(<Toolbar />);
      expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();
    });

    it('enables undo when there is past history and calls editHistory.undo on click', async () => {
      useAnnotationStore.temporal.setState({ pastStates: [{} as any] });
      const user = userEvent.setup();
      render(<Toolbar />);
      const undoBtn = screen.getByRole('button', { name: 'Undo' });
      expect(undoBtn).toBeEnabled();
      await user.click(undoBtn);
      expect(editHistory.undo).toHaveBeenCalledTimes(1);
    });

    it('enables redo when there is future history and calls editHistory.redo on click', async () => {
      useAnnotationStore.temporal.setState({ futureStates: [{} as any] });
      const user = userEvent.setup();
      render(<Toolbar />);
      const redoBtn = screen.getByRole('button', { name: 'Redo' });
      expect(redoBtn).toBeEnabled();
      await user.click(redoBtn);
      expect(editHistory.redo).toHaveBeenCalledTimes(1);
    });
  });

  describe('brush / eraser / threshold panel', () => {
    it('shows the brush radius controls for the brush tool', () => {
      useToolStore.setState({ tool: 'brush' });
      render(<Toolbar />);
      expect(screen.getByLabelText('Brush radius (px)')).toBeInTheDocument();
    });

    it('does not show brush radius controls for tools without a brush', () => {
      useToolStore.setState({ tool: 'polygon' });
      render(<Toolbar />);
      expect(screen.queryByLabelText('Brush radius')).not.toBeInTheDocument();
    });

    it('changing the brush radius number input snaps and updates the store', () => {
      useToolStore.setState({ tool: 'brush', brushSize: 10 });
      render(<Toolbar />);
      const input = screen.getByLabelText('Brush radius (px)') as HTMLInputElement;
      fireEvent.change(input, { target: { value: '25' } });
      expect(useToolStore.getState().brushSize).toBe(25);
    });

    it('shows the erase-scope radiogroup only for the eraser tool', () => {
      useToolStore.setState({ tool: 'eraser' });
      render(<Toolbar />);
      expect(screen.getByRole('radiogroup', { name: 'Erase scope' })).toBeInTheDocument();
    });

    it('toggles erase scope between class and all classes', async () => {
      useToolStore.setState({ tool: 'eraser', eraseAllClasses: false });
      const user = userEvent.setup();
      render(<Toolbar />);
      const allRadio = screen.getByRole('radio', { name: 'Erase all classes' });
      expect(allRadio).not.toBeChecked();
      await user.click(allRadio);
      expect(useToolStore.getState().eraseAllClasses).toBe(true);
    });
  });

  describe('threshold / sampler panel', () => {
    it('shows the threshold band controls for the threshold tool', () => {
      useToolStore.setState({ tool: 'threshold' });
      render(<Toolbar />);
      expect(screen.getByText('Set band from a region')).toBeInTheDocument();
      expect(screen.getByText('Show in-range overlay')).toBeInTheDocument();
    });

    it('shows the threshold panel for the sampler tool too, in sampling mode', () => {
      useToolStore.setState({ tool: 'sampler' });
      render(<Toolbar />);
      expect(screen.getByText('Sampling — draw a loop')).toBeInTheDocument();
    });

    it('toggles into sampler mode and back via the sample button', async () => {
      useToolStore.setState({ tool: 'threshold' });
      const user = userEvent.setup();
      render(<Toolbar />);
      await user.click(screen.getByText('Set band from a region'));
      expect(useToolStore.getState().tool).toBe('sampler');
      await user.click(screen.getByText('Sampling — draw a loop'));
      expect(useToolStore.getState().tool).toBe('threshold');
    });

    it('toggles the threshold overlay checkbox', async () => {
      useToolStore.setState({ tool: 'threshold', thresholdOverlay: true });
      const user = userEvent.setup();
      render(<Toolbar />);
      const checkbox = screen.getByRole('checkbox', { name: 'Show in-range overlay' });
      expect(checkbox).toBeChecked();
      await user.click(checkbox);
      expect(useToolStore.getState().thresholdOverlay).toBe(false);
    });

    it('shows a Sampler result readout when samplerFit is provided', () => {
      useToolStore.setState({ tool: 'threshold' });
      render(
        <Toolbar
          samplerFit={{
            mode: 'plain',
            collapsed: false,
            lo: 10,
            hi: 200,
            displayLo: 10,
            displayHi: 200,
            dice: 0.9,
            skill: 0.8,
            coverage: 0.5,
            extraSigma: 0,
            appliedBlur: 0,
          } as any}
        />,
      );
      expect(screen.getByText(/match 90%/)).toBeInTheDocument();
    });

    it('shows the collapsed-band warning when the fit collapsed', () => {
      useToolStore.setState({ tool: 'threshold' });
      render(<Toolbar samplerFit={{ collapsed: true, lo: 10, hi: 20 } as any} />);
      expect(screen.getByText('Band not applied')).toBeInTheDocument();
    });

    it('offers "View band in 3D" for a plain (non-projected) fit and sends the native lo/hi', async () => {
      const user = userEvent.setup();
      const onSendBandTo3D = vi.fn();
      useToolStore.setState({ tool: 'threshold' });
      render(
        <Toolbar
          samplerFit={{
            mode: 'plain',
            collapsed: false,
            lo: 10,
            hi: 200,
            displayLo: 40,
            displayHi: 230,
            dice: 0.9,
            skill: 0.8,
            coverage: 0.5,
            extraSigma: 0,
            appliedBlur: 0,
          } as any}
          onSendBandTo3D={onSendBandTo3D}
        />,
      );
      const button = screen.getByRole('button', { name: /view band in 3d/i });
      await user.click(button);
      // Native (lo/hi), not the displayed range shown in the readout above it.
      expect(onSendBandTo3D).toHaveBeenCalledWith(10, 200);
    });

    it('does not offer "View band in 3D" for a texture-projected fit', () => {
      useToolStore.setState({ tool: 'threshold' });
      render(
        <Toolbar
          samplerFit={{
            mode: 'projected',
            collapsed: false,
            lo: 10,
            hi: 200,
            displayLo: 10,
            displayHi: 200,
            dice: 0.9,
            skill: 0.8,
            coverage: 0.5,
            extraSigma: 0,
            appliedBlur: 0,
            intensitySkill: 0.4,
            weights: [],
          } as any}
          onSendBandTo3D={vi.fn()}
        />,
      );
      expect(screen.queryByRole('button', { name: /view band in 3d/i })).not.toBeInTheDocument();
    });
  });

  describe('select panel', () => {
    it('shows the select-scope radiogroup for the select tool', () => {
      useToolStore.setState({ tool: 'select' });
      render(<Toolbar />);
      expect(screen.getByRole('radiogroup', { name: 'Select scope' })).toBeInTheDocument();
    });

    it('toggles select scope between class and all', async () => {
      useToolStore.setState({ tool: 'select', selectScope: 'all' });
      const user = userEvent.setup();
      render(<Toolbar />);
      const classRadio = screen.getByRole('radio', { name: 'Select this class' });
      expect(classRadio).not.toBeChecked();
      await user.click(classRadio);
      expect(useToolStore.getState().selectScope).toBe('class');
    });
  });

  describe('fill panel', () => {
    it('shows the fill threshold slider for the fill tool', () => {
      useToolStore.setState({ tool: 'fill', fillThreshold: 0.1 });
      render(<Toolbar />);
      expect(screen.getByText(/Fill threshold/)).toBeInTheDocument();
      expect(screen.getByText(/10%/)).toBeInTheDocument();
    });
  });

  describe('magic panel', () => {
    it('shows the SAM engine controls by default', () => {
      useToolStore.setState({ tool: 'magic', magicEngine: 'sam' });
      render(<Toolbar />);
      expect(screen.getByRole('button', { name: 'Smart (AI)' })).toBeInTheDocument();
      expect(screen.getByText('Detail', { selector: 'label' })).toBeInTheDocument();
      expect(screen.getByText('Avoid other-class regions')).toBeInTheDocument();
    });

    it('switches to the classic engine and shows its controls', async () => {
      useToolStore.setState({ tool: 'magic', magicEngine: 'sam' });
      const user = userEvent.setup();
      render(<Toolbar />);
      await user.click(screen.getByRole('button', { name: 'Classic' }));
      expect(useToolStore.getState().magicEngine).toBe('classic');
      expect(await screen.findByRole('button', { name: 'Connected' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'All similar' })).toBeInTheDocument();
    });

    it('shows the edge-stop slider only in contiguous classic mode', () => {
      useToolStore.setState({ tool: 'magic', magicEngine: 'classic', magicMode: 'contiguous' });
      render(<Toolbar />);
      expect(screen.getByRole('slider', { name: 'Edge stop' })).toBeInTheDocument();
    });

    it('hides the edge-stop slider in global classic mode', () => {
      useToolStore.setState({ tool: 'magic', magicEngine: 'classic', magicMode: 'global' });
      render(<Toolbar />);
      expect(screen.queryByRole('slider', { name: 'Edge stop' })).not.toBeInTheDocument();
    });

    it('selecting a SAM detail level updates the store', async () => {
      useToolStore.setState({ tool: 'magic', magicEngine: 'sam', samDetail: 'auto' });
      const user = userEvent.setup();
      render(<Toolbar />);
      await user.click(screen.getByRole('button', { name: 'fine' }));
      expect(useToolStore.getState().samDetail).toBe('fine');
    });

    it('toggles the avoid-labeled and connected-only SAM checkboxes', async () => {
      useToolStore.setState({
        tool: 'magic', magicEngine: 'sam', samAvoidLabeled: true, samConnectedOnly: true,
      });
      const user = userEvent.setup();
      render(<Toolbar />);
      await user.click(screen.getByRole('checkbox', { name: /Avoid other-class regions/ }));
      expect(useToolStore.getState().samAvoidLabeled).toBe(false);
      await user.click(screen.getByRole('checkbox', { name: /Connected regions only/ }));
      expect(useToolStore.getState().samConnectedOnly).toBe(false);
    });
  });

  describe('global toggles', () => {
    it('toggles "Clip to other classes"', async () => {
      useToolStore.setState({ clipToOtherClasses: true });
      const user = userEvent.setup();
      render(<Toolbar />);
      await user.click(screen.getByRole('checkbox', { name: 'Clip to other classes' }));
      expect(useToolStore.getState().clipToOtherClasses).toBe(false);
    });

    it('toggles "Merge overlapping same class"', async () => {
      useToolStore.setState({ mergeOverlappingSameClass: false });
      const user = userEvent.setup();
      render(<Toolbar />);
      await user.click(screen.getByRole('checkbox', { name: 'Merge overlapping same class' }));
      expect(useToolStore.getState().mergeOverlappingSameClass).toBe(true);
    });
  });
});
