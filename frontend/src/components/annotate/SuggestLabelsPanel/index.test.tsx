/**
 * SuggestLabelsPanel — fully prop-driven, same style as PixelClassifierPanel's
 * own test file. Extracted from PixelClassifierPanel's "suggest labels
 * (manifold)" tests when the section was pulled out into its own component.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SuggestLabelsPanel, { type SuggestLabelsPanelProps } from './index';
import type { ManifoldParams } from '@/hooks/useFeatureManifold';

afterEach(() => {
  cleanup();
});

const BASE_MANIFOLD_PARAMS: ManifoldParams = { k: 24, boxSize: 64 } as ManifoldParams;

function baseProps(overrides: Partial<SuggestLabelsPanelProps> = {}): SuggestLabelsPanelProps {
  return {
    hasFeatureJob: true,
    manifoldParams: BASE_MANIFOLD_PARAMS,
    onManifoldParamsChange: vi.fn(),
    manifoldSampling: false,
    manifoldHasSample: false,
    manifoldShowHeatmap: false,
    onManifoldShowHeatmapChange: vi.fn(),
    manifoldShowMarkers: false,
    onManifoldShowMarkersChange: vi.fn(),
    manifoldHeatmapOpacity: 0.5,
    onManifoldHeatmapOpacityChange: vi.fn(),
    manifoldMeta: null,
    manifoldError: null,
    onManifoldSample: vi.fn(),
    onManifoldDismiss: vi.fn(),
    manifoldRoiShapeCount: 0,
    canCaptureManifoldRoi: false,
    onCaptureManifoldRoi: vi.fn(),
    onClearManifoldRoi: vi.fn(),
    ...overrides,
  };
}

describe('SuggestLabelsPanel', () => {
  it('is open by default (not buried like when it lived inside PixelClassifierPanel)', () => {
    render(<SuggestLabelsPanel {...baseProps()} />);
    expect(screen.getByText(/suggest regions to label/i)).toBeInTheDocument();
  });

  it('is disabled (non-clickable) when there is no feature job', async () => {
    const user = userEvent.setup();
    render(<SuggestLabelsPanel {...baseProps({ hasFeatureJob: false })} />);
    const header = screen.getByRole('button', { name: /suggest labels/i });
    expect(header).toBeDisabled();
    await user.click(header);
    // Still expanded by default; the header itself being disabled is what matters.
    expect(screen.getByRole('button', { name: /suggest regions to label/i })).toBeDisabled();
  });

  it('calls onManifoldSample and shows a sampling busy bar', async () => {
    const user = userEvent.setup();
    const onManifoldSample = vi.fn();
    const { rerender } = render(<SuggestLabelsPanel {...baseProps({ onManifoldSample })} />);
    const sampleBtn = screen.getByRole('button', { name: /suggest regions to label/i });
    await user.click(sampleBtn);
    expect(onManifoldSample).toHaveBeenCalledTimes(1);

    rerender(<SuggestLabelsPanel {...baseProps({ onManifoldSample, manifoldSampling: true })} />);
    expect(screen.getByText(/sampling manifold coverage…/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sampling…/i })).toBeDisabled();
  });

  it('shows manifold meta summary once sampled', () => {
    render(
      <SuggestLabelsPanel
        {...baseProps({
          manifoldMeta: { nPicked: 12, nSubsample: 5000, explainedVariance: 0.87 },
        })}
      />,
    );
    expect(screen.getByText(/12 boxes from 5,000 px · 87% variance explained/)).toBeInTheDocument();
  });

  it('shows heatmap/marker toggles and opacity slider once a sample exists, and dismiss clears it', async () => {
    const user = userEvent.setup();
    const onManifoldShowHeatmapChange = vi.fn();
    const onManifoldShowMarkersChange = vi.fn();
    const onManifoldDismiss = vi.fn();
    render(
      <SuggestLabelsPanel
        {...baseProps({
          manifoldHasSample: true,
          onManifoldShowHeatmapChange,
          onManifoldShowMarkersChange,
          onManifoldDismiss,
        })}
      />,
    );
    await user.click(screen.getByRole('checkbox', { name: /show coverage heatmap/i }));
    expect(onManifoldShowHeatmapChange).toHaveBeenCalledWith(true);
    await user.click(screen.getByRole('checkbox', { name: /show suggested boxes/i }));
    expect(onManifoldShowMarkersChange).toHaveBeenCalledWith(true);
    await user.click(screen.getByRole('button', { name: /dismiss suggestions/i }));
    expect(onManifoldDismiss).toHaveBeenCalledTimes(1);
  });

  it('restrict-to-selection button is disabled without a capturable ROI, and enabled+clearable with one', async () => {
    const user = userEvent.setup();
    const onCaptureManifoldRoi = vi.fn();
    const onClearManifoldRoi = vi.fn();
    const { rerender } = render(
      <SuggestLabelsPanel {...baseProps({ canCaptureManifoldRoi: false, manifoldRoiShapeCount: 0 })} />,
    );
    expect(screen.getByRole('button', { name: /restrict to selection \(0\)/i })).toBeDisabled();
    expect(screen.queryByRole('button', { name: /^clear$/i })).not.toBeInTheDocument();

    rerender(
      <SuggestLabelsPanel
        {...baseProps({
          canCaptureManifoldRoi: true,
          manifoldRoiShapeCount: 2,
          onCaptureManifoldRoi,
          onClearManifoldRoi,
        })}
      />,
    );
    const restrictBtn = screen.getByRole('button', { name: /restrict to selection \(2\)/i });
    expect(restrictBtn).toBeEnabled();
    await user.click(restrictBtn);
    expect(onCaptureManifoldRoi).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole('button', { name: /^clear$/i }));
    expect(onClearManifoldRoi).toHaveBeenCalledTimes(1);
  });

  it('shows a manifold error message', () => {
    render(<SuggestLabelsPanel {...baseProps({ manifoldError: 'Sampling failed.' })} />);
    expect(screen.getByText('Sampling failed.')).toBeInTheDocument();
  });

  it('updates K boxes / box size params', () => {
    const onManifoldParamsChange = vi.fn();
    render(<SuggestLabelsPanel {...baseProps({ onManifoldParamsChange })} />);
    // A fully-controlled numeric input whose value prop never changes across
    // this render — user.type() would accumulate keystrokes against jsdom's
    // own uncommitted DOM value instead. A single fireEvent.change avoids that.
    fireEvent.change(screen.getByLabelText(/k boxes/i), { target: { value: '50' } });
    expect(onManifoldParamsChange).toHaveBeenLastCalledWith({ k: 50, boxSize: 64 });
  });
});
