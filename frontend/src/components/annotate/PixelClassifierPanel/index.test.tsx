/**
 * PixelClassifierPanel — fully prop-driven (no internal store/hook access), so
 * tests construct prop objects directly rather than mocking usePixelClassifier
 * or @/lib/ipredApi. Covers: idle/canTrain gating, busy states during train/
 * multi-slice train/predict, model summary + feature importances, class-
 * probability cycling, commit/dismiss, volume-apply flow (apply -> progress ->
 * result -> commit/dismiss), predicted-pointer "make editable" banner, push-to-
 * Tiled + view-in-3D, train-deep-model hand-off, and error surfacing.
 * (Suggest-labels/manifold sampling was pulled out into its own
 * SuggestLabelsPanel — see that component's own test file.)
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PixelClassifierPanel, { type PixelClassifierPanelProps } from './index';
import type { ClfTrainResult, ClfPredictCounts } from '@/hooks/usePixelClassifier';

afterEach(() => {
  cleanup();
});

const BASE_PARAMS = { iterations: 200, depth: 6, learningRate: 0.1, alpha: 0.05 };

function makeModel(overrides: Partial<ClfTrainResult> = {}): ClfTrainResult {
  return {
    modelId: 'model-1',
    featureId: 'feat-1',
    nSamples: 100,
    nTrain: 80,
    nCal: 20,
    classIds: [1, 2],
    trainAccuracy: 0.912,
    params: BASE_PARAMS,
    nTrees: 200,
    usesSam: false,
    featureImportances: [
      { label: 'intensity', importance: 5.5 },
      { label: 'edges', importance: 2.1 },
    ],
    trainerId: 'catboost',
    compositionId: 'comp-1',
    ...overrides,
  };
}

function makeCounts(overrides: Partial<ClfPredictCounts> = {}): ClfPredictCounts {
  return { singleton: 700, multi: 200, abstain: 100, ...overrides };
}

function baseProps(overrides: Partial<PixelClassifierPanelProps> = {}): PixelClassifierPanelProps {
  return {
    hasFeatureJob: true,
    canAutoPreprocess: false,
    hasShapes: true,
    training: false,
    predicting: false,
    params: BASE_PARAMS,
    onParamsChange: vi.fn(),
    model: null,
    hasPrediction: false,
    predictCounts: null,
    error: null,
    onTrain: vi.fn(),
    onPredict: vi.fn(),
    onCommit: vi.fn(),
    onDismiss: vi.fn(),
    annotatedSliceCount: 1,
    trainAcrossSlices: false,
    onTrainAcrossSlicesChange: vi.fn(),
    multiTraining: false,
    multiTrainProgress: null,
    totalSliceCount: 1,
    commitClassIds: [],
    onToggleCommitClassId: vi.fn(),
    volumeApplying: false,
    volumeApplyProgress: null,
    volumeApplyResult: null,
    onApplyToVolume: vi.fn(),
    onCommitVolumeApply: vi.fn(),
    onCancelVolumeApply: vi.fn(),
    onDismissVolumeApply: vi.fn(),
    hasPredictedPointerOnCurrentSlice: false,
    vectorizingSlice: false,
    onMakeSliceEditable: vi.fn(),
    hasAnyPredictedPointers: false,
    ...overrides,
  };
}

describe('PixelClassifierPanel', () => {
  describe('idle / train gating', () => {
    it('enables Train when features exist and shapes are annotated', () => {
      render(<PixelClassifierPanel {...baseProps()} />);
      expect(screen.getByRole('button', { name: /train classifier/i })).toBeEnabled();
      // Predict is disabled with no model yet.
      expect(screen.getByRole('button', { name: /^predict$/i })).toBeDisabled();
    });

    it('disables Train when there are no shapes and no auto-preprocess', () => {
      render(<PixelClassifierPanel {...baseProps({ hasShapes: false, hasFeatureJob: false })} />);
      expect(screen.getByRole('button', { name: /train classifier/i })).toBeDisabled();
    });

    it('allows training with no feature job when canAutoPreprocess is set, and notes it will compute features', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({ hasFeatureJob: false, canAutoPreprocess: true, hasShapes: true })}
        />,
      );
      expect(screen.getByRole('button', { name: /train classifier/i })).toBeEnabled();
      expect(screen.getByText(/Train will compute features/i)).toBeInTheDocument();
    });

    it('calls onTrain when the train button is clicked', async () => {
      const user = userEvent.setup();
      const onTrain = vi.fn();
      render(<PixelClassifierPanel {...baseProps({ onTrain })} />);
      await user.click(screen.getByRole('button', { name: /train classifier/i }));
      expect(onTrain).toHaveBeenCalledTimes(1);
    });

    it('shows a training busy bar while training', () => {
      render(<PixelClassifierPanel {...baseProps({ training: true })} />);
      expect(screen.getByText(/Training…/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /train classifier/i })).toBeDisabled();
    });

    it('updates iteration/depth/learning-rate params via onParamsChange', () => {
      const onParamsChange = vi.fn();
      render(<PixelClassifierPanel {...baseProps({ onParamsChange })} />);
      const trees = screen.getByLabelText(/trees/i);
      fireEvent.change(trees, { target: { value: '300' } });
      expect(onParamsChange).toHaveBeenCalledTimes(1);
      expect(onParamsChange.mock.calls[0][0].iterations).toBe(300);
    });
  });

  describe('train across slices', () => {
    it('disables the checkbox when only one slice is annotated', () => {
      render(<PixelClassifierPanel {...baseProps({ annotatedSliceCount: 1 })} />);
      expect(screen.getByRole('checkbox', { name: /train across all annotated slices/i })).toBeDisabled();
    });

    it('enables the checkbox with multiple annotated slices and toggles it', async () => {
      const user = userEvent.setup();
      const onTrainAcrossSlicesChange = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({ annotatedSliceCount: 3, onTrainAcrossSlicesChange })}
        />,
      );
      const checkbox = screen.getByRole('checkbox', { name: /train across all annotated slices \(3\)/i });
      expect(checkbox).toBeEnabled();
      await user.click(checkbox);
      expect(onTrainAcrossSlicesChange).toHaveBeenCalledWith(true);
    });

    it('label switches to "Train across slices" and gates on annotatedSliceCount only when trainAcrossSlices is set', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            annotatedSliceCount: 2,
            trainAcrossSlices: true,
            hasShapes: false,
            hasFeatureJob: false,
          })}
        />,
      );
      expect(screen.getByRole('button', { name: /train across slices/i })).toBeEnabled();
    });

    it('shows multi-train job progress bar with done/total', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            multiTraining: true,
            multiTrainProgress: { done: 2, total: 5 },
          })}
        />,
      );
      expect(screen.getByText(/Training across slices…/i)).toBeInTheDocument();
      expect(screen.getByText('2/5')).toBeInTheDocument();
    });
  });

  describe('predict', () => {
    it('enables Predict once a model exists and calls onPredict', async () => {
      const user = userEvent.setup();
      const onPredict = vi.fn();
      render(<PixelClassifierPanel {...baseProps({ model: makeModel(), onPredict })} />);
      const btn = screen.getByRole('button', { name: /^predict$/i });
      expect(btn).toBeEnabled();
      await user.click(btn);
      expect(onPredict).toHaveBeenCalledTimes(1);
    });

    it('shows a predicting busy bar', () => {
      render(<PixelClassifierPanel {...baseProps({ model: makeModel(), predicting: true })} />);
      expect(screen.getByText(/Predicting…/i)).toBeInTheDocument();
    });
  });

  describe('model summary', () => {
    it('renders accuracy, sample counts, trees, classes, and feature importances', () => {
      render(<PixelClassifierPanel {...baseProps({ model: makeModel() })} />);
      expect(screen.getByText(/Acc 91\.2%/)).toBeInTheDocument();
      expect(screen.getByText(/train 80 \/ cal 20/)).toBeInTheDocument();
      expect(screen.getByText(/200 trees/)).toBeInTheDocument();
      expect(screen.getByText(/classes 1, 2/)).toBeInTheDocument();
      expect(screen.getByText('intensity')).toBeInTheDocument();
      expect(screen.getByText('edges')).toBeInTheDocument();
    });

    it('shows the SlimSAM suffix when usesSam is true', () => {
      render(<PixelClassifierPanel {...baseProps({ model: makeModel({ usesSam: true }) })} />);
      expect(screen.getByText(/\+SlimSAM/)).toBeInTheDocument();
    });
  });

  describe('commit classes + apply across volume', () => {
    it('toggles commit-class chips via onToggleCommitClassId', async () => {
      const user = userEvent.setup();
      const onToggleCommitClassId = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({ model: makeModel(), onToggleCommitClassId, commitClassIds: [1] })}
        />,
      );
      await user.click(screen.getByText('class 1'));
      expect(onToggleCommitClassId).toHaveBeenCalledWith(1);
      await user.click(screen.getByText('class 2'));
      expect(onToggleCommitClassId).toHaveBeenCalledWith(2);
    });

    it('disables Apply across volume with a single slice or no classes selected', () => {
      const { rerender } = render(
        <PixelClassifierPanel
          {...baseProps({ model: makeModel(), totalSliceCount: 1, commitClassIds: [1] })}
        />,
      );
      expect(screen.getByRole('button', { name: /apply across volume/i })).toBeDisabled();

      rerender(
        <PixelClassifierPanel
          {...baseProps({ model: makeModel(), totalSliceCount: 5, commitClassIds: [] })}
        />,
      );
      expect(screen.getByRole('button', { name: /apply across volume/i })).toBeDisabled();
    });

    it('enables Apply across volume with multiple slices and at least one selected class, and calls onApplyToVolume', async () => {
      const user = userEvent.setup();
      const onApplyToVolume = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            totalSliceCount: 10,
            commitClassIds: [1],
            onApplyToVolume,
          })}
        />,
      );
      const btn = screen.getByRole('button', { name: /apply across volume \(10 slices\)/i });
      expect(btn).toBeEnabled();
      await user.click(btn);
      expect(onApplyToVolume).toHaveBeenCalledTimes(1);
    });

    it('shows volume-apply progress and a cancel button while running', async () => {
      const user = userEvent.setup();
      const onCancelVolumeApply = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            volumeApplying: true,
            volumeApplyProgress: { done: 3, total: 10 },
            onCancelVolumeApply,
          })}
        />,
      );
      expect(screen.getByText(/Applying across volume…/i)).toBeInTheDocument();
      expect(screen.getByText('3/10')).toBeInTheDocument();
      await user.click(screen.getByRole('button', { name: /cancel/i }));
      expect(onCancelVolumeApply).toHaveBeenCalledTimes(1);
    });

    it('shows the volume-apply result with commit/dismiss actions', async () => {
      const user = userEvent.setup();
      const onCommitVolumeApply = vi.fn();
      const onDismissVolumeApply = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            commitClassIds: [1, 2],
            volumeApplyResult: { runCount: 8, errorCount: 2, cancelled: false },
            onCommitVolumeApply,
            onDismissVolumeApply,
          })}
        />,
      );
      expect(screen.getByText(/Predicted 8 slice\(s\), 2 failed\./)).toBeInTheDocument();
      const commitBtn = screen.getByRole('button', { name: /Commit predicted shapes \(2 classes\)/i });
      expect(commitBtn).toBeEnabled();
      await user.click(commitBtn);
      expect(onCommitVolumeApply).toHaveBeenCalledTimes(1);
      await user.click(screen.getByRole('button', { name: /^dismiss$/i }));
      expect(onDismissVolumeApply).toHaveBeenCalledTimes(1);
    });

    it('disables commit-predicted-shapes when runCount is 0, and shows cancelled prefix', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            commitClassIds: [1],
            volumeApplyResult: { runCount: 0, errorCount: 0, cancelled: true },
          })}
        />,
      );
      expect(screen.getByText(/^Cancelled — Predicted 0 slice\(s\)\./)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Commit predicted shapes \(1 class\)/i })).toBeDisabled();
    });
  });

  describe('predicted pointer / make editable', () => {
    it('shows the "make this slice editable" banner and calls onMakeSliceEditable', async () => {
      const user = userEvent.setup();
      const onMakeSliceEditable = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPredictedPointerOnCurrentSlice: true,
            onMakeSliceEditable,
          })}
        />,
      );
      const btn = screen.getByRole('button', { name: /make this slice editable/i });
      await user.click(btn);
      expect(onMakeSliceEditable).toHaveBeenCalledTimes(1);
    });

    it('shows "Vectorizing…" and disables the button while vectorizingSlice is true', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPredictedPointerOnCurrentSlice: true,
            vectorizingSlice: true,
          })}
        />,
      );
      const btn = screen.getByRole('button', { name: /vectorizing…/i });
      expect(btn).toBeDisabled();
    });
  });

  describe('push to Tiled / view in 3D / train deep model', () => {
    it('does not render the sync row without onSyncToTiled', () => {
      render(<PixelClassifierPanel {...baseProps({ model: makeModel(), annotatedSliceCount: 1 })} />);
      expect(screen.queryByRole('button', { name: /push to tiled/i })).not.toBeInTheDocument();
    });

    it('renders push-to-Tiled and view-in-3D and calls their handlers', async () => {
      const user = userEvent.setup();
      const onSyncToTiled = vi.fn();
      const onViewIn3D = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            annotatedSliceCount: 1,
            onSyncToTiled,
            onViewIn3D,
          })}
        />,
      );
      await user.click(screen.getByRole('button', { name: /push to tiled/i }));
      expect(onSyncToTiled).toHaveBeenCalledTimes(1);
      await user.click(screen.getByRole('button', { name: /view in 3d/i }));
      expect(onViewIn3D).toHaveBeenCalledTimes(1);
    });

    it('shows the syncing state, success message, and error message', () => {
      const { rerender } = render(
        <PixelClassifierPanel
          {...baseProps({ model: makeModel(), annotatedSliceCount: 1, onSyncToTiled: vi.fn(), syncingToTiled: true })}
        />,
      );
      expect(screen.getByRole('button', { name: /pushing…/i })).toBeDisabled();

      rerender(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            annotatedSliceCount: 1,
            onSyncToTiled: vi.fn(),
            syncedToTiled: true,
          })}
        />,
      );
      expect(screen.getByText(/pushed to tiled/i)).toBeInTheDocument();

      rerender(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            annotatedSliceCount: 1,
            onSyncToTiled: vi.fn(),
            syncToTiledError: 'network failed',
          })}
        />,
      );
      expect(screen.getByText('network failed')).toBeInTheDocument();
    });

    it('renders the sync row when there are only predicted pointers (no real shapes yet)', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            annotatedSliceCount: 0,
            hasAnyPredictedPointers: true,
            onSyncToTiled: vi.fn(),
          })}
        />,
      );
      expect(screen.getByRole('button', { name: /push to tiled/i })).toBeInTheDocument();
    });

    it('renders "Train a deep model on this" and calls onTrainDeepModel', async () => {
      const user = userEvent.setup();
      const onTrainDeepModel = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({ model: makeModel(), annotatedSliceCount: 2, onTrainDeepModel })}
        />,
      );
      await user.click(screen.getByRole('button', { name: /train a deep model on this/i }));
      expect(onTrainDeepModel).toHaveBeenCalledTimes(1);
    });
  });

  describe('prediction results: coverage headline, class probability, commit/dismiss', () => {
    it('shows a "good" coverage headline when confident percentage is high', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPrediction: true,
            predictCounts: makeCounts({ singleton: 900, multi: 50, abstain: 50 }),
          })}
        />,
      );
      expect(screen.getByText(/90% of pixels are confidently labeled at 95% coverage\./)).toBeInTheDocument();
    });

    it('shows a "warn" coverage headline when confident percentage is low', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPrediction: true,
            predictCounts: makeCounts({ singleton: 100, multi: 400, abstain: 500 }),
          })}
        />,
      );
      expect(screen.getByText(/Only 10% confidently labeled/)).toBeInTheDocument();
    });

    it('renders the class-probability cycler and calls onCycleProbaClass / onProbaThresholdChange', async () => {
      const user = userEvent.setup();
      const onCycleProbaClass = vi.fn();
      const onProbaThresholdChange = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPrediction: true,
            predictCounts: makeCounts(),
            activeProbaClassId: 1,
            activeProbaThreshold: 0.5,
            classLabelForId: (id) => (id === 1 ? 'Cell' : 'Background'),
            onCycleProbaClass,
            onProbaThresholdChange,
          })}
        />,
      );
      expect(screen.getByTitle('Cell')).toBeInTheDocument();
      await user.click(screen.getByRole('button', { name: /next class/i }));
      expect(onCycleProbaClass).toHaveBeenCalledWith(1);
      await user.click(screen.getByRole('button', { name: /previous class/i }));
      expect(onCycleProbaClass).toHaveBeenCalledWith(-1);
      expect(onProbaThresholdChange).not.toHaveBeenCalled();
    });

    it('does not render the class-probability cycler without the cycle/threshold callbacks', () => {
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPrediction: true,
            predictCounts: makeCounts(),
          })}
        />,
      );
      expect(screen.queryByText(/class probability/i)).not.toBeInTheDocument();
    });

    it('renders commit and dismiss buttons, using a custom commitLabel, and calls their handlers', async () => {
      const user = userEvent.setup();
      const onCommit = vi.fn();
      const onDismiss = vi.fn();
      render(
        <PixelClassifierPanel
          {...baseProps({
            model: makeModel(),
            hasPrediction: true,
            predictCounts: makeCounts(),
            commitLabel: 'Commit my classes',
            onCommit,
            onDismiss,
          })}
        />,
      );
      await user.click(screen.getByRole('button', { name: /commit my classes/i }));
      expect(onCommit).toHaveBeenCalledTimes(1);
      await user.click(screen.getByRole('button', { name: /^dismiss$/i }));
      expect(onDismiss).toHaveBeenCalledTimes(1);
    });
  });

  describe('error state', () => {
    it('renders the top-level error message', () => {
      render(<PixelClassifierPanel {...baseProps({ error: 'Feature bank missing.' })} />);
      expect(screen.getByText('Feature bank missing.')).toBeInTheDocument();
    });
  });
});
