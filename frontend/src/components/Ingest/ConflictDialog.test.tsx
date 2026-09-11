import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConflictDialog, { type IngestConflict } from './ConflictDialog';

function makeConflicts(n: number): IngestConflict[] {
  return Array.from({ length: n }, (_, i) => ({ filename: `img${i}.tif`, key: `img${i}` }));
}

function renderDialog(overrides: Partial<React.ComponentProps<typeof ConflictDialog>> = {}) {
  const props = {
    containerPath: 'browse/foo',
    totalFiles: 5,
    conflicts: makeConflicts(2),
    existingCount: 3,
    suggestedContainerPath: 'browse/foo_2',
    onReplace: vi.fn(),
    onSkip: vi.fn(),
    onNewDataset: vi.fn(),
    onBrowseExisting: vi.fn(),
    onCancel: vi.fn(),
    ...overrides,
  };
  const utils = render(<ConflictDialog {...props} />);
  return { ...utils, props };
}

afterEach(() => {
  cleanup();
});

describe('ConflictDialog', () => {
  it('renders the summary sentence with container path and existing count', () => {
    renderDialog();
    expect(
      screen.getByText((_, el) => el?.textContent === '2 of 5 images are already in browse/foo (3 samples there now).')
    ).toBeInTheDocument();
  });

  it('uses singular wording when totalFiles is 1 and conflicts is 1', () => {
    renderDialog({ totalFiles: 1, conflicts: makeConflicts(1), existingCount: 1 });
    expect(
      screen.getByText((_, el) => el?.textContent === '1 of 1 image is already in browse/foo (1 sample there now).')
    ).toBeInTheDocument();
  });

  it('omits the existing-count parenthetical when existingCount is 0', () => {
    renderDialog({ existingCount: 0 });
    expect(
      screen.getByText((_, el) => el?.textContent === '2 of 5 images are already in browse/foo.')
    ).toBeInTheDocument();
  });

  it('lists conflicting filenames, capped at MAX_LISTED=5 with an overflow line', () => {
    renderDialog({ conflicts: makeConflicts(7), totalFiles: 10 });
    for (let i = 0; i < 5; i++) {
      expect(screen.getByText(`img${i}.tif`)).toBeInTheDocument();
    }
    expect(screen.queryByText('img5.tif')).not.toBeInTheDocument();
    expect(screen.getByText('…and 2 more')).toBeInTheDocument();
  });

  it('shows the new-image count on the Skip button when not all files conflict', () => {
    renderDialog({ totalFiles: 5, conflicts: makeConflicts(2) });
    expect(screen.getByText('Ingests only the 3 new images')).toBeInTheDocument();
  });

  it('disables Skip and shows the all-conflict message when every file conflicts', () => {
    renderDialog({ totalFiles: 3, conflicts: makeConflicts(3) });
    const skipBtn = screen.getByRole('button', { name: /Skip the duplicates/ });
    expect(skipBtn).toBeDisabled();
    expect(screen.getByText('Nothing new to add — every image is already here')).toBeInTheDocument();
  });

  it('shows the suggested container path on the "new dataset" option', () => {
    renderDialog({ suggestedContainerPath: 'browse/foo_3' });
    expect(screen.getByText('browse/foo_3')).toBeInTheDocument();
  });

  it('calls onReplace, onSkip, onNewDataset, onBrowseExisting when their buttons are clicked', async () => {
    const user = userEvent.setup();
    const { props } = renderDialog();
    await user.click(screen.getByRole('button', { name: /Replace the existing images/ }));
    expect(props.onReplace).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: /Skip the duplicates/ }));
    expect(props.onSkip).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: /Ingest to a new dataset/ }));
    expect(props.onNewDataset).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: /Browse the existing dataset/ }));
    expect(props.onBrowseExisting).toHaveBeenCalledTimes(1);
  });

  it('calls onCancel from the X button, the Cancel button, and clicking the backdrop', async () => {
    const user = userEvent.setup();
    const { props } = renderDialog();

    await user.click(screen.getByLabelText('Close'));
    expect(props.onCancel).toHaveBeenCalledTimes(1);

    await user.click(screen.getByText('Cancel'));
    expect(props.onCancel).toHaveBeenCalledTimes(2);
  });

  it('does not call onCancel when clicking inside the dialog panel', async () => {
    const user = userEvent.setup();
    const { props } = renderDialog();
    await user.click(screen.getByText('This dataset already has these images'));
    expect(props.onCancel).not.toHaveBeenCalled();
  });

  it('calls onCancel when clicking the backdrop directly', async () => {
    const user = userEvent.setup();
    const { props, container } = renderDialog();
    const backdrop = container.firstChild as HTMLElement;
    await user.click(backdrop);
    expect(props.onCancel).toHaveBeenCalledTimes(1);
  });
});
