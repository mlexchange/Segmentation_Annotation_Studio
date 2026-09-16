import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VersionPreviewBar from './VersionPreviewBar';
import type { VersionMeta } from '@/hooks/useSave';

afterEach(() => {
  cleanup();
});

function makeVersions(): VersionMeta[] {
  return [
    { version: 1, saved_at: '2024-01-01T00:00:00Z', shape_count: 1, class_count: 1 },
    { version: 2, saved_at: '2024-01-02T00:00:00Z', shape_count: 3, class_count: 2 },
    { version: 3, saved_at: '2024-01-03T00:00:00Z', shape_count: 5, class_count: 2 },
  ];
}

describe('VersionPreviewBar', () => {
  it('renders nothing when there are no versions', () => {
    const { container } = render(
      <VersionPreviewBar versions={[]} current={1} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('shows the previewed version number, shape count, and (latest) tag on the max version', () => {
    render(
      <VersionPreviewBar versions={makeVersions()} current={3} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.getByText(/Previewing v3/)).toBeInTheDocument();
    expect(screen.getByText('(latest)')).toBeInTheDocument();
    expect(screen.getByText('5 shapes')).toBeInTheDocument();
  });

  it('omits the (latest) tag and uses singular "shape" for a single-shape non-latest version', () => {
    render(
      <VersionPreviewBar versions={makeVersions()} current={1} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.queryByText('(latest)')).not.toBeInTheDocument();
    expect(screen.getByText('1 shape')).toBeInTheDocument();
  });

  it('sorts versions oldest-first regardless of input order for min/max labels', () => {
    const shuffled = [makeVersions()[2], makeVersions()[0], makeVersions()[1]];
    render(
      <VersionPreviewBar versions={shuffled} current={2} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.getByText('v1')).toBeInTheDocument();
    expect(screen.getByText('v3')).toBeInTheDocument();
  });

  it('falls back to the latest version meta when current does not match any version', () => {
    render(
      <VersionPreviewBar versions={makeVersions()} current={99} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    // meta falls back to sorted[last] (v3), so the label reads v3, not the unmatched `current`.
    // isLatest compares `current` (99) to max (3), so the "(latest)" tag is NOT shown
    // even though the displayed meta is the latest version's — a slight label mismatch.
    expect(screen.getByText(/Previewing v3/)).toBeInTheDocument();
    expect(screen.queryByText('(latest)')).not.toBeInTheDocument();
    expect(screen.getByText('5 shapes')).toBeInTheDocument();
  });

  it('shows the loading indicator only when loading is true', () => {
    const { rerender } = render(
      <VersionPreviewBar versions={makeVersions()} current={2} loading onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.getByText('loading…')).toBeInTheDocument();
    rerender(
      <VersionPreviewBar versions={makeVersions()} current={2} loading={false} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.queryByText('loading…')).not.toBeInTheDocument();
  });

  it('calls onRestore with the current version when Restore is clicked', async () => {
    const onRestore = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionPreviewBar versions={makeVersions()} current={2} onChange={vi.fn()} onExit={vi.fn()} onRestore={onRestore} />,
    );
    await user.click(screen.getByRole('button', { name: /Restore this version/ }));
    expect(onRestore).toHaveBeenCalledWith(2);
  });

  it('calls onExit when Exit is clicked', async () => {
    const onExit = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionPreviewBar versions={makeVersions()} current={2} onChange={vi.fn()} onExit={onExit} onRestore={vi.fn()} />,
    );
    await user.click(screen.getByRole('button', { name: 'Exit' }));
    expect(onExit).toHaveBeenCalledOnce();
  });

  it('steps to the older version and disables the older button at the min', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionPreviewBar versions={makeVersions()} current={2} onChange={onChange} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    await user.click(screen.getByRole('button', { name: 'Older version' }));
    expect(onChange).toHaveBeenCalledWith(1);

    render(
      <VersionPreviewBar versions={makeVersions()} current={1} onChange={onChange} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.getAllByRole('button', { name: 'Older version' })[1]).toBeDisabled();
  });

  it('steps to the newer version and disables the newer button at the max', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionPreviewBar versions={makeVersions()} current={2} onChange={onChange} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    await user.click(screen.getByRole('button', { name: 'Newer version' }));
    expect(onChange).toHaveBeenCalledWith(3);

    render(
      <VersionPreviewBar versions={makeVersions()} current={3} onChange={onChange} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    expect(screen.getAllByRole('button', { name: 'Newer version' })[1]).toBeDisabled();
  });

  it('does not step past the range even if step is somehow invoked at the boundary', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionPreviewBar versions={makeVersions()} current={1} onChange={onChange} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    const olderBtn = screen.getByRole('button', { name: 'Older version' });
    expect(olderBtn).toBeDisabled();
    await user.click(olderBtn); // disabled — should not fire
    expect(onChange).not.toHaveBeenCalled();
  });

  it('renders a formatted date from the ISO timestamp', () => {
    render(
      <VersionPreviewBar versions={makeVersions()} current={2} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
    );
    const expected = new Date('2024-01-02T00:00:00Z').toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it('falls back to the raw string when the date cannot be formatted', () => {
    const versions = [{ version: 1, saved_at: 'not-a-date-###', shape_count: 0, class_count: 0 }];
    const original = Date.prototype.toLocaleString;
    // Force toLocaleString to throw to exercise the catch path.
    Date.prototype.toLocaleString = () => { throw new Error('boom'); };
    try {
      render(
        <VersionPreviewBar versions={versions} current={1} onChange={vi.fn()} onExit={vi.fn()} onRestore={vi.fn()} />,
      );
      expect(screen.getByText('not-a-date-###')).toBeInTheDocument();
    } finally {
      Date.prototype.toLocaleString = original;
    }
  });
});
