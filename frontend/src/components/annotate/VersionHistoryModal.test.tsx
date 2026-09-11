import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VersionHistoryModal from './VersionHistoryModal';
import type { VersionMeta } from '@/hooks/useSave';

const versions: VersionMeta[] = [
  { version: 1, saved_at: '2024-01-01T12:00:00Z', shape_count: 1, class_count: 1, annotated_by: null, notes: null },
  {
    version: 2,
    saved_at: '2024-01-02T12:00:00Z',
    shape_count: 5,
    class_count: 2,
    annotated_by: 'Ada',
    notes: 'second pass',
  },
];

afterEach(() => {
  cleanup();
});

describe('VersionHistoryModal', () => {
  it('shows an empty-state message with no versions', () => {
    render(
      <VersionHistoryModal versions={[]} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.getByText(/No saved versions yet/)).toBeInTheDocument();
  });

  it('renders versions newest-first with shape/class counts and pluralization', () => {
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={vi.fn()} onClose={vi.fn()} />
    );
    const rows = screen.getAllByText(/^v\d/);
    expect(rows[0]).toHaveTextContent('v2');
    expect(rows[1]).toHaveTextContent('v1');
    expect(screen.getByText(/5 shapes/)).toBeInTheDocument();
    expect(screen.getByText(/2 classes/)).toBeInTheDocument();
    expect(screen.getByText(/1 shape\b/)).toBeInTheDocument();
    expect(screen.getByText(/1 class\b/)).toBeInTheDocument();
  });

  it('shows annotator and notes only when present', () => {
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.getByText(/by Ada/)).toBeInTheDocument();
    expect(screen.getByText('second pass')).toBeInTheDocument();
  });

  it('does not render thumbnails when sourceKey is null', () => {
    render(
      <VersionHistoryModal versions={versions} sourceKey={null} onPreview={vi.fn()} onRestore={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.queryByAltText(/thumbnail/)).not.toBeInTheDocument();
  });

  it('renders a thumbnail image per version when sourceKey is set', () => {
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.getByAltText('Version 1 thumbnail')).toBeInTheDocument();
    expect(screen.getByAltText('Version 2 thumbnail')).toBeInTheDocument();
  });

  it('falls back to a placeholder icon when a thumbnail fails to load', () => {
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={vi.fn()} onClose={vi.fn()} />
    );
    const img = screen.getByAltText('Version 2 thumbnail');
    fireEvent.error(img);
    expect(screen.queryByAltText('Version 2 thumbnail')).not.toBeInTheDocument();
  });

  it('calls onPreview with the version number and does not close the modal', async () => {
    const onPreview = vi.fn();
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={onPreview} onRestore={vi.fn()} onClose={onClose} />
    );
    await user.click(screen.getAllByRole('button', { name: /Preview/ })[0]);
    expect(onPreview).toHaveBeenCalledWith(2);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('calls onRestore with the version number and then onClose', async () => {
    const onRestore = vi.fn();
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={onRestore} onClose={onClose} />
    );
    await user.click(screen.getAllByRole('button', { name: /Restore/ })[0]);
    expect(onRestore).toHaveBeenCalledWith(2);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onClose from the header close button', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <VersionHistoryModal versions={versions} sourceKey="local:sample.tif" onPreview={vi.fn()} onRestore={vi.fn()} onClose={onClose} />
    );
    const headerButtons = screen.getAllByRole('button').filter((b) => !/Preview|Restore/.test(b.textContent ?? ''));
    await user.click(headerButtons[0]);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
