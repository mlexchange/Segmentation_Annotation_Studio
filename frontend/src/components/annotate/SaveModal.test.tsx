import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SaveModal, { type SaveModalProps } from './SaveModal';
import { useSettingsStore } from '@/stores/settingsStore';
import type { SaveDraftPayload } from '@/hooks/useSave';

type OnSaveArg = Parameters<SaveModalProps['onSave']>[0];

const basePayload: SaveDraftPayload = {
  classes: [],
  slices: {},
  split_by_slice: {},
  negative_slices: [],
};

function baseProps(overrides: Partial<Parameters<typeof SaveModal>[0]> = {}) {
  return {
    sourceKey: 'local:sample.tif',
    payload: basePayload,
    shapeCount: 3,
    classCount: 2,
    isSaving: false,
    onSave: vi.fn(async () => {}),
    onClose: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  useSettingsStore.setState({ annotatorName: '' });
  localStorage.clear();
  global.fetch = vi.fn(async () => ({
    ok: true,
    blob: async () => new Blob(['fake-png'], { type: 'image/png' }),
  })) as unknown as typeof fetch;
  // jsdom doesn't implement these — stub them so the preview pipeline doesn't throw.
  global.URL.createObjectURL = vi.fn(() => 'blob:preview-url');
  global.URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('SaveModal', () => {
  it('shows a loading state then the fetched preview thumbnail', async () => {
    render(<SaveModal {...baseProps()} />);
    expect(screen.getByText(/Generating preview/)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByAltText('Annotation preview')).toBeInTheDocument();
    });
    expect(screen.getByAltText('Annotation preview')).toHaveAttribute('src', 'blob:preview-url');
  });

  it('shows preview-unavailable when the fetch fails', async () => {
    global.fetch = vi.fn(async () => ({ ok: false })) as unknown as typeof fetch;
    render(<SaveModal {...baseProps()} />);
    await waitFor(() => {
      expect(screen.getByText('Preview unavailable')).toBeInTheDocument();
    });
  });

  it('displays shape and class counts, singular vs plural', async () => {
    render(<SaveModal {...baseProps({ shapeCount: 1, classCount: 1 })} />);
    await waitFor(() => screen.getByAltText('Annotation preview'));
    expect(screen.getByText('1 shape · 1 class')).toBeInTheDocument();
  });

  it('displays plural shape/class counts', async () => {
    render(<SaveModal {...baseProps({ shapeCount: 3, classCount: 2 })} />);
    await waitFor(() => screen.getByAltText('Annotation preview'));
    expect(screen.getByText('3 shapes · 2 classes')).toBeInTheDocument();
  });

  it('prefills annotator name from the settings store', async () => {
    useSettingsStore.setState({ annotatorName: 'Ada' });
    render(<SaveModal {...baseProps()} />);
    expect(screen.getByLabelText('Who annotated this')).toHaveValue('Ada');
    await waitFor(() => screen.getByAltText('Annotation preview'));
  });

  it('falls back to the legacy localStorage key when the store is empty', async () => {
    localStorage.setItem('sam3_annotator_name', 'Legacy Name');
    render(<SaveModal {...baseProps()} />);
    expect(screen.getByLabelText('Who annotated this')).toHaveValue('Legacy Name');
    await waitFor(() => screen.getByAltText('Annotation preview'));
  });

  it('calls onClose when Cancel is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<SaveModal {...baseProps({ onClose })} />);
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onClose when the X button is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<SaveModal {...baseProps({ onClose })} />);
    const buttons = screen.getAllByRole('button');
    // The X close button is the first button in the header (no accessible name).
    await user.click(buttons[0]);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('submits with trimmed annotator name, notes, and base64 thumbnail; persists name to the store', async () => {
    const onSave = vi.fn(async (_opts: OnSaveArg) => {});
    const user = userEvent.setup();
    render(<SaveModal {...baseProps({ onSave })} />);
    await waitFor(() => screen.getByAltText('Annotation preview'));

    await user.type(screen.getByLabelText('Who annotated this'), '  Ada Lovelace  ');
    await user.type(screen.getByLabelText('Notes'), '  looks good  ');
    await user.click(screen.getByRole('button', { name: /Save version/ }));

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    const arg = onSave.mock.calls[0]![0];
    expect(arg.annotatedBy).toBe('Ada Lovelace');
    expect(arg.notes).toBe('looks good');
    expect(typeof arg.thumbnailBase64).toBe('string');
    expect(useSettingsStore.getState().annotatorName).toBe('Ada Lovelace');
  });

  it('submits without a thumbnail when the preview fetch failed', async () => {
    global.fetch = vi.fn(async () => ({ ok: false })) as unknown as typeof fetch;
    const onSave = vi.fn(async (_opts: OnSaveArg) => {});
    const user = userEvent.setup();
    render(<SaveModal {...baseProps({ onSave })} />);
    await waitFor(() => screen.getByText('Preview unavailable'));
    await user.click(screen.getByRole('button', { name: /Save version/ }));
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    expect(onSave.mock.calls[0]![0].thumbnailBase64).toBeUndefined();
  });

  it('disables Cancel/X/Submit and shows Saving state while isSaving is true', async () => {
    render(<SaveModal {...baseProps({ isSaving: true })} />);
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Saving/ })).toBeDisabled();
    await waitFor(() => screen.getByAltText('Annotation preview'));
  });
});
