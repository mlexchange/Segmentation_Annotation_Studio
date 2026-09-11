import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ClassManager from './index';
import { useClassStore } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useReferenceGuideStore } from '@/stores/referenceGuideStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useDatasetStore } from '@/stores/datasetStore';

beforeEach(() => {
  useClassStore.setState({ classes: [] });
  useAnnotationStore.getState().reset();
  useReferenceGuideStore.getState().clear();
  useSettingsStore.setState({ colorblindMode: false });
  useDatasetStore.setState({ source: null, kind: null, serverUri: null } as any);
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  vi.spyOn(window, 'alert').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('ClassManager', () => {
  it('shows an empty-state message with no classes', () => {
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    expect(screen.getByText(/No classes yet/)).toBeInTheDocument();
  });

  it('shows default quick-add suggestions when there is no guide', () => {
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    expect(screen.getByRole('button', { name: /air/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /substrate/ })).toBeInTheDocument();
  });

  it('quick-add creates and activates a class', async () => {
    const onActivate = vi.fn();
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={onActivate} />);
    await user.click(screen.getByRole('button', { name: 'air' }));
    expect(useClassStore.getState().classes).toHaveLength(1);
    expect(useClassStore.getState().classes[0].label).toBe('air');
    expect(onActivate).toHaveBeenCalledWith(useClassStore.getState().classes[0].classId);
  });

  it('quick-add re-activates an existing class of the same label instead of duplicating', async () => {
    useClassStore.getState().addClass('air', '#111111');
    const onActivate = vi.fn();
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={onActivate} />);
    // Already exists, so it's no longer offered as a quick-add suggestion —
    // the row itself is the only way to reactivate it.
    expect(screen.queryByRole('button', { name: 'air' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('option', { name: 'air' }));
    expect(useClassStore.getState().classes).toHaveLength(1);
  });

  it('adds a class via the add form', async () => {
    const onActivate = vi.fn();
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={onActivate} />);
    await user.click(screen.getByLabelText('Add class'));
    await user.type(screen.getByPlaceholderText('Class label'), 'Cell');
    await user.click(screen.getByRole('button', { name: 'Add' }));
    expect(useClassStore.getState().classes.map((c) => c.label)).toEqual(['Cell']);
  });

  it('rejects a duplicate label with an alert and does not add it', async () => {
    useClassStore.getState().addClass('Cell', '#111111');
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    await user.click(screen.getByLabelText('Add class'));
    await user.type(screen.getByPlaceholderText('Class label'), 'Cell');
    await user.click(screen.getByRole('button', { name: 'Add' }));
    expect(window.alert).toHaveBeenCalledWith('A class with that label already exists.');
    expect(useClassStore.getState().classes).toHaveLength(1);
  });

  it('toggles class visibility', async () => {
    useClassStore.getState().addClass('Cell', '#111111');
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    await user.click(screen.getByLabelText('Hide class'));
    expect(useClassStore.getState().classes[0].isVisible).toBe(false);
  });

  it('renames a class through the inline editor', async () => {
    useClassStore.getState().addClass('Cell', '#111111');
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    await user.click(screen.getByLabelText('Edit class label and color'));
    const input = screen.getByDisplayValue('Cell');
    await user.clear(input);
    await user.type(input, 'Nucleus{Enter}');
    expect(useClassStore.getState().classes[0].label).toBe('Nucleus');
  });

  it('deletes a class after confirmation', async () => {
    useClassStore.getState().addClass('Cell', '#111111');
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    await user.click(screen.getByLabelText('Delete class and its annotations'));
    expect(window.confirm).toHaveBeenCalled();
    expect(useClassStore.getState().classes).toHaveLength(0);
  });

  it('cancelling the confirm dialog keeps the class', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    useClassStore.getState().addClass('Cell', '#111111');
    const user = userEvent.setup();
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    await user.click(screen.getByLabelText('Delete class and its annotations'));
    expect(useClassStore.getState().classes).toHaveLength(1);
  });

  it('duplicates a class and its shapes into a new class', async () => {
    useDatasetStore.setState({ source: 'sample.tif', kind: 'local', serverUri: null } as any);
    const classId = useClassStore.getState().addClass('Cell', '#111111');
    useAnnotationStore.getState().replaceClassShapesOnSlice('local:sample.tif', 0, classId, [
      { id: 's1', classId, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    const onActivate = vi.fn();
    const user = userEvent.setup();
    render(<ClassManager activeClassId={classId} onActivate={onActivate} />);
    await user.click(screen.getByLabelText('Duplicate class and its annotations'));
    const classes = useClassStore.getState().classes;
    expect(classes).toHaveLength(2);
    expect(classes[1].label).toBe('Cell copy');
    const newId = classes[1].classId;
    expect(useAnnotationStore.getState().byImage['local:sample.tif']['0'].some((s) => s.classId === newId)).toBe(true);
  });

  it('shows guide-defined suggestions instead of the generic defaults when a guide is loaded', () => {
    useReferenceGuideStore.getState().setGuide(
      [{ label: 'Pore', color: '#123456', description: '', exampleCrops: [] }],
      '',
      'local:sample.tif',
    );
    render(<ClassManager activeClassId={null} onActivate={vi.fn()} />);
    expect(screen.getByRole('button', { name: /Pore/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'air' })).not.toBeInTheDocument();
  });
});
