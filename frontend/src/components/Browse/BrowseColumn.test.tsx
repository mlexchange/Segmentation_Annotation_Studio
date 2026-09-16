import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import BrowseColumn from './BrowseColumn';
import type { ColumnState } from './hooks/useBrowseData';

afterEach(() => {
  cleanup();
});

function makeColumn(overrides: Partial<ColumnState> = {}): ColumnState {
  return {
    field: 'sample_type',
    values: [],
    loading: false,
    error: null,
    selected: null,
    ...overrides,
  };
}

const baseProps = {
  colIndex: 0,
  facets: ['sample_type', 'technique', 'beamline'],
  width: 200,
  onFieldChange: vi.fn(),
  onSelect: vi.fn(),
  onRemove: vi.fn(),
  isLast: true,
};

describe('BrowseColumn', () => {
  it('renders all facets as select options', () => {
    render(<BrowseColumn {...baseProps} column={makeColumn()} />);
    const select = screen.getByTitle('sample_type') as HTMLSelectElement;
    expect(select.value).toBe('sample_type');
    for (const facet of baseProps.facets) {
      expect(screen.getByRole('option', { name: facet })).toBeInTheDocument();
    }
  });

  it('shows a loading state', () => {
    render(<BrowseColumn {...baseProps} column={makeColumn({ loading: true })} />);
    expect(screen.getByText('Loading…')).toBeInTheDocument();
  });

  it('shows an error state', () => {
    render(<BrowseColumn {...baseProps} column={makeColumn({ error: 'boom' })} />);
    expect(screen.getByText('boom')).toBeInTheDocument();
  });

  it('shows a "No values" empty state when not loading/erroring and empty', () => {
    render(<BrowseColumn {...baseProps} column={makeColumn({ values: [] })} />);
    expect(screen.getByText('No values')).toBeInTheDocument();
  });

  it('renders values with their counts', () => {
    render(
      <BrowseColumn
        {...baseProps}
        column={makeColumn({
          values: [
            { value: 'gold', count: 12, sample_paths: [] },
            { value: 'silver', count: 3, sample_paths: [] },
          ],
        })}
      />,
    );
    expect(screen.getByText('gold')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('silver')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('applies formatValue to displayed values but not to onSelect', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const formatValue = (field: string, value: string) => `${field}:${value}`;
    render(
      <BrowseColumn
        {...baseProps}
        onSelect={onSelect}
        formatValue={formatValue}
        column={makeColumn({ values: [{ value: 'gold', count: 12, sample_paths: [] }] })}
      />,
    );
    expect(screen.getByText('sample_type:gold')).toBeInTheDocument();
    await user.click(screen.getByText('sample_type:gold'));
    expect(onSelect).toHaveBeenCalledWith(0, 'gold');
  });

  it('clicking an unselected value selects it', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <BrowseColumn
        {...baseProps}
        onSelect={onSelect}
        column={makeColumn({ values: [{ value: 'gold', count: 12, sample_paths: [] }] })}
      />,
    );
    await user.click(screen.getByText('gold'));
    expect(onSelect).toHaveBeenCalledWith(0, 'gold');
  });

  it('clicking the already-selected value clears it (passes null)', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <BrowseColumn
        {...baseProps}
        onSelect={onSelect}
        column={makeColumn({
          selected: 'gold',
          values: [{ value: 'gold', count: 12, sample_paths: [] }],
        })}
      />,
    );
    await user.click(screen.getByText('gold'));
    expect(onSelect).toHaveBeenCalledWith(0, null);
  });

  it('changing the select calls onFieldChange with the column index and new field', async () => {
    const onFieldChange = vi.fn();
    const user = userEvent.setup();
    render(<BrowseColumn {...baseProps} onFieldChange={onFieldChange} column={makeColumn()} />);
    await user.selectOptions(screen.getByTitle('sample_type'), 'technique');
    expect(onFieldChange).toHaveBeenCalledWith(0, 'technique');
  });

  it('clicking remove calls onRemove with the column index', async () => {
    const onRemove = vi.fn();
    const user = userEvent.setup();
    render(<BrowseColumn {...baseProps} onRemove={onRemove} column={makeColumn()} />);
    await user.click(screen.getByTitle('Remove column'));
    expect(onRemove).toHaveBeenCalledWith(0);
  });
});
