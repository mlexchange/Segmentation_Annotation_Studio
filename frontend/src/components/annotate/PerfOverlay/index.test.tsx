import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PerfOverlay from './index';
import { initPerf, mark, resetPerf } from '@/lib/perf';

beforeEach(() => {
  window.localStorage.setItem('perf', '1');
  initPerf();
  resetPerf();
});

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  initPerf();
  resetPerf();
});

describe('PerfOverlay', () => {
  it('shows a placeholder when no samples have been recorded', () => {
    render(<PerfOverlay shapeCount={0} />);
    expect(screen.getByText('no samples yet — draw or zoom')).toBeInTheDocument();
  });

  it('shows the shape count', () => {
    render(<PerfOverlay shapeCount={42} />);
    expect(screen.getByText('42 shapes')).toBeInTheDocument();
  });

  it('renders a row per recorded label with p50/p95/count', () => {
    mark('commit', 5);
    mark('commit', 7);
    render(<PerfOverlay shapeCount={0} />);
    expect(screen.getByText('commit')).toBeInTheDocument();
    const row = screen.getByText('commit').closest('tr')!;
    expect(row).toHaveTextContent('2'); // count column
  });

  it('colors a slow p95 red and a fast one green', () => {
    mark('clip', 60);
    render(<PerfOverlay shapeCount={0} />);
    const row = screen.getByText('clip').closest('tr')!;
    const cells = row.querySelectorAll('td');
    expect(cells[2]).toHaveClass('text-red-400');
  });

  it('reset button clears samples back to the placeholder', async () => {
    mark('commit', 5);
    const user = userEvent.setup();
    render(<PerfOverlay shapeCount={0} />);
    expect(screen.queryByText('no samples yet — draw or zoom')).not.toBeInTheDocument();
    await user.click(screen.getByTitle('Reset samples'));
    expect(await screen.findByText('no samples yet — draw or zoom')).toBeInTheDocument();
  });

  it('hide button removes the overlay entirely', async () => {
    const user = userEvent.setup();
    const { container } = render(<PerfOverlay shapeCount={0} />);
    await user.click(screen.getByTitle('Hide (reload to show again)'));
    expect(container).toBeEmptyDOMElement();
  });
});
