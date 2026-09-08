import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import StarRating from './StarRating';

afterEach(() => {
  cleanup();
});

describe('StarRating', () => {
  it('renders three stars', () => {
    render(<StarRating value={0} onChange={vi.fn()} />);
    expect(screen.getAllByRole('button')).toHaveLength(3);
  });

  it('fills stars up to and including the current value', () => {
    render(<StarRating value={2} onChange={vi.fn()} />);
    const buttons = screen.getAllByRole('button');
    expect(buttons[0].querySelector('svg')).toHaveClass('text-amber-400');
    expect(buttons[1].querySelector('svg')).toHaveClass('text-amber-400');
    expect(buttons[2].querySelector('svg')).toHaveClass('text-slate-600');
  });

  it('clicking a star calls onChange with that star value', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<StarRating value={0} onChange={onChange} />);
    await user.click(screen.getByLabelText('2 stars'));
    expect(onChange).toHaveBeenCalledWith(2);
  });

  it('clicking the already-set star clears the rating to 0', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<StarRating value={2} onChange={onChange} />);
    await user.click(screen.getByLabelText('2 stars'));
    expect(onChange).toHaveBeenCalledWith(0);
  });

  it('readonly disables all buttons and omits the title', () => {
    render(<StarRating value={1} onChange={vi.fn()} readonly />);
    for (const button of screen.getAllByRole('button')) {
      expect(button).toBeDisabled();
      expect(button).not.toHaveAttribute('title');
    }
  });

  it('click does not propagate to a parent handler', async () => {
    const onParentClick = vi.fn();
    const user = userEvent.setup();
    render(
      <div onClick={onParentClick}>
        <StarRating value={0} onChange={vi.fn()} />
      </div>,
    );
    await user.click(screen.getByLabelText('1 star'));
    expect(onParentClick).not.toHaveBeenCalled();
  });
});
