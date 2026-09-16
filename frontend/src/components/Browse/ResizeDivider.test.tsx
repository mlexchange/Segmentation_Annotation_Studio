import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import ResizeDivider, { MAX_WIDTH, MIN_WIDTH } from './ResizeDivider';

afterEach(() => {
  cleanup();
});

describe('ResizeDivider', () => {
  it('renders a vertical separator', () => {
    render(<ResizeDivider currentWidth={200} onResize={vi.fn()} />);
    expect(screen.getByRole('separator')).toHaveAttribute('aria-orientation', 'vertical');
  });

  it('dragging right grows the column when resizeRight is false', () => {
    const onResize = vi.fn();
    render(<ResizeDivider currentWidth={200} onResize={onResize} />);
    const divider = screen.getByRole('separator');

    fireEvent.mouseDown(divider, { clientX: 100 });
    act(() => {
      fireEvent.mouseMove(document, { clientX: 150 });
    });
    expect(onResize).toHaveBeenCalledWith(250);
  });

  it('dragging right shrinks the column when resizeRight is true', () => {
    const onResize = vi.fn();
    render(<ResizeDivider currentWidth={200} onResize={onResize} resizeRight />);
    const divider = screen.getByRole('separator');

    fireEvent.mouseDown(divider, { clientX: 100 });
    act(() => {
      fireEvent.mouseMove(document, { clientX: 150 });
    });
    expect(onResize).toHaveBeenCalledWith(150);
  });

  it('clamps to minWidth/maxWidth', () => {
    const onResize = vi.fn();
    render(<ResizeDivider currentWidth={200} onResize={onResize} minWidth={120} maxWidth={600} />);
    const divider = screen.getByRole('separator');

    fireEvent.mouseDown(divider, { clientX: 100 });
    act(() => {
      fireEvent.mouseMove(document, { clientX: -10000 });
    });
    expect(onResize).toHaveBeenLastCalledWith(MIN_WIDTH);

    act(() => {
      fireEvent.mouseMove(document, { clientX: 10000 });
    });
    expect(onResize).toHaveBeenLastCalledWith(MAX_WIDTH);
  });

  it('stops tracking mouse movement after mouseup', () => {
    const onResize = vi.fn();
    render(<ResizeDivider currentWidth={200} onResize={onResize} />);
    const divider = screen.getByRole('separator');

    fireEvent.mouseDown(divider, { clientX: 100 });
    act(() => {
      fireEvent.mouseUp(document);
    });
    onResize.mockClear();
    act(() => {
      fireEvent.mouseMove(document, { clientX: 300 });
    });
    expect(onResize).not.toHaveBeenCalled();
  });
});
