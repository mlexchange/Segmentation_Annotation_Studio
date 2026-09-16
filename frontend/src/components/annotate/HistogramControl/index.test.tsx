import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import HistogramControl from './index';

beforeEach(() => {
  // jsdom implements neither in its default HTMLElement — the component calls
  // both unconditionally on pointerdown/up, and schedules onChange via rAF.
  Element.prototype.setPointerCapture = vi.fn();
  Element.prototype.releasePointerCapture = vi.fn();
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    cb(0);
    return 0;
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('HistogramControl', () => {
  it('shows the current lo/hi window', () => {
    render(<HistogramControl bins={null} lo={10} hi={200} onChange={vi.fn()} onReset={vi.fn()} />);
    expect(screen.getByText('10 – 200')).toBeInTheDocument();
  });

  it('defaults to the "Levels" label', () => {
    render(<HistogramControl bins={null} lo={0} hi={255} onChange={vi.fn()} onReset={vi.fn()} />);
    expect(screen.getByText(/Levels/)).toBeInTheDocument();
    expect(screen.getByLabelText('Levels minimum')).toBeInTheDocument();
  });

  it('supports a custom label (e.g. Threshold)', () => {
    render(<HistogramControl bins={null} lo={0} hi={255} onChange={vi.fn()} onReset={vi.fn()} label="Threshold" />);
    expect(screen.getByLabelText('Threshold minimum')).toBeInTheDocument();
    expect(screen.getByLabelText('Reset threshold')).toBeInTheDocument();
  });

  it('reset button calls onReset', () => {
    const onReset = vi.fn();
    render(<HistogramControl bins={null} lo={0} hi={255} onChange={vi.fn()} onReset={onReset} />);
    fireEvent.click(screen.getByLabelText('Reset levels'));
    expect(onReset).toHaveBeenCalledOnce();
  });

  it('ArrowRight on the max knob increases hi by 1', () => {
    const onChange = vi.fn();
    render(<HistogramControl bins={null} lo={0} hi={200} onChange={onChange} onReset={vi.fn()} />);
    fireEvent.keyDown(screen.getByLabelText('Levels maximum'), { key: 'ArrowRight' });
    expect(onChange).toHaveBeenCalledWith(0, 201);
  });

  it('Shift+ArrowLeft on the min knob decreases lo by 10', () => {
    const onChange = vi.fn();
    render(<HistogramControl bins={null} lo={50} hi={200} onChange={onChange} onReset={vi.fn()} />);
    fireEvent.keyDown(screen.getByLabelText('Levels minimum'), { key: 'ArrowLeft', shiftKey: true });
    expect(onChange).toHaveBeenCalledWith(40, 200);
  });

  it('clamps the min knob so it cannot cross the max', () => {
    const onChange = vi.fn();
    render(<HistogramControl bins={null} lo={198} hi={200} onChange={onChange} onReset={vi.fn()} />);
    fireEvent.keyDown(screen.getByLabelText('Levels minimum'), { key: 'ArrowRight', shiftKey: true });
    expect(onChange).toHaveBeenCalledWith(199, 200);
  });

  it('renders extra actions next to the reset button', () => {
    render(
      <HistogramControl
        bins={null} lo={0} hi={255} onChange={vi.fn()} onReset={vi.fn()}
        actions={<button>Auto</button>}
      />,
    );
    expect(screen.getByRole('button', { name: 'Auto' })).toBeInTheDocument();
  });

  /** jsdom has no PointerEvent constructor, and fireEvent's plain-object init
   * can't set MouseEvent's readonly clientX getter through it — so pointer
   * drags are dispatched as a real (jsdom-supported) MouseEvent typed as
   * 'pointerdown'/'pointermove', which React's delegated listener still picks
   * up (it just reads whatever's on the native event). */
  function pointerEvent(type: string, clientX: number): Event {
    return new MouseEvent(type, { clientX, bubbles: true, cancelable: true });
  }

  it('dragging near an edge (far from either knob) drags the whole band', () => {
    const onChange = vi.fn();
    const { container } = render(
      <HistogramControl bins={null} lo={0} hi={255} onChange={onChange} onReset={vi.fn()} />,
    );
    const track = container.querySelector('.touch-none') as HTMLElement;
    track.getBoundingClientRect = () => ({
      left: 0, right: 256, width: 256, top: 0, bottom: 60, height: 60, x: 0, y: 0, toJSON: () => ({}),
    });
    // v≈10 here is far from both lo=0 and hi=255 (tolerance is 6), and strictly
    // between them, so this is classified as a band drag, not a knob drag —
    // the window translates as a whole, preserving its width (255).
    track.dispatchEvent(pointerEvent('pointerdown', 10));
    expect(onChange).toHaveBeenCalled();
    const [nlo, nhi] = onChange.mock.calls[0];
    expect(nhi - nlo).toBe(255);
  });

  it('dragging near the min knob moves only lo, leaving hi fixed', () => {
    const onChange = vi.fn();
    const { container } = render(
      <HistogramControl bins={null} lo={50} hi={200} onChange={onChange} onReset={vi.fn()} />,
    );
    const track = container.querySelector('.touch-none') as HTMLElement;
    track.getBoundingClientRect = () => ({
      left: 0, right: 256, width: 256, top: 0, bottom: 60, height: 60, x: 0, y: 0, toJSON: () => ({}),
    });
    // v≈52 is within 6 of lo=50 — a knob drag, not a band drag.
    track.dispatchEvent(pointerEvent('pointerdown', 52));
    expect(onChange).toHaveBeenCalled();
    const [nlo, nhi] = onChange.mock.calls[0];
    expect(nhi).toBe(200);
    expect(nlo).toBeGreaterThanOrEqual(50);
  });
});
