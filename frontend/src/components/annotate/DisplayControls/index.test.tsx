import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DisplayControls, { type DisplayControlsProps } from './index';

afterEach(() => {
  cleanup();
});

function baseProps(overrides: Partial<DisplayControlsProps> = {}): DisplayControlsProps {
  return {
    brightness: 0,
    contrast: 0,
    onBrightnessChange: vi.fn(),
    onContrastChange: vi.fn(),
    onReset: vi.fn(),
    histogramBins: null,
    levelsLo: 0,
    levelsHi: 255,
    onLevelsChange: vi.fn(),
    onLevelsReset: vi.fn(),
    colormap: 'gray',
    gamma: 1,
    onColormapChange: vi.fn(),
    onGammaChange: vi.fn(),
    clahe: false,
    sharpen: false,
    onClaheChange: vi.fn(),
    onSharpenChange: vi.fn(),
    blur: 0,
    onBlurChange: vi.fn(),
    upscale: 1,
    onUpscaleChange: vi.fn(),
    ...overrides,
  };
}

describe('DisplayControls', () => {
  it('renders the reset button and calls onReset', async () => {
    const onReset = vi.fn();
    const user = userEvent.setup();
    render(<DisplayControls {...baseProps({ onReset })} />);
    await user.click(screen.getByLabelText('Reset brightness and contrast'));
    expect(onReset).toHaveBeenCalledOnce();
  });

  it('toggles CLAHE and Sharpen checkboxes', async () => {
    const onClaheChange = vi.fn();
    const onSharpenChange = vi.fn();
    const user = userEvent.setup();
    render(<DisplayControls {...baseProps({ onClaheChange, onSharpenChange })} />);
    await user.click(screen.getByLabelText('CLAHE'));
    expect(onClaheChange).toHaveBeenCalledWith(true);
    await user.click(screen.getByLabelText('Sharpen'));
    expect(onSharpenChange).toHaveBeenCalledWith(true);
  });

  it('marks the current colormap as pressed', () => {
    render(<DisplayControls {...baseProps({ colormap: 'viridis' })} />);
    expect(screen.getByTitle('viridis')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTitle('gray')).toHaveAttribute('aria-pressed', 'false');
  });

  it('clicking a colormap swatch calls onColormapChange', async () => {
    const onColormapChange = vi.fn();
    const user = userEvent.setup();
    render(<DisplayControls {...baseProps({ onColormapChange })} />);
    await user.click(screen.getByTitle('viridis'));
    expect(onColormapChange).toHaveBeenCalledWith('viridis');
  });

  it('working-resolution radio group reflects the current upscale', () => {
    render(<DisplayControls {...baseProps({ upscale: 2 })} />);
    const radios = screen.getAllByRole('radio');
    expect(radios[0]).toHaveAttribute('aria-checked', 'false');
    expect(radios[1]).toHaveAttribute('aria-checked', 'true');
  });

  it('disables an upscale option above maxUpscale', () => {
    render(<DisplayControls {...baseProps({ maxUpscale: 1 })} />);
    const radios = screen.getAllByRole('radio');
    expect(radios[0]).not.toBeDisabled();
    expect(radios[1]).toBeDisabled();
    expect(radios[2]).toBeDisabled();
  });

  it('clicking an enabled upscale option calls onUpscaleChange', async () => {
    const onUpscaleChange = vi.fn();
    const user = userEvent.setup();
    render(<DisplayControls {...baseProps({ onUpscaleChange })} />);
    await user.click(screen.getAllByRole('radio')[1]);
    expect(onUpscaleChange).toHaveBeenCalledWith(2);
  });

  it('renders the denoise slot when provided', () => {
    render(<DisplayControls {...baseProps({ denoiseSlot: <div data-testid="denoise-slot" /> })} />);
    expect(screen.getByTestId('denoise-slot')).toBeInTheDocument();
  });

  it('omits the denoise slot area when not provided', () => {
    render(<DisplayControls {...baseProps()} />);
    expect(screen.queryByTestId('denoise-slot')).not.toBeInTheDocument();
  });
});
