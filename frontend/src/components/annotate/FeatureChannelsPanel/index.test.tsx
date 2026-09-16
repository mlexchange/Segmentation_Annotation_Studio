import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FeatureChannelsPanel from './index';
import { useIpredStore, DEFAULT_COMPOSITION_ID } from '@/stores/ipredStore';
import type { FeatureJobInfo } from '@/hooks/useFeatureChannels';
import { listIpredCompositions, listIpredModules } from '@/lib/ipredApi';

vi.mock('@/lib/ipredApi', () => ({
  listIpredCompositions: vi.fn(async () => []),
  listIpredModules: vi.fn(async () => []),
}));

const mockListIpredCompositions = vi.mocked(listIpredCompositions);
const mockListIpredModules = vi.mocked(listIpredModules);

const initialIpredState = useIpredStore.getState();

const compSkimage = {
  id: 'comp-skimage',
  name: 'Skimage only',
  nodes: [{ id: 'n1', module: 'skimage_multiscale' }],
  outputs: ['n1'],
};

const compSlimsam = {
  id: 'comp-skimage-slimsam',
  name: 'Skimage + SlimSAM',
  nodes: [
    { id: 'n1', module: 'skimage_multiscale' },
    { id: 'n2', module: 'slimsam' },
  ],
  outputs: ['n1', 'n2'],
};

const compMark25 = {
  id: 'comp-skimage-mark25',
  name: 'Skimage + Mark25',
  nodes: [
    { id: 'n1', module: 'skimage_multiscale' },
    { id: 'n2', module: 'tomojepa_mark25' },
  ],
  outputs: ['n1', 'n2'],
};

const modSkimage = {
  id: 'skimage_multiscale',
  name: 'Skimage multiscale',
  description: '',
  runtime: 'numpy',
  ready: true,
  accepts_input_from: false,
  produces_channels: true,
  produces_embedding: false,
  params_schema: {},
};

const modSlimsam = {
  id: 'slimsam',
  name: 'SlimSAM',
  description: '',
  runtime: 'torch',
  ready: true,
  accepts_input_from: true,
  produces_channels: false,
  produces_embedding: true,
  params_schema: {},
};

const modMark25NotReady = {
  id: 'tomojepa_mark25',
  name: 'TomoJEPA Mark25',
  description: '',
  runtime: 'torch',
  ready: false,
  accepts_input_from: true,
  produces_channels: false,
  produces_embedding: true,
  params_schema: {},
};

function baseProps() {
  return {
    job: null as FeatureJobInfo | null,
    channelIndex: null as number | null,
    computing: false,
    error: null as string | null,
    onCompute: vi.fn(),
    onSelectChannel: vi.fn(),
    onCycle: vi.fn(),
    onOriginal: vi.fn(),
  };
}

function makeJob(overrides: Partial<FeatureJobInfo> = {}): FeatureJobInfo {
  return {
    jobId: 'job-1',
    width: 100,
    height: 100,
    channels: [
      { index: 0, label: 'edges' },
      { index: 1, label: 'texture' },
    ],
    hasSam: false,
    ...overrides,
  };
}

beforeEach(() => {
  useIpredStore.setState(initialIpredState, true);
  mockListIpredCompositions.mockReset().mockResolvedValue([]);
  mockListIpredModules.mockReset().mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
});

describe('FeatureChannelsPanel', () => {
  it('shows a loading placeholder for the recipe select while presets load', () => {
    mockListIpredCompositions.mockImplementation(() => new Promise(() => {}));
    const props = baseProps();
    render(<FeatureChannelsPanel {...props} />);
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.getByText('Recipe').parentElement?.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('renders preset options once loaded, marking the recommended one and disabling unready ones', async () => {
    mockListIpredCompositions.mockResolvedValue([compSkimage, compSlimsam, compMark25]);
    mockListIpredModules.mockResolvedValue([modSkimage, modSlimsam, modMark25NotReady]);
    useIpredStore.setState({ preferredCompositionId: 'comp-skimage-slimsam' });
    const props = baseProps();
    render(<FeatureChannelsPanel {...props} />);

    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
    const selects = screen.getAllByRole('combobox');
    const recipeSelect = selects[0];
    const options = within(recipeSelect).getAllByRole('option');
    expect(options).toHaveLength(3);

    const recommended = options.find((o) => o.textContent?.includes('★'));
    expect(recommended).toBeDefined();
    expect(recommended?.textContent).toContain('Texture-aware (+ SlimSAM)');

    const mark25Option = options.find((o) => o.getAttribute('value') === 'comp-skimage-mark25');
    expect(mark25Option).toBeDisabled();
    expect(mark25Option?.textContent).toContain('(unavailable)');

    // Recommended preset hint text is shown for the currently selected preset.
    expect(
      screen.getByText('Skimage filters + SlimSAM vision-encoder embeddings (PCA-reduced). Recommended default.'),
    ).toBeInTheDocument();
  });

  it('changing the recipe select calls setPreferredCompositionId', async () => {
    mockListIpredCompositions.mockResolvedValue([compSkimage, compSlimsam]);
    mockListIpredModules.mockResolvedValue([modSkimage, modSlimsam]);
    useIpredStore.setState({ preferredCompositionId: 'comp-skimage-slimsam' });
    const user = userEvent.setup();
    const props = baseProps();
    render(<FeatureChannelsPanel {...props} />);

    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
    const recipeSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(recipeSelect, 'comp-skimage');
    expect(useIpredStore.getState().preferredCompositionId).toBe('comp-skimage');
  });

  it('shows a warning and disables Compute when the selected preset needs a missing module', async () => {
    mockListIpredCompositions.mockResolvedValue([compMark25]);
    mockListIpredModules.mockResolvedValue([modSkimage, modMark25NotReady]);
    useIpredStore.setState({ preferredCompositionId: 'comp-skimage-mark25' });
    const props = baseProps();
    render(<FeatureChannelsPanel {...props} />);

    await waitFor(() => expect(screen.getByText(/isn't installed yet/)).toBeInTheDocument());
    expect(screen.getByText(/isn't installed yet/).closest('div')).toHaveTextContent('TomoJEPA Mark25');
    expect(screen.getByRole('button', { name: /Compute/ })).toBeDisabled();
  });

  it('falls back to the recommended preset if the stored preference is not ready once presets load', async () => {
    mockListIpredCompositions.mockResolvedValue([compSkimage, compSlimsam, compMark25]);
    mockListIpredModules.mockResolvedValue([modSkimage, modSlimsam, modMark25NotReady]);
    useIpredStore.setState({ preferredCompositionId: 'comp-skimage-mark25' });
    render(<FeatureChannelsPanel {...baseProps()} />);

    await waitFor(() =>
      expect(useIpredStore.getState().preferredCompositionId).toBe(DEFAULT_COMPOSITION_ID),
    );
  });

  it('surfaces a presets error message when loading compositions/modules fails', async () => {
    mockListIpredCompositions.mockRejectedValue(new Error('network down'));
    render(<FeatureChannelsPanel {...baseProps()} />);
    await waitFor(() => expect(screen.getByText('network down')).toBeInTheDocument());
  });

  it('clicking Compute calls onCompute', async () => {
    mockListIpredCompositions.mockResolvedValue([compSkimage]);
    mockListIpredModules.mockResolvedValue([modSkimage]);
    useIpredStore.setState({ preferredCompositionId: 'comp-skimage' });
    const user = userEvent.setup();
    const props = baseProps();
    render(<FeatureChannelsPanel {...props} />);

    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
    await user.click(screen.getByRole('button', { name: /Compute/ }));
    expect(props.onCompute).toHaveBeenCalledTimes(1);
  });

  it('shows a spinner label and disables Compute + recipe select while computing', async () => {
    mockListIpredCompositions.mockResolvedValue([compSkimage]);
    mockListIpredModules.mockResolvedValue([modSkimage]);
    const props = { ...baseProps(), computing: true };
    render(<FeatureChannelsPanel {...props} />);

    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
    expect(screen.getByText('Computing…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Computing/ })).toBeDisabled();
    expect(screen.getAllByRole('combobox')[0]).toBeDisabled();
  });

  it('disables the whole panel controls when disabled prop is set', async () => {
    mockListIpredCompositions.mockResolvedValue([compSkimage]);
    mockListIpredModules.mockResolvedValue([modSkimage]);
    const props = { ...baseProps(), disabled: true };
    render(<FeatureChannelsPanel {...props} />);

    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
    expect(screen.getByRole('button', { name: /Compute/ })).toBeDisabled();
    expect(screen.getAllByRole('combobox')[0]).toBeDisabled();
  });

  it('renders the compute-error message when error is set', async () => {
    const props = { ...baseProps(), error: 'Compute failed: boom' };
    render(<FeatureChannelsPanel {...props} />);
    expect(screen.getByText('Compute failed: boom')).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
  });

  it('shows no channel section when there is no job', async () => {
    render(<FeatureChannelsPanel {...baseProps()} />);
    expect(screen.queryByRole('button', { name: 'Previous channel' })).not.toBeInTheDocument();
    expect(screen.queryByText('Show original')).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
  });

  it('shows the header channel count summary including SAM and cache flags', async () => {
    const job = makeJob({ hasSam: true, cacheHit: true });
    render(<FeatureChannelsPanel {...baseProps()} job={job} />);
    expect(screen.getByText('2 ch +SAM · cache')).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
  });

  it('renders channel list, active label, and cycle/select controls once a job is present', async () => {
    const job = makeJob();
    const props = { ...baseProps(), job, channelIndex: 1 };
    render(<FeatureChannelsPanel {...props} />);
    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));

    expect(screen.getByRole('button', { name: 'Previous channel' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Next channel' })).toBeEnabled();
    expect(screen.getByTitle('texture')).toBeInTheDocument();

    const channelSelect = screen.getByDisplayValue('texture');
    const options = within(channelSelect).getAllByRole('option');
    expect(options.map((o) => o.textContent)).toEqual(['Original', 'edges', 'texture']);
  });

  it('disables cycle buttons when channelIndex is null (Original selected)', async () => {
    const job = makeJob();
    const props = { ...baseProps(), job, channelIndex: null };
    render(<FeatureChannelsPanel {...props} />);
    await waitFor(() => expect(screen.getAllByRole('combobox').length).toBeGreaterThan(0));
    expect(screen.getByRole('button', { name: 'Previous channel' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Next channel' })).toBeDisabled();
    expect(screen.queryByTitle('texture')).not.toBeInTheDocument();
  });

  it('clicking next/previous channel calls onCycle with the right delta', async () => {
    const job = makeJob();
    const props = { ...baseProps(), job, channelIndex: 0 };
    const user = userEvent.setup();
    render(<FeatureChannelsPanel {...props} />);

    await user.click(screen.getByRole('button', { name: 'Next channel' }));
    expect(props.onCycle).toHaveBeenCalledWith(1);
    await user.click(screen.getByRole('button', { name: 'Previous channel' }));
    expect(props.onCycle).toHaveBeenCalledWith(-1);
  });

  it('selecting a channel option calls onSelectChannel with the numeric index or null for Original', async () => {
    const job = makeJob();
    const props = { ...baseProps(), job, channelIndex: 0 };
    const user = userEvent.setup();
    render(<FeatureChannelsPanel {...props} />);

    const channelSelect = screen.getByDisplayValue('edges');
    await user.selectOptions(channelSelect, '1');
    expect(props.onSelectChannel).toHaveBeenCalledWith(1);

    await user.selectOptions(channelSelect, '');
    expect(props.onSelectChannel).toHaveBeenCalledWith(null);
  });

  it('clicking "Show original" calls onOriginal', async () => {
    const job = makeJob();
    const props = { ...baseProps(), job, channelIndex: 0 };
    const user = userEvent.setup();
    render(<FeatureChannelsPanel {...props} />);

    await user.click(screen.getByText('Show original'));
    expect(props.onOriginal).toHaveBeenCalledTimes(1);
  });

  it('toggles the Advanced disclosure to reveal the composition panel', async () => {
    const user = userEvent.setup();
    render(<FeatureChannelsPanel {...baseProps()} />);

    expect(screen.queryByText(/edit feature recipe/)).toBeInTheDocument();
    const toggle = screen.getByText(/Advanced: edit feature recipe/);
    expect(toggle.textContent).toContain('▸');

    await user.click(toggle);
    expect(toggle.textContent).toContain('▾');

    await user.click(toggle);
    expect(toggle.textContent).toContain('▸');
  });
});
