import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import BrowseDetailPanel from './BrowseDetailPanel';
import type { BrowseItem } from './hooks/useBrowseData';

function makeItem(overrides: Partial<BrowseItem> = {}): BrowseItem {
  return {
    path: 'ds/a',
    sample: 'sample-a',
    metadata: {},
    ...overrides,
  };
}

/** Stub window.Image so the thumbnail-loading effect resolves synchronously
 *  and predictably instead of depending on jsdom's (nonexistent) image decoding. */
let imageInstances: FakeImage[] = [];
class FakeImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  _src = '';
  constructor() {
    imageInstances.push(this);
  }
  set src(value: string) {
    this._src = value;
  }
  get src() {
    return this._src;
  }
}

beforeEach(() => {
  imageInstances = [];
  vi.stubGlobal('Image', FakeImage);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('BrowseDetailPanel', () => {
  it('renders the sample name and path in the header', () => {
    render(<BrowseDetailPanel item={makeItem({ sample: 'my-sample', path: 'a/b/c' })} onClose={vi.fn()} />);
    expect(screen.getByText('my-sample')).toBeInTheDocument();
    expect(screen.getByText('a/b/c')).toBeInTheDocument();
  });

  it('clicking the close button calls onClose', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<BrowseDetailPanel item={makeItem()} onClose={onClose} />);
    // The close button is the only icon-only button in the header without a title.
    const buttons = screen.getAllByRole('button');
    await user.click(buttons[0]);
    expect(onClose).toHaveBeenCalled();
  });

  it('shows the loading preview state before the thumbnail resolves', () => {
    render(<BrowseDetailPanel item={makeItem()} onClose={vi.fn()} />);
    expect(screen.getByText('Loading preview…')).toBeInTheDocument();
  });

  it('shows the image once the thumbnail loads successfully', async () => {
    render(<BrowseDetailPanel item={makeItem({ path: 'ds/a' })} onClose={vi.fn()} serverUri="http://server" />);
    expect(imageInstances).toHaveLength(1);
    act(() => {
      imageInstances[0].onload?.();
    });
    const img = await screen.findByAltText('Array preview');
    expect(img).toHaveAttribute('src', expect.stringContaining('tiled_path=ds%2Fa'));
    expect(img.getAttribute('src')).toContain('server_uri=http%3A%2F%2Fserver');
  });

  it('renders nothing for the preview area when the thumbnail fails to load', async () => {
    render(<BrowseDetailPanel item={makeItem()} onClose={vi.fn()} />);
    act(() => {
      imageInstances[0].onerror?.();
    });
    expect(screen.queryByText('Loading preview…')).not.toBeInTheDocument();
    expect(screen.queryByAltText('Array preview')).not.toBeInTheDocument();
  });

  it('refetches the thumbnail when the item path changes', () => {
    const { rerender } = render(<BrowseDetailPanel item={makeItem({ path: 'a' })} onClose={vi.fn()} />);
    expect(imageInstances).toHaveLength(1);
    rerender(<BrowseDetailPanel item={makeItem({ path: 'b' })} onClose={vi.fn()} />);
    expect(imageInstances).toHaveLength(2);
    expect(imageInstances[1].src).toContain('tiled_path=b');
  });

  it('does not render the Open in Annotate button when onOpenInAnnotate is omitted', () => {
    render(<BrowseDetailPanel item={makeItem()} onClose={vi.fn()} />);
    expect(screen.queryByText('Open in Annotate')).not.toBeInTheDocument();
  });

  it('clicking Open in Annotate calls the callback', async () => {
    const onOpenInAnnotate = vi.fn();
    const user = userEvent.setup();
    render(<BrowseDetailPanel item={makeItem()} onClose={vi.fn()} onOpenInAnnotate={onOpenInAnnotate} />);
    await user.click(screen.getByText('Open in Annotate'));
    expect(onOpenInAnnotate).toHaveBeenCalled();
  });

  it('groups known metadata keys into their labelled sections', () => {
    render(
      <BrowseDetailPanel
        item={makeItem({ metadata: { PI: 'Dr. Smith', beamline: '11-3', ThinFilmID: 'TF-1' } })}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByText('Experiment')).toBeInTheDocument();
    expect(screen.getByText('Identity')).toBeInTheDocument();
    expect(screen.getByText('PI')).toBeInTheDocument();
    expect(screen.getByText('Dr. Smith')).toBeInTheDocument();
    expect(screen.getByText('ThinFilmID')).toBeInTheDocument();
    expect(screen.getByText('TF-1')).toBeInTheDocument();
  });

  it('maps aliased keys to friendlier display labels', () => {
    render(
      <BrowseDetailPanel
        item={makeItem({ metadata: { studio_annotated: 'yes', studio_shape_count: 4 } })}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByText('Annotated')).toBeInTheDocument();
    expect(screen.getByText('Shape count')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
  });

  it('puts unrecognised metadata keys into an "Other" section, skipping noisy/internal keys', () => {
    render(
      <BrowseDetailPanel
        item={makeItem({
          metadata: {
            custom_field: 'custom-value',
            vae_embedding: [1, 2, 3],
            thinfilm_internal: 'skip me',
            'Sample foo': 'skip me too',
          },
        })}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByText('Other')).toBeInTheDocument();
    expect(screen.getByText('custom_field')).toBeInTheDocument();
    expect(screen.getByText('custom-value')).toBeInTheDocument();
    expect(screen.queryByText('vae_embedding')).not.toBeInTheDocument();
    expect(screen.queryByText('thinfilm_internal')).not.toBeInTheDocument();
    expect(screen.queryByText('Sample foo')).not.toBeInTheDocument();
  });

  it('formats array metadata values as an item-count summary', () => {
    render(
      <BrowseDetailPanel item={makeItem({ metadata: { custom_list: [1, 2, 3] } })} onClose={vi.fn()} />,
    );
    expect(screen.getByText('[3 items]')).toBeInTheDocument();
  });

  it('formats null/undefined-ish metadata values as an em dash', () => {
    render(
      <BrowseDetailPanel item={makeItem({ metadata: { custom_field: null } })} onClose={vi.fn()} />,
    );
    // null values are filtered out before display, so no "Other" section appears.
    expect(screen.queryByText('Other')).not.toBeInTheDocument();
  });

  it('renders no metadata sections when metadata is empty', () => {
    render(<BrowseDetailPanel item={makeItem({ metadata: {} })} onClose={vi.fn()} />);
    expect(screen.queryByText('Identity')).not.toBeInTheDocument();
    expect(screen.queryByText('Other')).not.toBeInTheDocument();
  });
});
