/**
 * CollapsibleSection — open/close toggle, defaultOpen, headerRight slot.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CollapsibleSection from './CollapsibleSection';

describe('CollapsibleSection', () => {
  it('renders children by default (defaultOpen defaults to true)', () => {
    render(
      <CollapsibleSection title="Layers">
        <p>body content</p>
      </CollapsibleSection>,
    );
    expect(screen.getByText('body content')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /layers/i })).toHaveAttribute('aria-expanded', 'true');
  });

  it('hides children when defaultOpen is false, and toggling shows them', async () => {
    const user = userEvent.setup();
    render(
      <CollapsibleSection title="Classifier" defaultOpen={false}>
        <p>hidden body</p>
      </CollapsibleSection>,
    );
    expect(screen.queryByText('hidden body')).not.toBeInTheDocument();
    const header = screen.getByRole('button', { name: /classifier/i });
    expect(header).toHaveAttribute('aria-expanded', 'false');

    await user.click(header);
    expect(screen.getByText('hidden body')).toBeInTheDocument();
    expect(header).toHaveAttribute('aria-expanded', 'true');

    await user.click(header);
    expect(screen.queryByText('hidden body')).not.toBeInTheDocument();
  });

  it('renders headerRight content regardless of open state', async () => {
    const user = userEvent.setup();
    render(
      <CollapsibleSection title="Classes" headerRight={<button aria-label="Add class">+</button>}>
        <p>rows</p>
      </CollapsibleSection>,
    );
    expect(screen.getByLabelText('Add class')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /classes/i }));
    expect(screen.getByLabelText('Add class')).toBeInTheDocument();
  });
});
