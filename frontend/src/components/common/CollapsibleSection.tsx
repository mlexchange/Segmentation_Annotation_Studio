/**
 * CollapsibleSection — shared expand/collapse header for sidebar subsections
 * (Classes, Layers, Display, Slice, Cross-slice, Measure, Features, Classifier).
 *
 * Reproduces the house header convention already used ad hoc by those panels
 * (`text-xs font-semibold uppercase tracking-wide text-gray-500` + a small
 * leading icon) and adds a Phosphor caret toggle in front of it, replacing two
 * divergent one-off disclosure patterns that existed before this component.
 */
import { useState, type ReactNode } from 'react';
import { CaretDown, CaretRight } from '@phosphor-icons/react';
import { cn } from '@/lib/utils';

export interface CollapsibleSectionProps {
  title: string;
  /** Small leading icon, e.g. `<Stack size={13} />` — matches each section's existing icon. */
  icon?: ReactNode;
  /** Whether the section starts open. Not persisted — a fresh mount always uses this. */
  defaultOpen?: boolean;
  /** Extra control(s) on the header's right edge, e.g. ClassManager's "+ Add" button. */
  headerRight?: ReactNode;
  disabled?: boolean;
  children: ReactNode;
}

/** Sidebar section wrapper: click the header to show/hide `children`. */
export default function CollapsibleSection({
  title,
  icon,
  defaultOpen = true,
  headerRight,
  disabled = false,
  children,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-1.5">
        <button
          type="button"
          disabled={disabled}
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
          className={cn(
            'flex flex-1 items-center gap-1.5 text-xs font-semibold uppercase tracking-wide transition-colors',
            disabled ? 'text-gray-300 cursor-not-allowed' : 'text-gray-500 hover:text-gray-700',
          )}
        >
          {open ? <CaretDown size={11} className="shrink-0" /> : <CaretRight size={11} className="shrink-0" />}
          {icon}
          {title}
        </button>
        {headerRight}
      </div>
      {open && <div className="flex flex-col gap-1.5">{children}</div>}
    </div>
  );
}
