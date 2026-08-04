import { useCallback, useEffect, useRef, useState } from 'react';

const MIN_WIDTH = 120;
const MAX_WIDTH = 600;

interface ResizeDividerProps {
  currentWidth: number;
  onResize: (newWidth: number) => void;
  /** When true, dragging right shrinks this column (resize the column to the right of the divider) */
  resizeRight?: boolean;
  /** Optional clamp overrides (defaults: 120 / 600). */
  minWidth?: number;
  maxWidth?: number;
  className?: string;
}

/**
 * ResizeDivider — draggable vertical separator that resizes an adjacent column.
 * Tracks mouse drag globally while pressed and reports the clamped width via onResize.
 */
export default function ResizeDivider({
  currentWidth,
  onResize,
  resizeRight = false,
  minWidth = MIN_WIDTH,
  maxWidth = MAX_WIDTH,
  className = '',
}: ResizeDividerProps) {
  const [dragging, setDragging] = useState(false);
  const startX = useRef(0);
  const startWidth = useRef(currentWidth);

  /** Captures the drag start position/width and begins the resize gesture. */
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      startX.current = e.clientX;
      startWidth.current = currentWidth;
      setDragging(true);
    },
    [currentWidth]
  );

  /** Arrow keys nudge the width by 16px (Shift = 48px); Home/End jump to the clamp bounds. */
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const step = e.shiftKey ? 48 : 16;
      let delta = 0;
      if (e.key === 'ArrowLeft') delta = resizeRight ? step : -step;
      else if (e.key === 'ArrowRight') delta = resizeRight ? -step : step;
      else if (e.key === 'Home') { onResize(minWidth); e.preventDefault(); return; }
      else if (e.key === 'End') { onResize(maxWidth); e.preventDefault(); return; }
      else return;
      e.preventDefault();
      onResize(Math.min(maxWidth, Math.max(minWidth, currentWidth + delta)));
    },
    [currentWidth, minWidth, maxWidth, resizeRight, onResize]
  );

  useEffect(() => {
    if (!dragging) return;

    const handleMove = (e: MouseEvent) => {
      const delta = e.clientX - startX.current;
      const signed = resizeRight ? -delta : delta;
      const newWidth = Math.min(maxWidth, Math.max(minWidth, startWidth.current + signed));
      onResize(newWidth);
    };

    const handleUp = () => {
      setDragging(false);
    };

    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [dragging, onResize, resizeRight, minWidth, maxWidth]);

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-valuenow={Math.round(currentWidth)}
      aria-valuemin={minWidth}
      aria-valuemax={maxWidth}
      aria-label="Resize column"
      tabIndex={0}
      onMouseDown={handleMouseDown}
      onKeyDown={handleKeyDown}
      className={`shrink-0 flex items-stretch cursor-col-resize select-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 ${className}`}
      style={{
        width: 8,
        marginLeft: -2,
        marginRight: -2,
        zIndex: dragging ? 10 : 1,
        background: dragging ? 'rgba(59, 130, 246, 0.4)' : 'transparent',
      }}
      title="Drag to resize column"
    >
      <div
        className="shrink-0 w-px self-stretch transition-colors"
        style={{
          margin: '0 3px',
          background: dragging ? '#3b82f6' : '#334155',
        }}
      />
    </div>
  );
}

export { MIN_WIDTH, MAX_WIDTH };
