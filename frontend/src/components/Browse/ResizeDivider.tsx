import { useCallback, useEffect, useRef, useState } from 'react';

const MIN_WIDTH = 120;
const MAX_WIDTH = 600;

interface ResizeDividerProps {
  currentWidth: number;
  onResize: (newWidth: number) => void;
  /** When true, dragging right shrinks this column (resize the column to the right of the divider) */
  resizeRight?: boolean;
  className?: string;
}

export default function ResizeDivider({ currentWidth, onResize, resizeRight = false, className = '' }: ResizeDividerProps) {
  const [dragging, setDragging] = useState(false);
  const startX = useRef(0);
  const startWidth = useRef(currentWidth);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      startX.current = e.clientX;
      startWidth.current = currentWidth;
      setDragging(true);
    },
    [currentWidth]
  );

  useEffect(() => {
    if (!dragging) return;

    const handleMove = (e: MouseEvent) => {
      const delta = e.clientX - startX.current;
      const signed = resizeRight ? -delta : delta;
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth.current + signed));
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
  }, [dragging, onResize, resizeRight]);

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      onMouseDown={handleMouseDown}
      className={`shrink-0 flex items-stretch cursor-col-resize select-none ${className}`}
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
