/**
 * Toolbar — tool selector (radiogroup), brush size, opacity, undo/redo.
 * Keybinds: q=polygon, w=ellipse, e=rectangle, a=pan, b=brush, x=eraser, s=select
 */
import { Hand, Cursor, Polygon, Rectangle, Circle, PaintBucket, Eraser, ArrowBendUpLeft, ArrowBendUpRight } from '@phosphor-icons/react';
import { useStore } from 'zustand';
import { useToolStore, type Tool } from '@/stores/toolStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { cn } from '@/lib/utils';

interface ToolButtonProps {
  tool: Tool;
  label: string;
  icon: React.ReactNode;
  keybind?: string;
  activeTool: Tool;
  onSelect: (t: Tool) => void;
}

function ToolButton({ tool, label, icon, keybind, activeTool, onSelect }: ToolButtonProps) {
  const isActive = tool === activeTool;
  return (
    <button
      role="radio"
      aria-checked={isActive}
      aria-label={`${label}${keybind ? ` (${keybind})` : ''}`}
      title={`${label}${keybind ? ` [${keybind.toUpperCase()}]` : ''}`}
      onClick={() => onSelect(tool)}
      className={cn(
        'relative flex flex-col items-center justify-center gap-0.5 py-2 px-1 rounded-md text-xs w-full',
        'transition-colors border',
        isActive
          ? 'bg-sky-600 text-white border-sky-700'
          : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
      )}
    >
      {icon}
      <span className="leading-tight">{label}</span>
      {keybind && (
        <span
          className={cn(
            'absolute top-1 right-1 text-[9px] font-mono font-semibold leading-none px-0.5 py-0.5 rounded',
            isActive ? 'text-sky-200/80' : 'text-gray-400'
          )}
        >
          {keybind.toUpperCase()}
        </span>
      )}
    </button>
  );
}

export default function Toolbar() {
  const { tool, setTool, brushSize, setBrushSize, fillOpacity, setFillOpacity } = useToolStore();
  const { undo, redo } = useStore(useAnnotationStore.temporal);
  const canUndo = useStore(useAnnotationStore.temporal, (s) => s.pastStates.length > 0);
  const canRedo = useStore(useAnnotationStore.temporal, (s) => s.futureStates.length > 0);

  const tools: Array<{ tool: Tool; label: string; icon: React.ReactNode; keybind: string }> = [
    { tool: 'pan',       label: 'Pan',     icon: <Hand size={18} />,        keybind: 'a' },
    { tool: 'select',    label: 'Select',  icon: <Cursor size={18} />,      keybind: 's' },
    { tool: 'polygon',   label: 'Polygon', icon: <Polygon size={18} />,     keybind: 'q' },
    { tool: 'rectangle', label: 'Rect',    icon: <Rectangle size={18} />,   keybind: 'e' },
    { tool: 'ellipse',   label: 'Ellipse', icon: <Circle size={18} />,      keybind: 'w' },
    { tool: 'brush',     label: 'Brush',   icon: <PaintBucket size={18} />, keybind: 'b' },
    { tool: 'eraser',    label: 'Eraser',  icon: <Eraser size={18} />,      keybind: 'x' },
  ];

  return (
    <div className="flex flex-col gap-2">
      {/* Undo / redo */}
      <div className="flex gap-1">
        <button
          type="button"
          onClick={() => undo()}
          disabled={!canUndo}
          title="Undo (Ctrl+Z)"
          aria-label="Undo"
          className={cn(
            'flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
            canUndo
              ? 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
              : 'bg-gray-50 text-gray-300 border-gray-100 cursor-not-allowed',
          )}
        >
          <ArrowBendUpLeft size={16} />
          Undo
        </button>
        <button
          type="button"
          onClick={() => redo()}
          disabled={!canRedo}
          title="Redo (Ctrl+Shift+Z)"
          aria-label="Redo"
          className={cn(
            'flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
            canRedo
              ? 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
              : 'bg-gray-50 text-gray-300 border-gray-100 cursor-not-allowed',
          )}
        >
          Redo
          <ArrowBendUpRight size={16} />
        </button>
      </div>

      <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Tools</span>
      <div role="radiogroup" aria-label="Drawing tools" className="grid grid-cols-2 gap-1">
        {tools.map((t) => (
          <ToolButton
            key={t.tool}
            {...t}
            activeTool={tool}
            onSelect={setTool}
          />
        ))}
      </div>

      {(tool === 'brush' || tool === 'eraser') && (
        <div className="flex flex-col gap-1 mt-1">
          <label className="text-xs text-gray-500">
            Brush radius (px): {brushSize}
          </label>
          <input
            type="range"
            min={1}
            max={100}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="w-full"
          />
        </div>
      )}

      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500">
          Annotation opacity: {Math.round(fillOpacity * 100)}%
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(fillOpacity * 100)}
          onChange={(e) => setFillOpacity(Number(e.target.value) / 100)}
          className="w-full"
        />
      </div>
    </div>
  );
}
