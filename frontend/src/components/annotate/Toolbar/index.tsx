/**
 * Toolbar — tool selector (radiogroup), brush size, opacity, undo/redo.
 * Keybinds: p=polygon, l=ellipse, e=rectangle, r=eraser, b=brush, f=fill, s=select,
 *           g=magic, m=magnetic, Space=pan (hold), x=next slice, t=fit to screen, Ctrl/Cmd+Z=undo
 */
import { Hand, Cursor, Polygon, MagnetStraight, MagicWand, Rectangle, Circle, PaintBrush, PaintBucket, Eraser, ArrowBendUpLeft, ArrowBendUpRight } from '@phosphor-icons/react';
import { useStore } from 'zustand';
import { useToolStore, type Tool } from '@/stores/toolStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import * as editHistory from '@/hooks/editHistory';
import { cn } from '@/lib/utils';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import { useSam } from '@/hooks/useSam';

// macOS labels the Alt key "Option" (⌥); the key name only differs on screen.
const IS_MAC = typeof navigator !== 'undefined' && /mac/i.test(navigator.userAgent);
const REMOVE_KEY_LABEL = IS_MAC ? 'Option' : 'Alt';

interface ToolButtonProps {
  tool: Tool;
  label: string;
  icon: React.ReactNode;
  keybind?: string;
  activeTool: Tool;
  disabled?: boolean;
  onSelect: (t: Tool) => void;
}

/** Renders one tool as a radio button showing its icon, label, and keybind badge. */
function ToolButton({ tool, label, icon, keybind, activeTool, disabled, onSelect }: ToolButtonProps) {
  const isActive = tool === activeTool && !disabled;
  // Underline the hotkey letter within the label (first case-insensitive match).
  const renderLabel = () => {
    if (!keybind || keybind.length !== 1) return label;
    const idx = label.toLowerCase().indexOf(keybind.toLowerCase());
    if (idx < 0) return label;
    return (
      <>
        {label.slice(0, idx)}
        <span className="underline underline-offset-2">{label[idx]}</span>
        {label.slice(idx + 1)}
      </>
    );
  };
  return (
    <button
      role="radio"
      aria-checked={isActive}
      aria-label={`${label}${keybind ? ` (${keybind})` : ''}`}
      title={`${label}${keybind ? ` [${keybind.toUpperCase()}]` : ''}`}
      disabled={disabled}
      onClick={() => onSelect(tool)}
      className={cn(
        'relative flex flex-col items-center justify-center gap-0.5 py-2 px-1 rounded-md text-xs w-full',
        'transition-colors border',
        disabled
          ? 'bg-gray-50 text-gray-300 border-gray-100 cursor-not-allowed'
          : isActive
            ? 'bg-sky-600 text-white border-sky-700'
            : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
      )}
    >
      {icon}
      <span className="leading-tight">{renderLabel()}</span>
      {keybind && (
        <span
          className={cn(
            'absolute top-1 right-1 text-[9px] font-mono font-semibold leading-none px-0.5 py-0.5 rounded',
            isActive ? 'text-sky-200/80' : 'text-gray-500'
          )}
        >
          {keybind.length > 1 ? keybind.slice(0, 3).toUpperCase() : keybind.toUpperCase()}
        </span>
      )}
    </button>
  );
}

interface ToolbarProps {
  /** When true, drawing tools are greyed out (e.g. no class defined yet). */
  disabled?: boolean;
}

/** Renders the tool radiogroup, undo/redo, and the active tool's parameter controls. */
export default function Toolbar({ disabled = false }: ToolbarProps) {
  const {
    tool, setTool, brushSize, setBrushSize, fillThreshold, setFillThreshold,
    magicTolerance, setMagicTolerance, magicMode, setMagicMode, magicSigma, setMagicSigma,
    magicEdgeStop, setMagicEdgeStop, magicEngine, setMagicEngine,
    samDetail, setSamDetail, samThreshold, setSamThreshold,
    samAvoidLabeled, setSamAvoidLabeled,
    samConnectedOnly, setSamConnectedOnly,
    clipToOtherClasses, setClipToOtherClasses,
    mergeOverlappingSameClass, setMergeOverlappingSameClass,
    eraseAllClasses, setEraseAllClasses,
    selectScope, setSelectScope,
  } = useToolStore();
  const sam = useSam(tool === 'magic' && magicEngine === 'sam');
  // Undo/redo route through editHistory so a class deletion replays alongside its region
  // change; canUndo/canRedo still reflect the (1:1) zundo stack.
  const canUndo = useStore(useAnnotationStore.temporal, (s) => s.pastStates.length > 0);
  const canRedo = useStore(useAnnotationStore.temporal, (s) => s.futureStates.length > 0);

  const tools: Array<{ tool: Tool; label: string; icon: React.ReactNode; keybind: string }> = [
    { tool: 'pan',       label: 'Pan',     icon: <Hand size={18} />,        keybind: 'space' },
    { tool: 'select',    label: 'Select',  icon: <Cursor size={18} />,      keybind: 's' },
    { tool: 'polygon',   label: 'Polygon', icon: <Polygon size={18} />,     keybind: 'p' },
    { tool: 'magnetic',  label: 'Magnetic',icon: <MagnetStraight size={18} />, keybind: 'm' },
    { tool: 'magic',     label: 'Magic',   icon: <MagicWand size={18} />,   keybind: 'g' },
    { tool: 'rectangle', label: 'Rect',    icon: <Rectangle size={18} />,   keybind: 'e' },
    { tool: 'ellipse',   label: 'Ellipse', icon: <Circle size={18} />,      keybind: 'l' },
    { tool: 'brush',     label: 'Brush',   icon: <PaintBrush size={18} />,  keybind: 'b' },
    { tool: 'fill',      label: 'Fill',    icon: <PaintBucket size={18} />, keybind: 'f' },
    { tool: 'eraser',    label: 'Eraser',  icon: <Eraser size={18} />,      keybind: 'r' },
  ];

  return (
    <div className="flex flex-col gap-2">
      {/* Undo / redo */}
      <div className="flex gap-1">
        <button
          type="button"
          onClick={() => editHistory.undo()}
          disabled={!canUndo}
          title="Undo (Ctrl/⌘+Z)"
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
          onClick={() => editHistory.redo()}
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
      {disabled && (
        <p className="text-[11px] text-amber-600 bg-amber-50 border border-amber-200 rounded px-2 py-1 leading-snug">
          Add a class above to start annotating.
        </p>
      )}
      <div role="radiogroup" aria-label="Drawing tools" className="grid grid-cols-2 gap-1">
        {tools.map((t) => (
          <ToolButton
            key={t.tool}
            {...t}
            activeTool={tool}
            disabled={disabled}
            onSelect={setTool}
          />
        ))}
      </div>

      {/* Extra (non-tool) shortcuts */}
      <div className="text-[10px] leading-relaxed text-gray-500">
        <span className="font-mono font-semibold text-gray-500">Space</span> pan (hold) ·{' '}
        <span className="font-mono font-semibold text-gray-500">X</span> next slice ·{' '}
        <span className="font-mono font-semibold text-gray-500">T</span> fit ·{' '}
        <span className="font-mono font-semibold text-gray-500">⌘/Ctrl+Z</span> undo
      </div>

      {(tool === 'brush' || tool === 'eraser') && (
        <div className="flex flex-col gap-1 mt-1">
          <label className="text-xs text-gray-500">Brush radius (px)</label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={1}
              max={500}
              value={brushSize}
              onChange={(e) => setBrushSize(Math.max(1, Number(e.target.value)))}
              className="flex-1 min-w-0"
              aria-label="Brush radius"
            />
            <input
              type="number"
              min={1}
              max={500}
              value={brushSize}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (Number.isFinite(n)) setBrushSize(Math.min(500, Math.max(1, Math.round(n))));
              }}
              className="w-16 flex-shrink-0 border border-gray-200 rounded px-1 py-0.5 text-xs text-right tabular-nums"
              aria-label="Brush radius (px)"
            />
          </div>
          {tool === 'eraser' && (
            <div role="radiogroup" aria-label="Erase scope" className="flex flex-col gap-1 mt-1">
              {([['class', 'Erase selected class'], ['all', 'Erase all classes']] as const).map(([v, label]) => (
                <label key={v} className="flex items-center gap-2 text-[11px] text-gray-600 cursor-pointer select-none">
                  <input
                    type="radio"
                    name="erase-scope"
                    checked={eraseAllClasses === (v === 'all')}
                    onChange={() => setEraseAllClasses(v === 'all')}
                    className="accent-sky-600"
                  />
                  {label}
                </label>
              ))}
            </div>
          )}
        </div>
      )}

      {tool === 'select' && (
        <div className="flex flex-col gap-1 mt-1">
          <label className="text-xs text-gray-500">Select all (⌘/Ctrl+A)</label>
          <div role="radiogroup" aria-label="Select scope" className="flex flex-col gap-1">
            {([['class', 'Select this class'], ['all', 'Select all classes']] as const).map(([v, label]) => (
              <label key={v} className="flex items-center gap-2 text-[11px] text-gray-600 cursor-pointer select-none">
                <input
                  type="radio"
                  name="select-scope"
                  checked={selectScope === v}
                  onChange={() => setSelectScope(v)}
                  className="accent-sky-600"
                />
                {label}
              </label>
            ))}
          </div>
        </div>
      )}

      {tool === 'fill' && (
        <div className="flex flex-col gap-1 mt-1">
          <DebouncedSlider
            label="Fill threshold"
            format={(v) => `${v}%`}
            min={1}
            max={100}
            value={Math.round(fillThreshold * 100)}
            onChange={(v) => setFillThreshold(v / 100)}
            allowLog
          />
          <p className="text-[10px] text-gray-500 leading-snug">
            Click a region to flood-fill pixels within this intensity threshold of the clicked point.
          </p>
        </div>
      )}

      {tool === 'magic' && (
        <div className="flex flex-col gap-2 mt-1">
          {/* Engine selector + status badge */}
          <div className="grid grid-cols-2 gap-1">
            {([['sam', 'Smart (AI)'], ['classic', 'Classic']] as const).map(([eng, label]) => (
              <button
                key={eng}
                type="button"
                onClick={() => setMagicEngine(eng)}
                disabled={eng === 'sam' && !sam.supported}
                className={cn(
                  'py-1 rounded-md text-xs border transition-colors',
                  magicEngine === eng
                    ? 'bg-sky-600 text-white border-sky-700'
                    : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 disabled:opacity-40 disabled:cursor-not-allowed',
                )}
                title={eng === 'sam'
                  ? 'Segment Anything — click an object, no tuning'
                  : 'Threshold flood-fill — pick by brightness similarity'}
              >
                {label}
              </button>
            ))}
          </div>

          {magicEngine === 'sam' ? (
            <>
              <p className="text-[10px] text-gray-500 leading-snug">
                {sam.status === 'loading-model'
                  ? 'Loading model… (first time only)'
                  : sam.status === 'encoding'
                    ? 'Encoding slice…'
                    : sam.status === 'ready'
                      ? `SAM ready · ${sam.backend === 'webgpu' ? 'WebGPU' : 'CPU'}`
                      : sam.webgpu
                        ? 'Drag a box around an object, or click it.'
                        : 'No WebGPU — SAM will run on CPU (slower).'}
              </p>
              <label className="text-xs text-gray-500">Detail</label>
              <div className="grid grid-cols-4 gap-1">
                {(['auto', 'fine', 'medium', 'coarse'] as const).map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setSamDetail(d)}
                    className={cn(
                      'py-1 rounded-md text-[11px] border transition-colors capitalize',
                      samDetail === d
                        ? 'bg-sky-600 text-white border-sky-700'
                        : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50',
                    )}
                    title={d === 'auto'
                      ? 'Highest-confidence mask'
                      : `Pick SAM's ${d} mask (${d === 'fine' ? 'smallest' : d === 'coarse' ? 'largest' : 'mid'} region)`}
                  >
                    {d}
                  </button>
                ))}
              </div>
              <DebouncedSlider
                label="Tightness"
                format={(v) => (v === 0 ? '0' : `${v > 0 ? '+' : ''}${v.toFixed(1)}`)}
                min={-6}
                max={6}
                step={0.5}
                value={samThreshold}
                onChange={setSamThreshold}
              />
              <DebouncedSlider
                label="Edge smoothing"
                format={(v) => v.toFixed(1)}
                min={0}
                max={5}
                step={0.5}
                value={magicSigma}
                onChange={setMagicSigma}
              />
              <label className="flex items-start gap-2 text-[11px] text-gray-600 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={samAvoidLabeled}
                  onChange={(e) => setSamAvoidLabeled(e.target.checked)}
                  className="mt-0.5 accent-sky-600"
                />
                <span>
                  Avoid other-class regions
                  <span className="block text-[10px] text-gray-500 leading-snug">
                    Uses nearby regions of other classes as "not" hints (faint red rings) so a new
                    selection won't bleed into them.
                  </span>
                </span>
              </label>
              <label className="flex items-start gap-2 text-[11px] text-gray-600 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={samConnectedOnly}
                  onChange={(e) => setSamConnectedOnly(e.target.checked)}
                  className="mt-0.5 accent-sky-600"
                />
                <span>
                  Connected regions only
                  <span className="block text-[10px] text-gray-500 leading-snug">
                    Keeps only the region at your click/box and drops detached speckle.
                  </span>
                </span>
              </label>
              <p className="text-[10px] text-gray-500 leading-snug">
                Drag a box around the object (most reliable), or click it. Shift-click adds to the
                object; <b>{REMOVE_KEY_LABEL}-click drops a "not" point (red)</b> to remove an area
                SAM grabbed by mistake. Grabs too much? Lower <b>Detail</b> or raise <b>Tightness</b>.
                Adjust Display brightness/contrast to re-encode what SAM sees.
              </p>
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-1">
                {(['contiguous', 'global'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMagicMode(m)}
                    className={cn(
                      'py-1 rounded-md text-xs border transition-colors',
                      magicMode === m
                        ? 'bg-sky-600 text-white border-sky-700'
                        : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50',
                    )}
                    title={m === 'contiguous' ? 'Select the connected region you click' : 'Select all similar regions on the slice'}
                  >
                    {m === 'contiguous' ? 'Connected' : 'All similar'}
                  </button>
                ))}
              </div>
              <DebouncedSlider
                label="Tolerance"
                format={(v) => `${v}%`}
                min={1}
                max={60}
                value={Math.round(magicTolerance * 100)}
                onChange={(v) => setMagicTolerance(v / 100)}
                allowLog
              />
              {magicMode === 'contiguous' && (
                <DebouncedSlider
                  label="Edge stop"
                  format={(v) => `${v}%`}
                  min={0}
                  max={100}
                  value={Math.round(magicEdgeStop * 100)}
                  onChange={(v) => setMagicEdgeStop(v / 100)}
                />
              )}
              <DebouncedSlider
                label="Edge smoothing"
                format={(v) => v.toFixed(1)}
                min={0}
                max={5}
                step={0.5}
                value={magicSigma}
                onChange={setMagicSigma}
              />
              <p className="text-[10px] text-gray-500 leading-snug">
                Click a region on the image. For voids that leak, raise <b>Edge stop</b>; for
                low-contrast scans, adjust contrast (Display) first.
              </p>
            </>
          )}
        </div>
      )}

      <label
        className="flex items-center gap-2 text-[11px] text-gray-600 cursor-pointer mt-1"
        title="New annotations won't overlap regions already labeled with a different class on this slice"
      >
        <input
          type="checkbox"
          checked={clipToOtherClasses}
          onChange={(e) => setClipToOtherClasses(e.target.checked)}
          className="accent-sky-600"
        />
        Clip to other classes
      </label>

      <label
        className="flex items-center gap-2 text-[11px] text-gray-600 cursor-pointer"
        title="A new annotation overlapping existing regions of the same class is merged with them into one shape"
      >
        <input
          type="checkbox"
          checked={mergeOverlappingSameClass}
          onChange={(e) => setMergeOverlappingSameClass(e.target.checked)}
          className="accent-sky-600"
        />
        Merge overlapping same class
      </label>
    </div>
  );
}
