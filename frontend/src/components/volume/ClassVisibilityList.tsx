/**
 * ClassVisibilityList — per-class visibility checkboxes for the 3D volume
 * view's sidebar. Distinct from Annotate's `ClassManager` (no add/edit/
 * delete here, just show/hide, since this list only controls which label
 * colors the volume renderer includes when compositing the label texture)
 * but mirrors its row layout: a color swatch, the label, and a control.
 */
import type { AnnotationClass } from '@/stores/classStore';
import { cn } from '@/lib/utils';

export interface ClassVisibilityListProps {
  classes: AnnotationClass[];
  /** Keyed by classId; a class with no entry is treated as visible. */
  visible: Record<number, boolean>;
  onChange: (next: Record<number, boolean>) => void;
}

/** Renders one row per class with a checkbox toggling its 3D-view visibility,
 *  plus "All"/"None" shortcuts above the list. */
export default function ClassVisibilityList({ classes, visible, onChange }: ClassVisibilityListProps) {
  const setAll = (value: boolean) => {
    const next: Record<number, boolean> = {};
    for (const cls of classes) next[cls.classId] = value;
    onChange(next);
  };

  if (classes.length === 0) {
    return <p className="text-xs text-gray-500">No classes yet — create them in Annotate.</p>;
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Classes</span>
        <div className="flex gap-1">
          <button
            type="button"
            className="text-xs px-1.5 py-0.5 rounded hover:bg-gray-100 text-gray-600"
            onClick={() => setAll(true)}
          >
            All
          </button>
          <button
            type="button"
            className="text-xs px-1.5 py-0.5 rounded hover:bg-gray-100 text-gray-600"
            onClick={() => setAll(false)}
          >
            None
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-0.5">
        {classes.map((cls) => {
          const isVisible = visible[cls.classId] ?? true;
          return (
            <label
              key={cls.classId}
              className={cn(
                'flex items-center gap-1.5 px-2 py-1.5 rounded-md cursor-pointer select-none min-w-0',
                'hover:bg-gray-100'
              )}
            >
              <span
                className="w-3 h-3 rounded-full flex-shrink-0 border border-black/10"
                style={{ backgroundColor: cls.color }}
              />
              <span className="flex-1 min-w-0 text-sm truncate">{cls.label}</span>
              <input
                type="checkbox"
                aria-label={isVisible ? `Hide ${cls.label}` : `Show ${cls.label}`}
                checked={isVisible}
                onChange={() => onChange({ ...visible, [cls.classId]: !isVisible })}
                className="h-3.5 w-3.5 cursor-pointer accent-sky-600 flex-shrink-0"
              />
            </label>
          );
        })}
      </div>
    </div>
  );
}
