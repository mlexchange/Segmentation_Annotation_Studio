/**
 * ClassManager — add/edit/delete/hide annotation classes.
 * Color and label deduplication enforced.
 * Deleting a class removes all of its annotations (with confirmation).
 */
import { useState } from 'react';
import { Eye, EyeSlash, Pencil, Trash, Plus, Info, Eyedropper, Check } from '@phosphor-icons/react';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useReferenceGuideStore, type GuideClass } from '@/stores/referenceGuideStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { getClassPalette } from '@/lib/classColors';
import { cn } from '@/lib/utils';

/** Counts all shapes assigned to a class across every image/slice in the annotation store. */
function countShapesForClass(classId: number): number {
  const { byImage } = useAnnotationStore.getState();
  let total = 0;
  for (const slices of Object.values(byImage)) {
    for (const shapes of Object.values(slices)) {
      total += shapes.filter((sh) => sh.classId === classId).length;
    }
  }
  return total;
}

interface ClassRowProps {
  cls: AnnotationClass;
  isActive: boolean;
  onActivate: () => void;
  onClassDeleted: (deletedClassId: number) => void;
  /** Matching guide entry (description + example crops), if the guide defines this class. */
  guide?: GuideClass;
}

/** Renders a single class row with inline rename, visibility toggle, and delete. */
function ClassRow({ cls, isActive, onActivate, onClassDeleted, guide }: ClassRowProps) {
  const { updateClass, deleteClass, toggleVisibility } = useClassStore();
  const removeShapesByClassId = useAnnotationStore((s) => s.removeShapesByClassId);
  const setSelectedShapeId = useToolStore((s) => s.setSelectedShapeId);
  const [editing, setEditing] = useState(false);
  const [showGuide, setShowGuide] = useState(false);
  const [labelInput, setLabelInput] = useState(cls.label);
  const hasGuide = !!guide && (!!guide.description || guide.exampleCrops.length > 0);

  /** Saves the edited label (if non-empty and changed) without leaving edit mode. */
  const commitLabel = () => {
    const trimmed = labelInput.trim();
    if (trimmed && trimmed !== cls.label) updateClass(cls.classId, { label: trimmed });
  };

  /** Commits the label and exits edit mode. */
  const finishEditing = () => {
    commitLabel();
    setEditing(false);
  };

  /** Discards any label edit and exits edit mode (color changes are applied live). */
  const cancelEditing = () => {
    setLabelInput(cls.label);
    setEditing(false);
  };

  /** Confirms with the user, then removes the class and all of its shapes from the stores. */
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    const shapeCount = countShapesForClass(cls.classId);
    const annotationNote =
      shapeCount === 0
        ? 'This class has no annotations.'
        : shapeCount === 1
          ? 'This will permanently delete 1 annotation.'
          : `This will permanently delete ${shapeCount} annotations.`;

    const confirmed = window.confirm(
      `Do you really want to delete "${cls.label}" and its annotations?\n\n${annotationNote}`
    );
    if (!confirmed) return;

    removeShapesByClassId(cls.classId);
    deleteClass(cls.classId);
    setSelectedShapeId(null);
    onClassDeleted(cls.classId);
  };

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer select-none',
          isActive ? 'bg-sky-100 dark:bg-sky-900' : 'hover:bg-gray-100 dark:hover:bg-gray-800'
        )}
        onClick={onActivate}
      >
        {editing ? (
          <input
            type="color"
            aria-label="Class color"
            title="Change class color"
            value={cls.color}
            onChange={(e) => updateClass(cls.classId, { color: e.target.value })}
            onClick={(e) => e.stopPropagation()}
            className="w-4 h-4 flex-shrink-0 cursor-pointer rounded-sm border border-black/10 bg-transparent p-0"
          />
        ) : (
          <span
            className="w-4 h-4 rounded-sm flex-shrink-0 border border-black/10"
            style={{ backgroundColor: cls.color }}
          />
        )}
        {editing ? (
          <input
            autoFocus
            className="flex-1 text-sm border rounded px-1 py-0.5"
            value={labelInput}
            onChange={(e) => setLabelInput(e.target.value)}
            onBlur={commitLabel}
            onKeyDown={(e) => { if (e.key === 'Enter') finishEditing(); if (e.key === 'Escape') cancelEditing(); }}
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className="flex-1 text-sm truncate">{cls.label}</span>
        )}
        {hasGuide && (
          <button
            aria-label="Show annotation guide for this class"
            aria-pressed={showGuide}
            className={cn('p-0.5 hover:text-sky-600', showGuide && 'text-sky-600')}
            onClick={(e) => { e.stopPropagation(); setShowGuide((v) => !v); }}
          >
            <Info size={14} />
          </button>
        )}
        <button
          aria-pressed={cls.isVisible}
          aria-label={cls.isVisible ? 'Hide class' : 'Show class'}
          className="p-0.5 hover:text-sky-600"
          onClick={(e) => { e.stopPropagation(); toggleVisibility(cls.classId); }}
        >
          {cls.isVisible ? <Eye size={14} /> : <EyeSlash size={14} className="text-gray-400" />}
        </button>
        <button
          aria-label={editing ? 'Finish editing class' : 'Edit class label and color'}
          aria-pressed={editing}
          title={editing ? 'Done' : 'Edit label and color'}
          className={cn('p-0.5 hover:text-sky-600', editing && 'text-sky-600')}
          onClick={(e) => {
            e.stopPropagation();
            if (editing) { finishEditing(); }
            else { setLabelInput(cls.label); setEditing(true); }
          }}
        >
          {editing ? <Check size={14} /> : <Pencil size={14} />}
        </button>
        <button
          aria-label="Delete class and its annotations"
          className="p-0.5 hover:text-red-500"
          onClick={handleDelete}
        >
          <Trash size={14} />
        </button>
      </div>

      {hasGuide && showGuide && guide && (
        <div className="mx-2 mt-1 mb-1 rounded-md border border-sky-100 bg-sky-50/60 p-2 text-xs text-gray-600">
          {guide.description && <p className="whitespace-pre-wrap">{guide.description}</p>}
          {guide.exampleCrops.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {guide.exampleCrops.map((src, i) => (
                <img
                  key={i}
                  src={src}
                  alt={`${cls.label} example ${i + 1}`}
                  className="h-12 w-12 rounded border border-gray-200 object-cover"
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export interface ClassManagerProps {
  activeClassId: number | null;
  onActivate: (classId: number) => void;
  onClassDeleted?: (deletedClassId: number) => void;
}

/** Fallback quick-add chips when the dataset has no annotation guide. */
const DEFAULT_SUGGESTED_CLASSES = ['air', 'sample', 'void', 'pore', 'background', 'substrate'];

/** Renders the class list, add form, and quick-add suggestion chips. */
export default function ClassManager({ activeClassId, onActivate, onClassDeleted }: ClassManagerProps) {
  const { classes, addClass } = useClassStore();
  const guideEntries = useReferenceGuideStore((s) => s.entries);
  const colorblindMode = useSettingsStore((s) => s.colorblindMode);
  const setColorblindMode = useSettingsStore((s) => s.setColorblindMode);
  const [showAdd, setShowAdd] = useState(false);
  const [newLabel, setNewLabel] = useState('');
  const [newColor, setNewColor] = useState('');

  // Toggle the colorblind-safe palette preference. This only affects the color
  // auto-assigned to NEWLY added classes; existing classes keep their colors so
  // we never silently rewrite already-annotated datasets.
  const handleToggleColorblind = () => setColorblindMode(!colorblindMode);

  // Guide-defined classes drive suggestions + colors so annotators stay consistent
  // with the lead's intended labels; fall back to generic defaults when no guide.
  const guideByLabel = new Map(
    guideEntries.filter((g) => g.label.trim()).map((g) => [g.label.trim().toLowerCase(), g]),
  );

  /** Notifies the parent of a deletion and re-activates the first remaining class if the active one was removed. */
  const handleClassDeleted = (deletedClassId: number) => {
    onClassDeleted?.(deletedClassId);
    if (activeClassId === deletedClassId) {
      const remaining = useClassStore.getState().classes;
      if (remaining.length > 0) {
        onActivate(remaining[0].classId);
      }
    }
  };

  /** Returns the first unused color from the active palette, or cycles back if all are taken. */
  const nextColor = () => {
    const palette = getClassPalette();
    const used = new Set(classes.map((c) => c.color));
    return palette.find((c) => !used.has(c)) ?? palette[classes.length % palette.length];
  };

  /** Validates the new-class form (rejecting duplicate labels), adds the class, and activates it. */
  const handleAdd = () => {
    const label = newLabel.trim();
    if (!label) return;
    const color = newColor || nextColor();
    const dupLabel = classes.some((c) => c.label.toLowerCase() === label.toLowerCase());
    const dupColor = classes.some((c) => c.color === color);
    if (dupLabel) { alert('A class with that label already exists.'); return; }
    if (dupColor && !newColor) {
      // silently pick another
    }
    const classId = addClass(label, color || nextColor());
    onActivate(classId);
    setNewLabel('');
    setNewColor('');
    setShowAdd(false);
  };

  /** One-click add (or re-activate) a suggested class. Inherits the guide color if
   *  defined, unless colorblind mode is on — then the colorblind-safe palette wins. */
  const handleQuickAdd = (label: string) => {
    const existing = classes.find((c) => c.label.toLowerCase() === label.toLowerCase());
    if (existing) { onActivate(existing.classId); return; }
    const guideColor = guideByLabel.get(label.toLowerCase())?.color;
    const color = colorblindMode ? nextColor() : (guideColor || nextColor());
    const classId = addClass(label, color);
    onActivate(classId);
  };

  // Prefer the guide's classes (in guide order); otherwise the generic defaults.
  const suggestionLabels =
    guideByLabel.size > 0
      ? guideEntries.map((g) => g.label.trim()).filter(Boolean)
      : DEFAULT_SUGGESTED_CLASSES;
  const suggestions = suggestionLabels.filter(
    (label) => !classes.some((c) => c.label.toLowerCase() === label.toLowerCase()),
  );

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Classes</span>
        <button
          aria-label="Add class"
          className="p-0.5 rounded hover:bg-sky-100 hover:text-sky-700"
          onClick={() => { setShowAdd((v) => !v); setNewLabel(''); setNewColor(nextColor()); }}
        >
          <Plus size={14} />
        </button>
      </div>

      <label
        className="flex items-center gap-1.5 mb-1 text-xs text-gray-600 cursor-pointer select-none"
        title="Color newly added classes from a colorblind-safe palette (Okabe–Ito). Existing classes keep their colors."
      >
        <input
          type="checkbox"
          checked={colorblindMode}
          onChange={handleToggleColorblind}
          className="h-3.5 w-3.5 cursor-pointer accent-sky-600"
        />
        <Eyedropper size={13} className={colorblindMode ? 'text-sky-600' : 'text-gray-400'} />
        <span>Colorblind-safe colors</span>
      </label>

      {showAdd && (
        <div className="flex flex-col gap-1 bg-gray-50 rounded p-2 mb-1 text-sm">
          <input
            autoFocus
            placeholder="Class label"
            className="border rounded px-2 py-1 text-sm"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleAdd(); }}
          />
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500">Color</label>
            <input
              type="color"
              value={newColor || nextColor()}
              onChange={(e) => setNewColor(e.target.value)}
              className="w-8 h-6 cursor-pointer"
            />
          </div>
          <div className="flex gap-1 justify-end">
            <button className="text-xs px-2 py-1 rounded hover:bg-gray-200" onClick={() => setShowAdd(false)}>Cancel</button>
            <button className="text-xs px-2 py-1 rounded bg-sky-600 text-white hover:bg-sky-700" onClick={handleAdd}>Add</button>
          </div>
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 mb-1">
          <span className="text-[10px] uppercase tracking-wide text-gray-400 mr-0.5">Quick add</span>
          {suggestions.map((label) => (
            <button
              key={label}
              type="button"
              onClick={() => handleQuickAdd(label)}
              title={guideByLabel.get(label.toLowerCase())?.description || undefined}
              className="flex items-center gap-0.5 text-xs px-1.5 py-0.5 rounded-full border border-gray-200 text-gray-600 hover:bg-sky-50 hover:border-sky-300 hover:text-sky-700 transition-colors"
            >
              <Plus size={10} />
              {label}
            </button>
          ))}
        </div>
      )}

      <div role="listbox" aria-label="Annotation classes" className="flex flex-col gap-0.5">
        {classes.length === 0 && (
          <p className="text-xs text-gray-400 text-center py-2">No classes yet. Click + to add one.</p>
        )}
        {classes.map((cls) => (
          <ClassRow
            key={cls.classId}
            cls={cls}
            isActive={cls.classId === activeClassId}
            onActivate={() => onActivate(cls.classId)}
            onClassDeleted={handleClassDeleted}
            guide={guideByLabel.get(cls.label.trim().toLowerCase())}
          />
        ))}
      </div>
    </div>
  );
}
