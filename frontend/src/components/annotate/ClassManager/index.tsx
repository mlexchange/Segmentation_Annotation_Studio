/**
 * ClassManager — add/edit/delete/hide annotation classes.
 * Color and label deduplication enforced.
 * Deleting a class removes all of its annotations (with confirmation).
 */
import { useState } from 'react';
import { Eye, EyeSlash, Pencil, Trash, Plus } from '@phosphor-icons/react';
import { useClassStore, DEFAULT_COLORS, type AnnotationClass } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { cn } from '@/lib/utils';

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
}

function ClassRow({ cls, isActive, onActivate, onClassDeleted }: ClassRowProps) {
  const { updateClass, deleteClass, toggleVisibility } = useClassStore();
  const removeShapesByClassId = useAnnotationStore((s) => s.removeShapesByClassId);
  const setSelectedShapeId = useToolStore((s) => s.setSelectedShapeId);
  const [editing, setEditing] = useState(false);
  const [labelInput, setLabelInput] = useState(cls.label);

  const commitLabel = () => {
    const trimmed = labelInput.trim();
    if (trimmed) updateClass(cls.classId, { label: trimmed });
    setEditing(false);
  };

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
    <div
      className={cn(
        'flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer select-none',
        isActive ? 'bg-sky-100 dark:bg-sky-900' : 'hover:bg-gray-100 dark:hover:bg-gray-800'
      )}
      onClick={onActivate}
    >
      <span
        className="w-4 h-4 rounded-sm flex-shrink-0 border border-black/10"
        style={{ backgroundColor: cls.color }}
      />
      {editing ? (
        <input
          autoFocus
          className="flex-1 text-sm border rounded px-1 py-0.5"
          value={labelInput}
          onChange={(e) => setLabelInput(e.target.value)}
          onBlur={commitLabel}
          onKeyDown={(e) => { if (e.key === 'Enter') commitLabel(); if (e.key === 'Escape') setEditing(false); }}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <span className="flex-1 text-sm truncate">{cls.label}</span>
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
        aria-label="Edit class label"
        className="p-0.5 hover:text-sky-600"
        onClick={(e) => { e.stopPropagation(); setEditing(true); setLabelInput(cls.label); }}
      >
        <Pencil size={14} />
      </button>
      <button
        aria-label="Delete class and its annotations"
        className="p-0.5 hover:text-red-500"
        onClick={handleDelete}
      >
        <Trash size={14} />
      </button>
    </div>
  );
}

export interface ClassManagerProps {
  activeClassId: number | null;
  onActivate: (classId: number) => void;
  onClassDeleted?: (deletedClassId: number) => void;
}

export default function ClassManager({ activeClassId, onActivate, onClassDeleted }: ClassManagerProps) {
  const { classes, addClass } = useClassStore();
  const [showAdd, setShowAdd] = useState(false);
  const [newLabel, setNewLabel] = useState('');
  const [newColor, setNewColor] = useState('');

  const handleClassDeleted = (deletedClassId: number) => {
    onClassDeleted?.(deletedClassId);
    if (activeClassId === deletedClassId) {
      const remaining = useClassStore.getState().classes;
      if (remaining.length > 0) {
        onActivate(remaining[0].classId);
      }
    }
  };

  const nextColor = () => {
    const used = new Set(classes.map((c) => c.color));
    return DEFAULT_COLORS.find((c) => !used.has(c)) ?? DEFAULT_COLORS[classes.length % DEFAULT_COLORS.length];
  };

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
          />
        ))}
      </div>
    </div>
  );
}
