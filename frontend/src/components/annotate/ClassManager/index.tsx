/**
 * ClassManager — add/edit/delete/hide annotation classes.
 * Color and label deduplication enforced.
 */
import { useState } from 'react';
import { Eye, EyeSlash, Pencil, Trash, Plus } from '@phosphor-icons/react';
import { useClassStore, DEFAULT_COLORS, type AnnotationClass } from '@/stores/classStore';
import { cn } from '@/lib/utils';

interface ClassRowProps {
  cls: AnnotationClass;
  isActive: boolean;
  onActivate: () => void;
}

function ClassRow({ cls, isActive, onActivate }: ClassRowProps) {
  const { updateClass, deleteClass, toggleVisibility } = useClassStore();
  const [editing, setEditing] = useState(false);
  const [labelInput, setLabelInput] = useState(cls.label);

  const commitLabel = () => {
    const trimmed = labelInput.trim();
    if (trimmed) updateClass(cls.classId, { label: trimmed });
    setEditing(false);
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
        aria-label="Delete class"
        className="p-0.5 hover:text-red-500"
        onClick={(e) => {
          e.stopPropagation();
          if (window.confirm(`Delete class "${cls.label}"? Its shapes will be removed.`)) {
            deleteClass(cls.classId);
          }
        }}
      >
        <Trash size={14} />
      </button>
    </div>
  );
}

export interface ClassManagerProps {
  activeClassId: number | null;
  onActivate: (classId: number) => void;
}

export default function ClassManager({ activeClassId, onActivate }: ClassManagerProps) {
  const { classes, addClass } = useClassStore();
  const [showAdd, setShowAdd] = useState(false);
  const [newLabel, setNewLabel] = useState('');
  const [newColor, setNewColor] = useState('');

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
    addClass(label, color || nextColor());
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
          />
        ))}
      </div>
    </div>
  );
}
