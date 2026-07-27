/**
 * Class store — annotation classes (add/edit/delete/hide).
 */
import { create } from 'zustand';

// Re-exported for backwards compatibility; palettes now live in lib/classColors.
export { DEFAULT_COLORS } from '@/lib/classColors';

export interface AnnotationClass {
  classId: number;
  label: string;
  color: string;
  isVisible: boolean;
}

interface ClassStore {
  classes: AnnotationClass[];
  addClass: (label: string, color: string) => number;
  updateClass: (classId: number, updates: Partial<Omit<AnnotationClass, 'classId'>>) => void;
  deleteClass: (classId: number) => void;
  insertClass: (cls: AnnotationClass, index: number) => void;
  toggleVisibility: (classId: number) => void;
  setClasses: (classes: AnnotationClass[]) => void;
  remapColors: (palette: string[]) => void;
}

export const useClassStore = create<ClassStore>((set, get) => ({
  classes: [],
  /** Appends a new visible class with an auto-assigned id (max existing + 1); returns the new classId. */
  addClass: (label, color) => {
    const { classes } = get();
    const existingIds = classes.map((c) => c.classId);
    const classId = existingIds.length ? Math.max(...existingIds) + 1 : 1;
    set({ classes: [...classes, { classId, label, color, isVisible: true }] });
    return classId;
  },
  /** Merges partial updates (label/color/visibility) into the matching class. */
  updateClass: (classId, updates) =>
    set((s) => ({
      classes: s.classes.map((c) =>
        c.classId === classId ? { ...c, ...updates } : c
      ),
    })),
  /** Removes the class with the given id (does not touch existing shapes). */
  deleteClass: (classId) =>
    set((s) => ({ classes: s.classes.filter((c) => c.classId !== classId) })),
  /** Re-inserts a previously-removed class at `index`, preserving its original classId
   *  (used to undo a delete). No-op if a class with that id already exists. */
  insertClass: (cls, index) =>
    set((s) => {
      if (s.classes.some((c) => c.classId === cls.classId)) return s;
      const next = [...s.classes];
      next.splice(Math.max(0, Math.min(index, next.length)), 0, cls);
      return { classes: next };
    }),
  /** Flips the class's isVisible flag (controls whether its shapes render). */
  toggleVisibility: (classId) =>
    set((s) => ({
      classes: s.classes.map((c) =>
        c.classId === classId ? { ...c, isVisible: !c.isVisible } : c
      ),
    })),
  /** Replaces the entire class list (used when loading a saved set). */
  setClasses: (classes) => set({ classes }),
  /** Reassigns every class's color from `palette` by list order (cycling if needed). */
  remapColors: (palette) =>
    set((s) => ({
      classes: palette.length
        ? s.classes.map((c, i) => ({ ...c, color: palette[i % palette.length] }))
        : s.classes,
    })),
}));
