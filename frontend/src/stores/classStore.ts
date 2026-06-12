/**
 * Class store — annotation classes (add/edit/delete/hide).
 */
import { create } from 'zustand';

export interface AnnotationClass {
  classId: number;
  label: string;
  color: string;
  isVisible: boolean;
}

/** Default color palette seeded from ALS design tokens / mlex Dark24 fallback */
export const DEFAULT_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
];

interface ClassStore {
  classes: AnnotationClass[];
  addClass: (label: string, color: string) => number;
  updateClass: (classId: number, updates: Partial<Omit<AnnotationClass, 'classId'>>) => void;
  deleteClass: (classId: number) => void;
  toggleVisibility: (classId: number) => void;
  setClasses: (classes: AnnotationClass[]) => void;
}

export const useClassStore = create<ClassStore>((set, get) => ({
  classes: [],
  addClass: (label, color) => {
    const { classes } = get();
    const existingIds = classes.map((c) => c.classId);
    const classId = existingIds.length ? Math.max(...existingIds) + 1 : 1;
    set({ classes: [...classes, { classId, label, color, isVisible: true }] });
    return classId;
  },
  updateClass: (classId, updates) =>
    set((s) => ({
      classes: s.classes.map((c) =>
        c.classId === classId ? { ...c, ...updates } : c
      ),
    })),
  deleteClass: (classId) =>
    set((s) => ({ classes: s.classes.filter((c) => c.classId !== classId) })),
  toggleVisibility: (classId) =>
    set((s) => ({
      classes: s.classes.map((c) =>
        c.classId === classId ? { ...c, isVisible: !c.isVisible } : c
      ),
    })),
  setClasses: (classes) => set({ classes }),
}));
