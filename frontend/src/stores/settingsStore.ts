/**
 * Settings store — machine-local user preferences, persisted to localStorage.
 *
 * `annotatorName` identifies who is doing the annotating. There is no login;
 * this name defaults into saved-version metadata and is stamped into COCO
 * exports so downloads are self-identifying for external inter-annotator
 * agreement analysis (each annotator runs their own local install).
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsStore {
  annotatorName: string;
  setAnnotatorName: (name: string) => void;
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      annotatorName: '',
      setAnnotatorName: (name) => set({ annotatorName: name }),
    }),
    { name: 'sam3_settings' },
  ),
);
